import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
    def forward(self, x):
        skip = self.conv1(x)
        skip = self.conv2(skip)
        down = self.pool(skip)
        return down, skip

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        
    def forward(self, x, skip):
        x = self.up(x)
        
        # 确保尺寸匹配
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, 
                      diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([skip, x], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class AECUNet(nn.Module):
    def __init__(self, input_channels=None, output_channels=None, freq_bins=None):
        """
        回声消除U-Net模型
        
        参数:
            input_channels (int, optional): 输入通道数，如果为None则使用config中的值
            output_channels (int, optional): 输出通道数，如果为None则使用config中的值
            freq_bins (int, optional): 频率维度大小，如果为None则使用config中的值
        """
        super(AECUNet, self).__init__()
        self.input_channels = input_channels if input_channels is not None else config['input_channels']
        self.output_channels = output_channels if output_channels is not None else config['output_channels']
        self.freq_bins = freq_bins if freq_bins is not None else config['freq_bins']
        
        # 初始卷积层
        self.input_conv = ConvBlock(self.input_channels, 32)
        
        # 下采样路径
        self.down1 = DownBlock(32, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            nn.Dropout2d(0.5)
        )
        
        # 上采样路径
        self.up1 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        
        # 输出层
        self.output_conv = nn.Conv2d(64, self.output_channels, kernel_size=1)

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=256 * 2,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (Tensor): 输入特征 [batch_size, input_channels, freq_bins, time_frames]
        
        返回:
            Tensor: 输出特征 [batch_size, output_channels, freq_bins, time_frames]
        """
        # 输入形状已经是 [batch_size, input_channels, freq_bins, time_frames]
        
        # 编码器路径
        x = self.input_conv(x)
        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        
        # 瓶颈
        x = self.bottleneck(x3)
        batch_size, channels, freq_bins, time_frames = x.shape
        x = x.reshape(batch_size,-1,channels)
        x,_ = self.lstm(x)
        x = x.reshape(batch_size, channels, freq_bins, time_frames)
        
        # 解码器路径
        x = self.up1(x, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)
        
        # 输出层
        x = self.output_conv(x)
        
        return x

class AECLSTMNet(nn.Module):
    def __init__(self, input_channels=None, output_channels=None, freq_bins=None, hidden_size=None):
        """
        基于LSTM的回声消除模型
        
        参数:
            input_channels (int, optional): 输入通道数，如果为None则使用config中的值
            output_channels (int, optional): 输出通道数，如果为None则使用config中的值
            freq_bins (int, optional): 频率维度大小，如果为None则使用config中的值
            hidden_size (int, optional): LSTM隐藏层大小，如果为None则使用config中的值
        """
        super(AECLSTMNet, self).__init__()
        self.input_channels = input_channels if input_channels is not None else config['input_channels']
        self.output_channels = output_channels if output_channels is not None else config['output_channels']
        self.freq_bins = freq_bins if freq_bins is not None else config['freq_bins']
        self.hidden_size = hidden_size if hidden_size is not None else config['hidden_size']
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_channels * self.freq_bins, self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.hidden_size * 2,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_channels * self.freq_bins)
        )
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (Tensor): 输入特征 [batch_size, input_channels, freq_bins, time_frames]
        
        返回:
            Tensor: 输出特征 [batch_size, output_channels, freq_bins, time_frames]
        """
        batch_size, channels, freq_bins, time_frames = x.shape
        
        # 重塑为 [batch_size, time_frames, input_channels*freq_bins]
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, time_frames, self.input_channels * self.freq_bins)
        
        # 特征提取
        x = self.feature_extractor(x)
        
        # LSTM处理
        x, _ = self.lstm(x)
        
        # 输出层
        x = self.output_layer(x)
        
        # 重塑回 [batch_size, output_channels, freq_bins, time_frames]
        x = x.view(batch_size, time_frames, self.output_channels, self.freq_bins)
        x = x.permute(0, 2, 3, 1).contiguous()
        
        return x 