import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import random
from config import config

class PreprocessedAECDataset(Dataset):
    def __init__(self, data_dir, split='train', segment_length=None, transform=None):
        """
        预处理后的回声消除数据集
        
        参数:
            data_dir (str): 预处理后的数据目录路径
            split (str): 'train', 'val', 或 'test'
            segment_length (int, optional): 模型输入的时间步长度，如果为None则使用config中的值
            transform (callable, optional): 可选的数据变换
        """
        self.data_dir = data_dir
        self.split = split
        self.segment_length = segment_length if segment_length is not None else config['segment_length']
        self.transform = transform
        self.fs = config['sample_rate']  # 采样率
        self.fft_size = config['fft_size']  # DFT变换点数
        self.hop_size = config['hop_size']  # 帧移
        self.win_length = config['win_length']  # 窗长度
        self.freq_bins = config['freq_bins']  # 频率维度大小
        
        # 获取所有文件组ID
        self.file_groups = []
        farend_dir = os.path.join(data_dir, 'farend')
        for file_name in os.listdir(farend_dir):
            if file_name.endswith('.wav'):
                group_id = file_name.split('_')[0]  # 例如从f00001_farend.wav提取f00001
                self.file_groups.append(group_id)
        
        # 根据split划分数据集
        if split == 'train':
            self.file_groups = self.file_groups[:int(len(self.file_groups) * 0.8)]
        elif split == 'val':
            self.file_groups = self.file_groups[int(len(self.file_groups) * 0.8):int(len(self.file_groups) * 0.9)]
        else:  # test
            self.file_groups = self.file_groups[int(len(self.file_groups) * 0.9):]
    
    def __len__(self):
        return len(self.file_groups)
    
    def __getitem__(self, idx):
        group_id = self.file_groups[idx]
        
        # 加载预处理后的信号
        farend_path = os.path.join(self.data_dir, 'farend', f'{group_id}')
        mic_path = os.path.join(self.data_dir, 'mic', f'{group_id}')
        nearend_path = os.path.join(self.data_dir, 'nearend', f'{group_id}')
        error_path = os.path.join(self.data_dir, 'error', f'{group_id}')
        
        farend, _ = librosa.load(farend_path, sr=self.fs)
        mic, _ = librosa.load(mic_path, sr=self.fs)
        nearend, _ = librosa.load(nearend_path, sr=self.fs)
        error, _ = librosa.load(error_path, sr=self.fs)
        
        # 确保长度一致
        min_length = min(len(farend), len(mic), len(nearend), len(error))
        farend = farend[:min_length]
        mic = mic[:min_length]
        nearend = nearend[:min_length]
        error = error[:min_length]
        
        # 进行短时傅里叶变换(STFT)
        mic_stft = librosa.stft(mic, n_fft=self.fft_size, hop_length=self.hop_size, win_length=self.win_length)
        farend_stft = librosa.stft(farend, n_fft=self.fft_size, hop_length=self.hop_size, win_length=self.win_length)
        error_stft = librosa.stft(error, n_fft=self.fft_size, hop_length=self.hop_size, win_length=self.win_length)
        nearend_stft = librosa.stft(nearend, n_fft=self.fft_size, hop_length=self.hop_size, win_length=self.win_length)
        
        # 获取实部和虚部
        mic_real = np.real(mic_stft)
        mic_imag = np.imag(mic_stft)
        farend_real = np.real(farend_stft)
        farend_imag = np.imag(farend_stft)
        error_real = np.real(error_stft)
        error_imag = np.imag(error_stft)
        nearend_real = np.real(nearend_stft)
        nearend_imag = np.imag(nearend_stft)
        
        # 将特征按通道拼接，形状为[通道数，特征数，帧数]
        # 输入特征: [mic_real, mic_imag, farend_real, farend_imag, error_real, error_imag]
        # 输出特征: [nearend_real, nearend_imag]
        features = np.stack([
            mic_real, mic_imag, 
            farend_real, farend_imag, 
            error_real, error_imag
        ], axis=0)  # shape: [6, freq_bins, time_frames]
        
        targets = np.stack([
            nearend_real, nearend_imag
        ], axis=0)  # shape: [2, freq_bins, time_frames]
        
        # 获取总时间帧数
        total_frames = features.shape[2]
        
        # 如果总长度不足，则进行0填充
        if total_frames < self.segment_length:
            pad_length = self.segment_length - total_frames
            features = np.pad(features, ((0, 0), (0, 0), (0, pad_length)), 'constant')
            targets = np.pad(targets, ((0, 0), (0, 0), (0, pad_length)), 'constant')
            # 随机返回整个序列
            start_idx = 0
        else:
            # 随机选择起始位置
            start_idx = random.randint(0, total_frames - self.segment_length)
        
        # 截取指定长度的片段
        features = features[:, :, start_idx:start_idx + self.segment_length]
        targets = targets[:, :, start_idx:start_idx + self.segment_length]
        
        # 转换为torch张量
        features = torch.from_numpy(features).float()
        targets = torch.from_numpy(targets).float()
        
        # 应用变换（如果有）
        if self.transform:
            features = self.transform(features)
            targets = self.transform(targets)
        
        return features, targets

def get_preprocessed_dataloader(data_dir, batch_size=16, segment_length=None, num_workers=4):
    """
    获取预处理数据的数据加载器
    
    参数:
        data_dir (str): 预处理后的数据目录路径
        batch_size (int): 批次大小
        segment_length (int, optional): 模型输入的时间步长度，如果为None则使用config中的值
        num_workers (int): 数据加载的工作线程数
    
    返回:
        train_loader, val_loader, test_loader
    """
    train_dataset = PreprocessedAECDataset(data_dir, split='train', segment_length=segment_length)
    val_dataset = PreprocessedAECDataset(data_dir, split='val', segment_length=segment_length)
    test_dataset = PreprocessedAECDataset(data_dir, split='test', segment_length=segment_length)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader 