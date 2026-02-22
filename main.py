import os
import argparse
import numpy as np
import torch
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from TDC_wRLS import w_RLS_all
from models.unet import AECUNet, AECLSTMNet
from config import config

def process_audio(args):
    # 加载音频文件
    farend, sr = librosa.load(args.farend_path, sr=config['sample_rate'])
    mic, sr = librosa.load(args.mic_path, sr=config['sample_rate'])
    
    # 确保长度一致
    min_length = min(len(farend), len(mic))
    farend = farend[:min_length]
    mic = mic[:min_length]
    
    # 第一步：使用线性滤波（TDC和WRLS）
    print("第一步：应用线性滤波...")
    error, echo_est, _, taus, _ = w_RLS_all(mic, farend, config, fs=config['sample_rate'])
    
    # 确保长度一致
    min_length = min(len(error), len(farend), len(mic))
    error = error[:min_length]
    farend = farend[:min_length]
    mic = mic[:min_length]
    
    # 保存线性滤波结果
    sf.write(os.path.join(args.output_dir, 'linear_output.wav'), error, config['sample_rate'])
    sf.write(os.path.join(args.output_dir, 'estimated_echo.wav'), echo_est, config['sample_rate'])
    
    # 第二步：使用神经网络消除非线性回声
    print("第二步：应用神经网络消除非线性回声...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    if args.model_type == 'unet':
        model = AECUNet()
    elif args.model_type == 'lstm':
        model = AECLSTMNet()
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    
    model = model.to(device)
    
    # 加载检查点
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.model_type}_best.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载检查点: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    # 进行短时傅里叶变换(STFT)
    mic_stft = librosa.stft(mic, n_fft=config['fft_size'], hop_length=config['hop_size'], win_length=config['win_length'])
    farend_stft = librosa.stft(farend, n_fft=config['fft_size'], hop_length=config['hop_size'], win_length=config['win_length'])
    error_stft = librosa.stft(error, n_fft=config['fft_size'], hop_length=config['hop_size'], win_length=config['win_length'])
    
    # 获取实部和虚部
    mic_real = np.real(mic_stft)
    mic_imag = np.imag(mic_stft)
    farend_real = np.real(farend_stft)
    farend_imag = np.imag(farend_stft)
    error_real = np.real(error_stft)
    error_imag = np.imag(error_stft)
    
    # 将特征按通道拼接，形状为[通道数，特征数，帧数]
    features = np.stack([
        mic_real, mic_imag, 
        farend_real, farend_imag, 
        error_real, error_imag
    ], axis=0)  # shape: [6, freq_bins, time_frames]
    
    # 转换为torch张量
    features = torch.from_numpy(features).float().unsqueeze(0).to(device)  # 添加批次维度
    
    # 模型推理
    model.eval()
    with torch.no_grad():
        outputs = model(features)
    
    # 将输出转换回numpy
    outputs = outputs.squeeze(0).cpu().numpy()
    
    # 将实部和虚部转换为复数
    output_complex = outputs[0] + 1j * outputs[1]
    
    # 反向STFT
    neural_output = librosa.istft(output_complex, hop_length=config['hop_size'], win_length=config['win_length'])
    
    # 保存神经网络输出
    sf.write(os.path.join(args.output_dir, 'neural_output.wav'), neural_output, config['sample_rate'])
    
    # 绘制波形图
    plt.figure(figsize=(15, 10))
    
    # 麦克风信号
    plt.subplot(5, 1, 1)
    plt.plot(mic)
    plt.title('麦克风信号')
    plt.xlabel('样本')
    plt.ylabel('幅度')
    
    # 线性滤波估计的回声信号
    plt.subplot(5, 1, 2)
    plt.plot(echo_est)
    plt.title('线性滤波估计的回声信号')
    plt.xlabel('样本')
    plt.ylabel('幅度')
    
    # 线性滤波后的误差信号
    plt.subplot(5, 1, 3)
    plt.plot(error)
    plt.title('线性滤波后的误差信号')
    plt.xlabel('样本')
    plt.ylabel('幅度')
    
    # 神经网络输出
    plt.subplot(5, 1, 4)
    plt.plot(neural_output)
    plt.title('神经网络输出')
    plt.xlabel('样本')
    plt.ylabel('幅度')
    
    # 远端信号
    plt.subplot(5, 1, 5)
    plt.plot(farend)
    plt.title('远端信号')
    plt.xlabel('样本')
    plt.ylabel('幅度')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'waveforms.png'))
    plt.close()
    
    print(f"处理完成! 结果保存在 {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="回声消除系统")
    parser.add_argument('--farend_path', type=str, default="test/farend_speech.wav", help='远端信号路径')
    parser.add_argument('--mic_path', type=str, default="test/nearend_mic.wav", help='麦克风信号路径')
    parser.add_argument('--model_type', type=str, default='unet', choices=['unet', 'lstm'], help='模型类型')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='检查点目录')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    process_audio(args) 