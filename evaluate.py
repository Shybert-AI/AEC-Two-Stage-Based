import os
import numpy as np
import torch
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd
import random
import scipy.signal as signal
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import matplotlib
from TDC_wRLS import w_RLS_all
from models.unet import AECUNet,AECLSTMNet
from config import config

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'

# 性能评估指标
def calculate_metrics(clean_signal, noisy_signal, processed_signal, fs):
    """
    计算各种音频质量指标
    
    参数:
        clean_signal: 干净的参考信号
        noisy_signal: 带噪声的信号
        processed_signal: 处理后的信号
        fs: 采样率
    
    返回:
        包含各种指标的字典
    """
    # 确保信号长度相同
    min_length = min(len(clean_signal), len(noisy_signal), len(processed_signal))
    clean_signal = clean_signal[:min_length]
    noisy_signal = noisy_signal[:min_length]
    processed_signal = processed_signal[:min_length]
    
    # 计算信噪比 (SNR)
    def calculate_snr(clean, noisy):
        noise = clean - noisy
        signal_power = np.sum(clean ** 2)
        noise_power = np.sum(noise ** 2)
        if noise_power == 0:
            return float('inf')
        return 10 * np.log10(signal_power / noise_power)
    
    # 计算回声返回损失增强 (ERLE)
    def calculate_erle(mic, processed):
        mic_power = np.mean(mic ** 2)
        error_power = np.mean(processed ** 2)
        if error_power == 0:
            return float('inf')
        return 10 * np.log10(mic_power / error_power)
    
    # 计算感知评估语音质量 (PESQ)
    try:
        from pesq import pesq
        pesq_noisy = pesq(fs, clean_signal, noisy_signal, 'nb')
        pesq_processed = pesq(fs, clean_signal, processed_signal, 'nb')
    except:
        # 如果没有安装pesq或出错，使用模拟值
        pesq_noisy = -1
        pesq_processed = -1
        print("警告: 无法计算PESQ指标, 请安装pesq包")
    
    # 计算短时傅里叶变换的频谱距离
    def stft_distance(clean, processed):
        clean_stft = librosa.stft(clean, n_fft=config['fft_size'], 
                                 hop_length=config['hop_size'], 
                                 win_length=config['win_length'])
        proc_stft = librosa.stft(processed, n_fft=config['fft_size'], 
                               hop_length=config['hop_size'], 
                               win_length=config['win_length'])
        
        clean_mag = np.abs(clean_stft)
        proc_mag = np.abs(proc_stft)
        
        # 计算频谱距离
        return np.mean(np.abs(clean_mag - proc_mag))
    
    # 计算MSE
    mse_noisy = mean_squared_error(clean_signal, noisy_signal)
    mse_processed = mean_squared_error(clean_signal, processed_signal)
    
    # 计算相关系数
    corr_noisy = np.corrcoef(clean_signal, noisy_signal)[0, 1]
    corr_processed = np.corrcoef(clean_signal, processed_signal)[0, 1]
    
    # 收集所有指标
    metrics = {
        "SNR_noisy": calculate_snr(clean_signal, noisy_signal),
        "SNR_processed": calculate_snr(clean_signal, processed_signal),
        "ERLE": calculate_erle(noisy_signal, processed_signal),
        "PESQ_noisy": pesq_noisy,
        "PESQ_processed": pesq_processed,
        "MSE_noisy": mse_noisy,
        "MSE_processed": mse_processed,
        "Corr_noisy": corr_noisy,
        "Corr_processed": corr_processed,
        "STFT_distance_noisy": stft_distance(clean_signal, noisy_signal),
        "STFT_distance_processed": stft_distance(clean_signal, processed_signal)
    }
    
    return metrics

def evaluate_signals(file_indices, output_dir="evaluation"):
    """
    评估指定索引的信号集合
    
    参数:
        file_indices: 要评估的文件索引列表
        output_dir: 输出目录
    
    返回:
        包含性能指标的DataFrame
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化存储结果的DataFrame
    results = []
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model_type = "unet"
    if model_type == 'unet':
        # 加载模型
        model = AECUNet().to(device)
        checkpoint_path = os.path.join("checkpoints", "unet_best.pth")
    else:
        model = AECLSTMNet().to(device)
        checkpoint_path = os.path.join("checkpoints", "lstm_best.pth")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载检查点: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    model.eval()
    # 遍历所有文件
    for idx in tqdm(file_indices, desc="处理文件"):

        # 构建文件路径
        farend_path = os.path.join("data", "processed", "farend", f"{idx}.wav")
        mic_path = os.path.join("data", "processed", "mic", f"{idx}.wav")
        error_path = os.path.join("data", "processed", "error", f"{idx}.wav")
        
        # 检查文件是否存在
        if not all(os.path.exists(path) for path in [farend_path, mic_path, error_path]):
            continue
        
        # 加载音频
        farend, sr = librosa.load(farend_path, sr=config['sample_rate'])
        mic, _ = librosa.load(mic_path, sr=config['sample_rate'])
        error, _ = librosa.load(error_path, sr=config['sample_rate'])
        
        # 确保长度一致
        min_length = min(len(farend), len(mic), len(error))
        farend = farend[:min_length]
        mic = mic[:min_length]
        error = error[:min_length]
        
        # 第一步：计算线性滤波的性能指标
        metrics_linear = calculate_metrics(farend, mic, error, config['sample_rate'])
        
        # 第二步：使用神经网络处理
        # 进行STFT变换
        mic_stft = librosa.stft(mic, n_fft=config['fft_size'], 
                               hop_length=config['hop_size'], 
                               win_length=config['win_length'])
        farend_stft = librosa.stft(farend, n_fft=config['fft_size'], 
                                  hop_length=config['hop_size'], 
                                  win_length=config['win_length'])
        error_stft = librosa.stft(error, n_fft=config['fft_size'], 
                                 hop_length=config['hop_size'], 
                                 win_length=config['win_length'])
        
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
        with torch.no_grad():
            outputs = model(features)
        
        # 将输出转换回numpy
        outputs = outputs.squeeze(0).cpu().numpy()
        
        # 将实部和虚部转换为复数
        output_complex = outputs[0] + 1j * outputs[1]
        
        # 反向STFT
        neural_output = librosa.istft(output_complex, hop_length=config['hop_size'], 
                                     win_length=config['win_length'])
        
        # 确保神经网络输出与其他信号长度一致
        if len(neural_output) > min_length:
            neural_output = neural_output[:min_length]
        elif len(neural_output) < min_length:
            # 填充0使其长度一致
            padding = np.zeros(min_length - len(neural_output))
            neural_output = np.concatenate([neural_output, padding])
        
        # 计算神经网络处理后的性能指标
        metrics_neural = calculate_metrics(farend, mic, neural_output, config['sample_rate'])
        
        # 合并结果并添加到DataFrame
        result = {
            "file_id": idx,
            "linear_SNR": metrics_linear["SNR_processed"],
            "neural_SNR": metrics_neural["SNR_processed"],
            "linear_ERLE": metrics_linear["ERLE"],
            "neural_ERLE": metrics_neural["ERLE"],
            "linear_PESQ": metrics_linear["PESQ_processed"],
            "neural_PESQ": metrics_neural["PESQ_processed"],
            "linear_MSE": metrics_linear["MSE_processed"],
            "neural_MSE": metrics_neural["MSE_processed"],
            "linear_Corr": metrics_linear["Corr_processed"],
            "neural_Corr": metrics_neural["Corr_processed"],
            "linear_STFT_distance": metrics_linear["STFT_distance_processed"],
            "neural_STFT_distance": metrics_neural["STFT_distance_processed"]
        }
        
        results.append(result)
        
        # 保存部分样本进行可视化
        if len(results) <= 5:  # 只保存前5个样本
            # 保存处理后的音频
            sf.write(os.path.join(output_dir, f"{idx}_neural.wav"), neural_output, config['sample_rate'])
            
            # 绘制波形图
            plt.figure(figsize=(15, 10))
            
            t = np.arange(min_length) / config['sample_rate']
            
            # 麦克风信号
            plt.subplot(4, 1, 1)
            plt.plot(t, mic)
            plt.title('麦克风信号')
            plt.xlabel('时间 (秒)')
            plt.ylabel('幅度')
            
            # 线性滤波后的误差信号
            plt.subplot(4, 1, 2)
            plt.plot(t, error)
            plt.title('线性滤波后的误差信号')
            plt.xlabel('时间 (秒)')
            plt.ylabel('幅度')
            
            # 神经网络输出
            plt.subplot(4, 1, 3)
            plt.plot(t, neural_output)
            plt.title('神经网络输出')
            plt.xlabel('时间 (秒)')
            plt.ylabel('幅度')
            
            # 远端信号
            plt.subplot(4, 1, 4)
            plt.plot(t, farend)
            plt.title('远端信号')
            plt.xlabel('时间 (秒)')
            plt.ylabel('幅度')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{idx}_waveforms.png"))
            plt.close()
    
    # 将结果转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存结果为CSV
    results_df.to_csv(os.path.join(output_dir, "evaluation_results.csv"), index=False)
    
    return results_df

def visualize_results(results_df, output_dir="evaluation"):
    """
    可视化评估结果
    
    参数:
        results_df: 包含性能指标的DataFrame
        output_dir: 输出目录
    """
    # 计算平均指标
    avg_metrics = {
        "SNR": {
            "线性滤波": results_df["linear_SNR"].mean(),
            "神经网络": results_df["neural_SNR"].mean()
        },
        "ERLE": {
            "线性滤波": results_df["linear_ERLE"].mean(),
            "神经网络": results_df["neural_ERLE"].mean()
        },
        "PESQ": {
            "线性滤波": results_df["linear_PESQ"].mean(),
            "神经网络": results_df["neural_PESQ"].mean()
        },
        "MSE": {
            "线性滤波": results_df["linear_MSE"].mean(),
            "神经网络": results_df["neural_MSE"].mean()
        },
        "相关系数": {
            "线性滤波": results_df["linear_Corr"].mean(),
            "神经网络": results_df["neural_Corr"].mean()
        },
        "STFT距离": {
            "线性滤波": results_df["linear_STFT_distance"].mean(),
            "神经网络": results_df["neural_STFT_distance"].mean()
        }
    }
    
    # 创建条形图比较指标
    plt.figure(figsize=(15, 10))
    
    # 设置子图数量
    metrics_count = len(avg_metrics)
    cols = 2
    rows = (metrics_count + 1) // cols
    
    # 遍历所有指标，绘制条形图
    for i, (metric_name, values) in enumerate(avg_metrics.items(), 1):
        plt.subplot(rows, cols, i)
        
        methods = list(values.keys())
        metric_values = list(values.values())
        
        bars = plt.bar(methods, metric_values, color=['#3498db', '#e74c3c'])
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.,
                     height + 0.01 * max(metric_values),
                     f'{height:.4f}',
                     ha='center', va='bottom', rotation=0)
        
        plt.title(f'平均{metric_name}')
        plt.ylabel(metric_name)
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"))
    
    # 创建信号改进的箱线图
    plt.figure(figsize=(15, 8))
    
    # 计算每个文件的性能改进
    results_df['SNR_improvement'] = results_df['neural_SNR'] - results_df['linear_SNR']
    results_df['ERLE_improvement'] = results_df['neural_ERLE'] - results_df['linear_ERLE']
    results_df['PESQ_improvement'] = results_df['neural_PESQ'] - results_df['linear_PESQ']
    
    plt.subplot(1, 3, 1)
    plt.boxplot(results_df['SNR_improvement'])
    plt.title('SNR改进')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 3, 2)
    plt.boxplot(results_df['ERLE_improvement'])
    plt.title('ERLE改进')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 3, 3)
    plt.boxplot(results_df['PESQ_improvement'])
    plt.title('PESQ改进')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle('神经网络相对于线性滤波的性能改进')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "improvements_boxplot.png"))
    
    # 创建散点图比较线性滤波和神经网络结果
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.scatter(results_df['linear_SNR'], results_df['neural_SNR'], alpha=0.7)
    plt.plot([min(results_df['linear_SNR']), max(results_df['linear_SNR'])], 
             [min(results_df['linear_SNR']), max(results_df['linear_SNR'])], 'r--')
    plt.xlabel('线性滤波 SNR')
    plt.ylabel('神经网络 SNR')
    plt.title('SNR比较')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.scatter(results_df['linear_ERLE'], results_df['neural_ERLE'], alpha=0.7)
    plt.plot([min(results_df['linear_ERLE']), max(results_df['linear_ERLE'])], 
             [min(results_df['linear_ERLE']), max(results_df['linear_ERLE'])], 'r--')
    plt.xlabel('线性滤波 ERLE')
    plt.ylabel('神经网络 ERLE')
    plt.title('ERLE比较')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.scatter(results_df['linear_PESQ'], results_df['neural_PESQ'], alpha=0.7)
    plt.plot([min(results_df['linear_PESQ']), max(results_df['linear_PESQ'])], 
             [min(results_df['linear_PESQ']), max(results_df['linear_PESQ'])], 'r--')
    plt.xlabel('线性滤波 PESQ')
    plt.ylabel('神经网络 PESQ')
    plt.title('PESQ比较')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.scatter(results_df['linear_Corr'], results_df['neural_Corr'], alpha=0.7)
    plt.plot([min(results_df['linear_Corr']), max(results_df['linear_Corr'])], 
             [min(results_df['linear_Corr']), max(results_df['linear_Corr'])], 'r--')
    plt.xlabel('线性滤波 相关系数')
    plt.ylabel('神经网络 相关系数')
    plt.title('相关系数比较')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_scatter.png"))

def main():
    # 创建输出目录
    output_dir = "evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    # 在9000-9999范围内随机选择100个文件索引
    all_indices = list(range(9000, 10000))
    selected_indices = random.sample(all_indices, 100)
    
    print(f"随机选择了{len(selected_indices)}个文件进行评估:")
    print(selected_indices[:5], "...")
    
    # 评估所选文件
    results_df = evaluate_signals(selected_indices, output_dir)
    print(122,results_df)
    
    # 可视化结果
    visualize_results(results_df, output_dir)
    
    print(f"评估完成！结果保存在 {output_dir} 目录中")
    
    # 输出平均指标
    print("\n平均性能指标:")
    print(f"线性滤波 SNR: {results_df['linear_SNR'].mean():.4f} dB")
    print(f"神经网络 SNR: {results_df['neural_SNR'].mean():.4f} dB")
    print(f"SNR 改进: {(results_df['neural_SNR'] - results_df['linear_SNR']).mean():.4f} dB")
    print()
    print(f"线性滤波 ERLE: {results_df['linear_ERLE'].mean():.4f} dB")
    print(f"神经网络 ERLE: {results_df['neural_ERLE'].mean():.4f} dB")
    print(f"ERLE 改进: {(results_df['neural_ERLE'] - results_df['linear_ERLE']).mean():.4f} dB")
    print()
    print(f"线性滤波 PESQ: {results_df['linear_PESQ'].mean():.4f}")
    print(f"神经网络 PESQ: {results_df['neural_PESQ'].mean():.4f}")
    print(f"PESQ 改进: {(results_df['neural_PESQ'] - results_df['linear_PESQ']).mean():.4f}")

if __name__ == "__main__":
    main()

    """
    config = {
            'TDE_win_len':0.5,
            'TDE_win_inc':0.25,
            'WRLS_win_len':0.02,
            'WRLS_win_inc':0.01,
            'L':5,
            'B':0.2,
            'eps':0.001,
            # STFT参数
            'fft_size': 320,
            'hop_size': 160,
            'win_length': 320,
            'freq_bins': 161,  # fft_size//2 + 1
            # 数据集参数
            'segment_length': 160,
            'sample_rate': 16000,
            # 模型参数
            'input_channels': 6,
            'output_channels': 2,
            'hidden_size': 256
    } 
    
    unet 
    平均性能指标:
    线性滤波 SNR: -3.2988 dB
    神经网络 SNR: -2.5413 dB
    SNR 改进: 0.7574 dB
    
    线性滤波 ERLE: 0.9472 dB
    神经网络 ERLE: 2.2396 dB
    ERLE 改进: 1.2924 dB
    
    线性滤波 PESQ: 1.2685
    神经网络 PESQ: 1.3219
    PESQ 改进: 0.0533
    
    lstm 
    平均性能指标:
    线性滤波 SNR: -3.4216 dB
    神经网络 SNR: -2.4584 dB
    SNR 改进: 0.9632 dB
    
    线性滤波 ERLE: 0.5371 dB
    神经网络 ERLE: 2.2811 dB
    ERLE 改进: 1.7440 dB
    
    线性滤波 PESQ: 1.2705
    神经网络 PESQ: 1.2912
    PESQ 改进: 0.0207
    
    
    config = {
            'TDE_win_len':0.5,
            'TDE_win_inc':0.25,
            'WRLS_win_len':0.02,
            'WRLS_win_inc':0.01,
            'L':5,
            'B':0.2,
            'eps':0.001,
            # STFT参数
            'fft_size': 512,
            'hop_size': 160,
            'win_length': 256,
            'freq_bins': 257,  # fft_size//2 + 1
            # 数据集参数
            'segment_length': 160,
            'sample_rate': 16000,
            # 模型参数
            'input_channels': 6,
            'output_channels': 2,
            'hidden_size': 256
    } 
    lstm
    平均性能指标:
    线性滤波 SNR: -3.3360 dB
    神经网络 SNR: -2.2915 dB
    SNR 改进: 1.0445 dB
    
    线性滤波 ERLE: 0.7992 dB
    神经网络 ERLE: 2.8647 dB
    ERLE 改进: 2.0654 dB
    
    线性滤波 PESQ: 1.2675
    神经网络 PESQ: 1.2913
    PESQ 改进: 0.0238
        
    """