import os
import argparse
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import sys
sys.path.insert(0,".")
print(sys.path)
from TDC_wRLS import w_RLS_all
from config import config
import multiprocessing
from functools import partial
import torch
import torchaudio
import time

def process_file_group(group_id, input_dir, output_dir, fs, use_gpu=False, gpu_id=0):
    """
    处理单个文件组，应用TDC和线性滤波
    
    参数:
        group_id (str): 文件组ID
        input_dir (str): 输入数据目录
        output_dir (str): 输出数据目录
        fs (int): 采样率
        use_gpu (bool): 是否使用GPU加速
        gpu_id (int): 使用的GPU ID
    """
    try:
        # 设置设备
        if use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cpu')
            
        # 加载原始信号
        farend_path = os.path.join(input_dir, 'farend', f'{group_id}')
        mic_path = os.path.join(input_dir, 'mic', f'{group_id}')
        nearend_path = os.path.join(input_dir, 'nearend', f'{group_id}')
        
        farend, _ = librosa.load(farend_path, sr=fs)
        mic, _ = librosa.load(mic_path, sr=fs)
        nearend, _ = librosa.load(nearend_path, sr=fs)
        
        # 裁剪长度到信号的最小值
        min_length = min(len(farend), len(mic), len(nearend))
        farend = farend[:min_length]
        mic = mic[:min_length]
        nearend = nearend[:min_length]
        
        # 应用线性滤波 (TDC和WRLS)
        # w_RLS_all返回: e (误差信号), y (线性滤波输出的回声信号), mic (麦克风信号), taus, inc_TDE
        error, echo_est, _, _, _ = w_RLS_all(mic, farend, config, fs=fs)
        
        # 确保长度一致
        min_length = min(len(error), len(farend), len(mic), len(nearend))
        error = error[:min_length]
        farend = farend[:min_length]
        mic = mic[:min_length]
        nearend = nearend[:min_length]
        
        # 保存处理后的信号
        os.makedirs(os.path.join(output_dir, 'farend'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'mic'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'nearend'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'error'), exist_ok=True)
        
        sf.write(os.path.join(output_dir, 'farend', f'{group_id}'), farend, fs)
        sf.write(os.path.join(output_dir, 'mic', f'{group_id}'), mic, fs)
        sf.write(os.path.join(output_dir, 'nearend', f'{group_id}'), nearend, fs)
        sf.write(os.path.join(output_dir, 'error', f'{group_id}'), error, fs)
        
        return True
    except Exception as e:
        print(f"处理文件组 {group_id} 时出错: {str(e)}")
        return False

def process_file_group_gpu(group_id, input_dir, output_dir, fs, device):
    """
    使用GPU处理单个文件组，应用TDC和线性滤波
    
    参数:
        group_id (str): 文件组ID
        input_dir (str): 输入数据目录
        output_dir (str): 输出数据目录
        fs (int): 采样率
        device (torch.device): 计算设备
    """
    try:
        # 加载原始信号
        farend_path = os.path.join(input_dir, 'farend', f'{group_id}')
        mic_path = os.path.join(input_dir, 'mic', f'{group_id}')
        nearend_path = os.path.join(input_dir, 'nearend', f'{group_id}')
        
        # 使用torchaudio加载音频
        farend_tensor, sr = torchaudio.load(farend_path)
        mic_tensor, _ = torchaudio.load(mic_path)
        nearend_tensor, _ = torchaudio.load(nearend_path)
        
        # 重采样到目标采样率（如果需要）
        if sr != fs:
            resampler = torchaudio.transforms.Resample(sr, fs).to(device)
            farend_tensor = resampler(farend_tensor)
            mic_tensor = resampler(mic_tensor)
            nearend_tensor = resampler(nearend_tensor)
        
        # 转换为单声道（如果需要）
        if farend_tensor.shape[0] > 1:
            farend_tensor = torch.mean(farend_tensor, dim=0, keepdim=True)
        if mic_tensor.shape[0] > 1:
            mic_tensor = torch.mean(mic_tensor, dim=0, keepdim=True)
        if nearend_tensor.shape[0] > 1:
            nearend_tensor = torch.mean(nearend_tensor, dim=0, keepdim=True)
        
        # 移动到GPU
        farend_tensor = farend_tensor.to(device)
        mic_tensor = mic_tensor.to(device)
        nearend_tensor = nearend_tensor.to(device)
        
        # 裁剪长度到信号的最小值
        min_length = min(farend_tensor.shape[1], mic_tensor.shape[1], nearend_tensor.shape[1])
        farend_tensor = farend_tensor[:, :min_length]
        mic_tensor = mic_tensor[:, :min_length]
        nearend_tensor = nearend_tensor[:, :min_length]
        
        # 转换回CPU进行线性滤波（因为w_RLS_all不支持GPU）
        farend = farend_tensor.cpu().numpy().squeeze()
        mic = mic_tensor.cpu().numpy().squeeze()
        nearend = nearend_tensor.cpu().numpy().squeeze()
        
        # 应用线性滤波 (TDC和WRLS)
        error, echo_est, _, _, _ = w_RLS_all(mic, farend, config, fs=fs)
        
        # 确保长度一致
        min_length = min(len(error), len(farend), len(mic), len(nearend))
        error = error[:min_length]
        farend = farend[:min_length]
        mic = mic[:min_length]
        nearend = nearend[:min_length]
        
        # 保存处理后的信号
        os.makedirs(os.path.join(output_dir, 'farend'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'mic'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'nearend'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'error'), exist_ok=True)
        
        sf.write(os.path.join(output_dir, 'farend', f'{group_id}'), farend, fs)
        sf.write(os.path.join(output_dir, 'mic', f'{group_id}'), mic, fs)
        sf.write(os.path.join(output_dir, 'nearend', f'{group_id}'), nearend, fs)
        sf.write(os.path.join(output_dir, 'error', f'{group_id}'), error, fs)
        
        return True
    except Exception as e:
        print(f"处理文件组 {group_id} 时出错: {str(e)}")
        return False

def preprocess_dataset(args):
    """
    预处理整个数据集，应用TDC和线性滤波
    
    参数:
        args: 命令行参数
    """
    input_dir = args.input_dir
    output_dir = args.output_dir
    fs = config['sample_rate']
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有文件组ID
    file_groups = []
    farend_dir = os.path.join(input_dir, 'farend')
    for file_name in os.listdir(farend_dir):
        if file_name.endswith('.wav'):
            group_id = file_name.split('_')[0]  # 例如从f00001_farend.wav提取f00001
            file_groups.append(group_id)
    
    print(f"找到 {len(file_groups)} 个文件组")
    
    # 检查是否使用GPU
    if args.use_gpu:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{args.gpu_id}')
            print(f"使用GPU: {torch.cuda.get_device_name(device)}")
        else:
            device = torch.device('cpu')
            print("GPU不可用，使用CPU")
            args.use_gpu = False
    else:
        device = torch.device('cpu')
        print("使用CPU")
    
    start_time = time.time()
    
    # 使用多进程处理文件
    if args.num_workers > 1 and not args.use_gpu:
        print(f"使用 {args.num_workers} 个CPU进程进行处理")
        process_func = partial(process_file_group, input_dir=input_dir, output_dir=output_dir, fs=fs)
        
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            results = list(tqdm(pool.imap(process_func, file_groups), total=len(file_groups), desc="预处理数据"))
        
        success_count = sum(results)
    else:
        # 单进程处理（CPU或GPU）
        success_count = 0
        if args.use_gpu:
            print(f"使用GPU进行处理")
            for group_id in tqdm(file_groups, desc="预处理数据"):
                if process_file_group_gpu(group_id, input_dir, output_dir, fs, device):
                    success_count += 1
        else:
            print(f"使用单CPU进程进行处理")
            for group_id in tqdm(file_groups, desc="预处理数据"):
                if process_file_group(group_id, input_dir, output_dir, fs):
                    success_count += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"预处理完成! 成功处理 {success_count}/{len(file_groups)} 个文件组")
    print(f"总处理时间: {total_time:.2f}秒, 平均每个文件组: {total_time/len(file_groups):.2f}秒")
    print(f"处理后的数据保存在 {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据预处理脚本 - 应用TDC和线性滤波")
    parser.add_argument('--input_dir', type=str, default="data/resampled", help='原始数据目录路径')
    parser.add_argument('--output_dir', type=str, default="data/processed", help='处理后数据保存目录路径')
    parser.add_argument('--num_workers', type=int, default=16, help='处理的CPU工作进程数')
    parser.add_argument('--use_gpu', default=False, help='是否使用GPU加速')
    parser.add_argument('--gpu_id', type=int, default=0, help='使用的GPU ID')
    
    args = parser.parse_args()
    preprocess_dataset(args)