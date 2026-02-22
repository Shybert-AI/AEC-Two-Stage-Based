import os
import librosa
import soundfile as sf
from tqdm import tqdm
import numpy as np
import concurrent.futures

# 定义输入和输出目录
input_dir = 'synthetic'
output_base_dir = 'data/resampled'

# 定义不同类型的音频
audio_types = ['echo', 'farend', 'mic', 'nearend', 'target']

# 创建输出目录
for audio_type in audio_types:
    os.makedirs(os.path.join(output_base_dir, audio_type), exist_ok=True)

# 获取所有音频文件
audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
# 按组ID对文件进行分组
file_groups = {}
for file in audio_files:
    group_id = file.split('_')[0]  # 提取组ID (例如 f00000)
    if group_id not in file_groups:
        file_groups[group_id] = []
    file_groups[group_id].append(file)

# 处理单个音频文件的函数
def process_audio(file):
    input_path = os.path.join(input_dir, file)
    
    # 确定音频类型
    audio_type = file.split('_')[1].split('.')[0]  # 提取类型 (echo, farend, mic, nearend, target)
    
    # 设置输出路径
    output_path = os.path.join(output_base_dir, audio_type, file)
    
    # 加载音频并重采样到16kHz
    try:
        audio, sr = librosa.load(input_path, sr=None)  # 加载原始采样率
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # 保存重采样后的音频
        sf.write(output_path, audio, 16000)
        return True
    except Exception as e:
        print(f"处理文件 {file} 时出错: {e}")
        return False

# 使用多线程处理所有文件
def process_all_files():
    total_files = len(audio_files)
    print(f"开始处理 {total_files} 个音频文件...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_audio, audio_files), total=total_files))
    
    success_count = sum(results)
    print(f"处理完成! 成功处理 {success_count}/{total_files} 个文件")

if __name__ == "__main__":
    process_all_files() 