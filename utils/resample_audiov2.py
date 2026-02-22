import os
import librosa
import soundfile as sf
from tqdm import tqdm
import numpy as np
import concurrent.futures
from glob import iglob

# - `farend/`: 远端参考信号，命名格式为`f{编号:05d}_farend.wav`
# - `mic/`: 麦克风录制的混合信号，命名格式为`f{编号:05d}_mic.wav`
# - `error/`: 线性滤波后的信号，命名格式为`f{编号:05d}_error.wav`
# - `nearend/`: 近端的信号，命名格式为`f{编号:05d}_nearend.wav`

# 定义输入和输出目录
input_dir = 'D:/mywork/pythonProject/android_project/AEC-UNET-main/synthetic'
output_base_dir = 'D:/mywork/pythonProject/android_project/AEC-UNET-main//data/resampled'

# 定义不同类型的音频
audio_types = ['farend', 'mic', 'nearend']

# 创建输出目录
for audio_type in audio_types:
    os.makedirs(os.path.join(output_base_dir, audio_type), exist_ok=True)

# 获取所有音频文件
audio_files = list(iglob(os.path.join(input_dir,"**/*.wav"),recursive=True))

"""
`farend_speech` - 远端信号，其中一些包括背景噪声（在meta.csv文件中用`is_farend_noisy=1`表示）。
`echo_signal` - 远端语音的转换版本，用作回声信号。
`nearend_speech` - 干净的近端信号，可用作目标信号
"""
# 处理单个音频文件的函数
def process_audio(file):
    input_path = os.path.join(input_dir, file)
    # 确定音频类型
    audio_type = None
    if "farend_speech" in input_path:
        audio_type = "farend"
    elif "nearend_mic_signal" in input_path:
        audio_type = "mic"
    elif "nearend_speech" in input_path:
        audio_type = "nearend"

    if audio_type:
        # 设置输出路径
        output_path = os.path.join(output_base_dir, audio_type, os.path.basename(file).split("_")[-1])
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
    # for line in audio_files:
    #     process_audio(line)

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_audio, audio_files), total=total_files))

    # 删除3个文件中没有共同数据的文件
    mic = list(iglob(os.path.join(output_base_dir,"farend/*.wav"),recursive=True))
    for line in mic:
        if os.path.isfile(line) and  os.path.isfile(line.replace("farend","mic")) and os.path.isfile(line.replace("farend","nearend")):
            continue
        else:
            if os.path.isfile(line):
                os.remove(line)
            if os.path.isfile(line.replace("farend","mic")):
                os.remove(line.replace("farend","mic"))
            if os.path.isfile(line.replace("farend","nearend")):
                os.remove(line.replace("farend","nearend"))

if __name__ == "__main__":
    process_all_files()