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