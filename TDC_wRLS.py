import numpy as np
from numpy import matrix as mat
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib
from config import config

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 优先使用的中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
matplotlib.rcParams['font.family'] = 'sans-serif'  # 使用sans-serif字体

# 将认为有延时的信号放第一个参数
# 将认为没有延时的参考信号放第二个参数
def  Time_delay_Estimate(S_near,S_far,alpha=0.1):
    D,T = S_near.shape
    Phi = np.zeros((D))
   
    taus = []
    for i in range(T):
        Phi = alpha*Phi + (1-alpha)*(S_near[:,i]*np.conjugate(S_far[:,i]))
        tau = np.argmax(np.abs(np.fft.irfft(Phi/np.abs(Phi))))
        taus.append((tau))
    return taus

def TDE_X(X,taus,inc_TDE,inc_frame):

    D,T = X.shape

    N_spread = int(inc_TDE/inc_frame)
    
    taus = np.expand_dims(np.array(taus),-1)
    taus_frame = taus/inc_frame
    
    taus = np.tile(taus_frame,[1,N_spread]).reshape([-1])
    X_TDE = np.zeros_like(X,dtype=complex)

    for i in range(T):       
        # index = max(i-int(taus[i]),0)
        index = i-int(taus[i])
        if i-int(taus[i])<0:
            continue
        else:
            X_TDE[:,i]=X[:,index]

    return X_TDE

def w_RLS(S_near,S_far,N_fft,N_win,N_inc,L=5,B=0.2,eps=0.01):
    # 进行 w-RLS
    F,T = S_near.shape

    X = mat(S_far).T
    D = mat(S_near).T
   
    gamma = mat(np.zeros([T,F]))
    W = mat(np.zeros([L,F],dtype=complex))
    E = mat(np.zeros([T,F]),dtype=complex)
    Y = mat(np.zeros([T,F]),dtype=complex)
    R_LF = [mat(np.eye(L)*eps,dtype=complex) for i in range(F)]
    r_LF = [mat(np.zeros([L,1]),dtype=complex) for i in range(F)]
    
    for t in range(T):
        
        buff_index = [max(t-j,0) for j in range(L)]
        X_L = X[buff_index,:]
        
        for f in range(F):
            W_Lf = W[:,f]
            Y_tf = mat.conj(W_Lf).T*X_L[:,f]
            Y[t,f] = Y_tf[0,0]
            E[t,f] = D[t,f]-Y[t,f]

            gamma[t,f] = np.abs(E[t,f])**(2-B)
          
            R_new = gamma[t,f]*X_L[:,f]*(mat.conj(X_L[:,f]).T)
            R_LF[f] = R_LF[f]+ R_new
            r_new = gamma[t,f]*X_L[:,f]*mat.conj(mat(D[t,f]))
            r_LF[f] =r_LF[f]+ r_new
           
            
            W_Lf = R_LF[f].I*r_LF[f]
            W[:,f]= W_Lf
    
    E = np.array(E.T)
    e = librosa.istft(E,n_fft=N_fft,hop_length=N_inc,win_length=N_win)
   
    Y = np.array(Y.T)
    y = librosa.istft(Y,n_fft=N_fft,hop_length=N_inc,win_length=N_win)
    
    mic = librosa.istft(S_near,n_fft=N_fft,hop_length=N_inc,win_length=N_win)
    return e,y,mic

def w_RLS_all(s_near,s_far,config,fs =16000):
    win_TDE=int(config['TDE_win_len']*fs)
    inc_TDE = int(config['TDE_win_inc']*fs)
    win_frame=int(config['WRLS_win_len']*fs)
    inc_frame=int(config['WRLS_win_inc']*fs)
    L= config['L']
    B= config['B']
    eps=config['eps']
    # 长度规整
    min_L = min(s_near.shape[0],s_far.shape[0]) 
    s_near = s_near[:min_L]
    s_far = s_far[:min_L]

    # 进行延时估计
    S_near_TDE = librosa.stft(s_near,n_fft=win_TDE,hop_length=inc_TDE,win_length=win_TDE)
    S_far_TDE = librosa.stft(s_far,n_fft=win_TDE,hop_length=inc_TDE,win_length=win_TDE)
    taus = Time_delay_Estimate(S_near_TDE,S_far_TDE)

  
    S_near = librosa.stft(s_near,n_fft=win_frame,hop_length=inc_frame,win_length=win_frame)
    S_far = librosa.stft(s_far,n_fft=win_frame,hop_length=inc_frame,win_length=win_frame)

    # 利用延时信息对 S_far进行矫正
    S_far = TDE_X(S_far,taus,inc_TDE=inc_TDE,inc_frame=inc_frame)

    # 进行 w-RLS
    e,y,mic = w_RLS(S_near,S_far,win_frame,win_frame,inc_frame,L=L,B=B,eps=eps)
    return e,y,mic,taus,inc_TDE

if __name__ == "__main__":
    # 加载测试音频文件
    try:
        # 尝试加载音频文件
        s_near, fs = sf.read("test/nearend_mic.wav")
        s_far, _ = sf.read("test/farend_speech.wav")
        print(f"成功加载音频文件，采样率: {fs}Hz")
    except:
        # 如果没有音频文件，生成测试信号
        print("未找到音频文件，生成测试信号...")
        fs = 16000
        t = np.arange(0, 3, 1/fs)  # 3秒的信号
        # 生成原始信号
        original = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
        # 添加延时生成远场信号
        delay_samples = 800  # 50ms的延时
        s_far = np.zeros_like(original)
        s_far[delay_samples:] = original[:-delay_samples]
        # 添加噪声生成近场信号
        noise = np.random.normal(0, 0.1, len(original))
        s_near = original + noise
        print(f"生成测试信号，采样率: {fs}Hz，人为延时: {delay_samples/fs*1000:.2f}ms")
    
    # 运行算法
    e, y, mic, taus, inc_TDE = w_RLS_all(s_near, s_far, config, fs)
    
    # 计算延时的时间（毫秒）
    # 修正延时计算：tau是FFT结果的索引，需要根据FFT大小转换为实际时间
    # 原来的计算方法：delay_times = [tau * (inc_TDE/fs) * 1000 for tau in taus]
    delay_times = [tau / fs * 1000 for tau in taus]
    
    # 打印延时信息
    print("\n延时估计结果:")
    print(f"最小延时: {min(delay_times):.2f}ms")
    print(f"最大延时: {max(delay_times):.2f}ms")
    print(f"平均延时: {np.mean(delay_times):.2f}ms")
    
    # 可视化延时和信号
    plt.figure(figsize=(15, 12))
    
    # 绘制延时估计结果
    plt.subplot(5, 1, 1)
    plt.plot(delay_times)
    plt.title("延时估计结果")
    plt.xlabel("帧索引")
    plt.ylabel("延时 (ms)")
    plt.grid(True)
    
    # 绘制信号波形
    t = np.arange(len(s_near)) / fs
    
    # 绘制远场参考信号
    plt.subplot(5, 1, 2)
    plt.plot(t[:len(s_far)], s_far)
    plt.title("远场参考信号波形")
    plt.xlabel("时间 (s)")
    plt.ylabel("幅度")
    plt.grid(True)
    
    # 绘制线性滤波输出的回声信号
    plt.subplot(5, 1, 3)
    plt.plot(t[:len(y)], y)
    plt.title("线性滤波输出的回声信号")
    plt.xlabel("时间 (s)")
    plt.ylabel("幅度")
    plt.grid(True)
    
    # 绘制近端信号（带回声）
    plt.subplot(5, 1, 4)
    plt.plot(t, s_near)
    plt.title("近端信号波形（带回声）")
    plt.xlabel("时间 (s)")
    plt.ylabel("幅度")
    plt.grid(True)
    
    # 绘制处理后的近端信号
    plt.subplot(5, 1, 5)
    plt.plot(t[:len(e)], e)
    plt.title("处理后的近端信号波形（回声消除后）")
    plt.xlabel("时间 (s)")
    plt.ylabel("幅度")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("test/delay_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存处理后的音频
    sf.write("test/processed.wav", e, fs)
    
    # 保存线性滤波输出的回声信号
    sf.write("test/estimated_echo.wav", y, fs)
    
    print("\n分析完成！结果已保存：")
    print("1. 延时分析图：test/delay_analysis.png")
    print("2. 处理后的近端信号：test/processed.wav")
    print("3. 估计的回声信号：test/estimated_echo.wav")