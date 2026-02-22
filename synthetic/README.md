### Overview
The synthetic dataset provides 10,000 examples representing single talk, double talk, near end noise, far end noise, and various nonlinear distortion situations. 
Each example includes a far end speech, echo signal, near end speech and near end microphone signal clip. 
The clips can be linked by the `fileid_{int}` attribute.
The meta.csv file includes information about the synthesis parameters such as source clip ids, speaker ids and signal-to-echo ratios.

For a more detailed description of the generation process, please refer to the [paper](https://arxiv.org/abs/2009.04972).

### Directory contents
`farend_speech` - far end signals, some of these include background noise (indicated by `is_farend_noisy=1` in the meta.csv file).

`echo_signal` - transformed version of far end speech, used as echo signals.

`nearend_speech` - clean near end signals that can be used as targets. Note that these signals are clean and do not include near end noise. The use of near end noise is indicated by `is_nearend_noisy=1` in the meta.csv file. The meta.csv file includes a `nearend_scale` column - if you multiply the scale factor with the near end signal, you get the signal in the same scale as in the `nearend_mic_signal` clip.

`nearend_mic_signal` - near end microphone signals - mixtures of nearend speech and echo signals. The signal-to-echo ratio is indicated in the `ser` column in the meta.csv file. The clips might also include near end noise (indicated by `is_nearend_noisy=1`).



### 概述
该合成数据集提供了10,000个样本，分别代表单端通话、双端通话、近端噪声、远端噪声以及各种非线性失真情况。
每个示例都包含远端语音、回声信号、近端语音和近端麦克风信号片段。
这些剪辑可以通过`fileid_{int}`属性进行链接。
meta.csv文件包含有关合成参数的信息，如源剪辑ID、说话者ID和信号回声比。
如需了解生成过程的更详细描述，请参阅[论文](https://arxiv.org/abs/2009.04972)。
### 目录内容
`farend_speech` - 远端信号，其中一些包括背景噪声（在meta.csv文件中用`is_farend_noisy=1`表示）。
`echo_signal` - 远端语音的转换版本，用作回声信号。
`nearend_speech` - 干净的近端信号，可用作目标信号。请注意，这些信号是干净的，不包含近端噪声。近端噪声的使用情况由meta.csv文件中的`is_nearend_noisy=1`表示。meta.csv文件包含一个`nearend_scale`列 - 如果将比例因子与近端信号相乘，则得到的信号与`nearend_mic_signal`片段中的信号比例相同。
`nearend_mic_signal` - 近端麦克风信号 - 近端语音和回声信号的混合。信号与回声比在meta.csv文件的`ser`列中指示。片段中还可能包含近端噪声（由`is_nearend_noisy=1`指示）。