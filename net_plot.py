from models.unet import AECUNet,AECLSTMNet
# 模型可视化工具：NETRON
import netron
import torch
import onnx
from onnxsim import simplify
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
model_type = "lstm"
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

#batch_size, channels, freq_bins, time_frames = x.shape
# 按照输入格式，设计随机输入
dummy_input =torch.randn(1, 6, 257, 160).cuda()
# 导出模型
model_path = f'{model_type}_best.onnx'
model_path_sample = f'{model_type}_best_sample.onnx'
torch.onnx.export(model, dummy_input, model_path, verbose=False)

# 加载模型

model = onnx.load(model_path)

# 简化模型
simplify(model)
onnx.save(model, model_path_sample)
print("模型简化完成，输出路径为:", model_path_sample)

#打开服务
#netron.start(model_path_sample)