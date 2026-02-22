import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.preprocessed_dataset import get_preprocessed_dataloader
from models.unet import AECUNet, AECLSTMNet
from config import config
import time


def train(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器 (使用预处理后的数据)
    train_loader, val_loader, _ = get_preprocessed_dataloader(
        args.data_dir, 
        batch_size=args.batch_size,
        segment_length=args.segment_length,
        num_workers=args.num_workers
    )
    
    # 创建模型
    if args.model_type == 'unet':
        model = AECUNet()
    elif args.model_type == 'lstm':
        model = AECLSTMNet()
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 初始化训练参数
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # 如果指定了恢复训练的检查点，则加载模型状态
    if args.resume and args.resume_checkpoint:
        if os.path.exists(args.resume_checkpoint):
            print(f"从检查点恢复训练: {args.resume_checkpoint}")
            checkpoint = torch.load(args.resume_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            
            # 如果检查点中包含训练历史，则加载
            if 'train_losses' in checkpoint:
                train_losses = checkpoint['train_losses']
            if 'val_losses' in checkpoint:
                val_losses = checkpoint['val_losses']
                
            print(f"恢复训练从 epoch {start_epoch}, 最佳验证损失: {best_val_loss:.6f}")
        else:
            print(f"检查点文件不存在: {args.resume_checkpoint}, 从头开始训练")
    
    # 创建图表目录
    plots_dir = os.path.join(args.checkpoint_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for features, targets in train_loop:
            features = features.to(device)
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新统计信息
            train_loss += loss.item()
            train_batches += 1
            train_loop.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for features, targets in val_loop:
                features = features.to(device)
                targets = targets.to(device)
                
                # 前向传播
                outputs = model(features)
                loss = criterion(outputs, targets)
                
                # 更新统计信息
                val_loss += loss.item()
                val_batches += 1
                val_loop.set_postfix(loss=loss.item())
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        # 计算每个epoch的训练时间
        epoch_time = time.time() - epoch_start_time
        
        # 打印训练信息
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Time: {epoch_time:.2f}s")
        
        # 绘制并保存当前的损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, f"{args.model_type}_loss_curve_epoch{epoch+1}.png"))
        plt.close()
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.model_type}_best.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, checkpoint_path)
            print(f"保存最佳模型到 {checkpoint_path}")
        
        # 每N个epoch保存一次检查点
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.model_type}_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, checkpoint_path)
            print(f"保存检查点到 {checkpoint_path}")
    
    # 绘制最终的损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.checkpoint_dir, f"{args.model_type}_loss_curve_final.png"))
    plt.close()
    
    print("训练完成!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="回声消除神经网络训练脚本 (使用预处理数据)")
    parser.add_argument('--data_dir', type=str, required=True, help='预处理后的数据目录路径')
    parser.add_argument('--model_type', type=str, default='unet', choices=['unet', 'lstm'], help='模型类型')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--segment_length', type=int, default=None, help='时间段长度，如果为None则使用config中的值')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载的工作线程数')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='检查点保存目录')
    parser.add_argument('--save_interval', type=int, default=10, help='保存检查点的间隔轮数')
    parser.add_argument('--resume', action='store_true', help='是否从检查点恢复训练')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    train(args) 