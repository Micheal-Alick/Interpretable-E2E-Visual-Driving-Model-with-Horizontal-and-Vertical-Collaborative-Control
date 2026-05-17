"""
train.py

功能：在 CARLA 数据集上训练转向（steering）回归模型。

结构说明：
- 在脚本顶部设置若干环境变量以减少内存占用并禁用 albumentations 的在线更新检查。
- 通过 `Trainer` 类封装训练/验证循环、优化器、学习率调度与早停逻辑。
- `create_data_loaders` 用于加载多个 Town 的数据并构造加权采样的 DataLoader。
- `main` 为入口，负责解析命令行参数、构建模型并运行训练流程。
"""

import os
import warnings
import multiprocessing as mp
import argparse
from pathlib import Path

"""
以下一大段的作用是为了训练过程中提升效率和稳定性
"""
# 抑制无关警告
warnings.filterwarnings('ignore')
# 禁用 albumentations 在线版本检查以避免网络请求
os.environ["NO_ALBUMENTATIONS_UPDATE_CHECK"] = "1"
# 限制 PyTorch 内存分配行为，避免部分机器上出现 MemoryError（经验配置）
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TORCH_USE_CUDA_DSA'] = '0'
# 下面这段用于在 Windows 上设置进程启动方式为 spawn，在非 Windows 上尝试 forkserver。
# 放在 __main__ 下以避免在模块 import 时修改全局启动方式。
if __name__ == '__main__':
    try:
        import platform
        if platform.system() == 'Windows':
            mp.set_start_method('spawn', force=True)
        else:
            mp.set_start_method('forkserver', force=True)
    except RuntimeError:
        # 已经设置过启动方式则忽略。
        pass

    # 在主进程中设置 PyTorch 线程数以减少线程开销
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)

    # 建议的默认 batch_size（仅为参考，实际以命令行参数为准）
    batch_size = 32

    # 在主进程中延迟导入部分较重的库（例如 albumentations），以便在模块导入阶段更轻量。
    import albumentations as A

"""
从这里开始才是脚本的核心部分，包含 Trainer 类、数据加载器构建函数和 main 函数。
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import numpy as np

from model import NvidiaModelTransferLearning, NvidiaModel
from dataset_loader import CarlaDataset
from config import config


class Trainer:
    """训练器：封装训练/验证循环、优化器、调度器与早停逻辑。

    主要成员：
    - `model`: PyTorch 模型（会被移动到指定 device）。
    - `train_loader` / `val_loader`: 训练与验证的 DataLoader。
    - `criterion`: 均方误差（MSE）回归损失。
    - `optimizer`: 对 backbone 与 head 使用不同学习率的 Adam 优化器（迁移学习常见做法）。
    - `scheduler`: 基于验证损失的 ReduceLROnPlateau 调度器。
    - 早停相关变量用于判断训练何时结束。
    """

    def __init__(self, model, train_loader, val_loader, device, accumulate_steps=1):
        # 将模型移动到目标设备并保存数据加载器引用
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # 梯度累积步数：每 accumulate_steps 个小 batch 聚合一次梯度并更新一次参数
        self.accumulate_steps = int(accumulate_steps)

        # 损失函数（回归常用 MSE）
        self.criterion = nn.MSELoss()

        # 按名称将模型参数拆分为 backbone 与 head 两组，便于设置不同学习率
        backbone, head = [], []
        for n, p in model.named_parameters():
            (head if n.startswith('regressor') else backbone).append(p)

        # 优化器配置：backbone lr 小，head lr 大
        self.optimizer = optim.Adam([
            # 回归头的参数学习率小
            {'params': backbone, 'lr': 1e-4},
            # 特征提取层的参数学习率大
            {'params': head,     'lr': 1e-3}
        ], weight_decay=1e-4)   # 权重衰减（L2 正则化）有助于防止过拟合

        # 学习率调度器：验证损失不下降时降低学习率
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

        # 早停相关变量
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = config.early_stopping_patience  # 在配置文件中读入，默认10次

    def train_epoch(self):
        """执行一个训练周期并返回平均训练损失。"""
        self.model.train()  # 使用PyTorch的训练模式（启用 dropout 和 batchnorm 的训练行为）,方法来自torch.nn.Module
        total_loss = 0.0    # 总损失累积，用于计算平均损失

        """ 梯度累积：在循环前清零梯度，按 accumulate_steps 更新参数 """
        # 初始化梯度累积计数器和上一个 batch 的索引
        self.optimizer.zero_grad()
        last_idx = -1

        # 可视化训练进度，使用 tqdm 包装训练数据加载器
        with tqdm(self.train_loader, desc="Training", leave=False) as pbar:
            for batch_idx, (images, targets) in enumerate(pbar):
                # 读入图片输入和目标转向角，并移动到指定设备（GPU 或 CPU）
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                # 前向传播计算模型输出和损失，本质是调用PyTorch模型的 forward 方法，并计算输出与目标之间的 MSE 损失
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                # 缩放 loss 以对应等效大 batch,并反向传播计算梯度
                (loss / self.accumulate_steps).backward()

                total_loss += loss.item()   # item()方法将tensor转换为python标量（int, float等）

                # 如果满足累积步数则更新参数并清零梯度
                if (batch_idx + 1) % self.accumulate_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
                last_idx = batch_idx

        # 若最后剩余未更新的梯度，补一次更新
        if (last_idx + 1) % self.accumulate_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return total_loss / len(self.train_loader)

    def validate_epoch(self):
        """执行一个验证周期并返回平均验证损失（无梯度）。"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            # 读入验证数据并计算损失，使用 tqdm 可视化验证进度
            with tqdm(self.val_loader, desc="Validation", leave=False) as pbar:
                for images, targets in pbar:
                    images = images.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                    total_loss += loss.item()
                    pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch, val_loss, filepath):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
        }
        torch.save(checkpoint, filepath)

    def early_stop_check(self, val_loss):
        """若验证损失没有改善则增加计数器，达到 patience 返回 True。"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.patience


def create_data_loaders(batch_size=16, num_workers=2, use_all_cameras=True):
    """创建训练与验证用的 DataLoader。

    步骤概述：
    1. 遍历预定义的 Town 文件夹，使用 `CarlaDataset` 加载每个城市的数据并收集采样器。
    2. 使用 `ConcatDataset` 合并所有子数据集。
    3. 从每个子数据集采样器获取权重并拼接为 combined_weights。
    4. 使用 `random_split` 划分训练/验证子集。
    5. 为训练子集构造针对子集索引的 `WeightedRandomSampler`，并创建 DataLoader。
    """

    datasets_path = Path("data_weathers")   # 读入数据集路径
    town_folders = [
        "dataset_carla_001_Town01",
        "dataset_carla_001_Town02",
        "dataset_carla_001_Town03",
        "dataset_carla_001_Town04",
        "dataset_carla_001_Town05",
    ]

    all_datasets = []
    all_samplers = []

    for town_folder in town_folders:
        town_path = datasets_path / town_folder
        if town_path.exists():
            print(f"Loading dataset: {town_folder}")
            # 实例化data_loader.py中的CarlaDataset类加载数据，并收集采样器以便后续拼接权重
            dataset = CarlaDataset(root_dir=str(town_path), use_all_cameras=use_all_cameras)
            all_datasets.append(dataset)
            all_samplers.append(dataset.sampler)
            print(f"  Loaded {len(dataset)} samples")
        else:
            print(f"Warning: {town_folder} not found")

    if not all_datasets:
        raise ValueError("No datasets found!")

    # 结合不同地图的数据集
    combined_dataset = ConcatDataset(all_datasets)
    print(f"Total combined samples: {len(combined_dataset)}")

    # 拼接所有子数据集的采样权重
    combined_weights = []
    for dataset, sampler in zip(all_datasets, all_samplers):
        dataset_weights = sampler.weights
        combined_weights.extend(dataset_weights)

    # 划分训练/验证子集（固定随机种子以保证可重复性）
    train_size = int(config.train_split_size * len(combined_dataset))
    val_size = len(combined_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        combined_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # 为训练子集创建采样器（映射到 combined_weights）
    train_indices = train_dataset.indices
    train_weights = [combined_weights[i] for i in train_indices]
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=train_weights, num_samples=len(train_weights), replacement=True
    )

    # 构建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_loader, val_loader


def main():
    """主入口：解析参数、构建数据加载器并运行训练过程。"""

    # 用于接收命令行参数，提供训练配置的灵活性
    parser = argparse.ArgumentParser(description="Train CARLA Steering Model")
    # batch_size
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    # 梯度累积步数：每 accumulate_steps 个小 batch 聚合一次梯度并更新一次参数
    parser.add_argument("--accumulate_steps", type=int, default=2, help="Gradient accumulation steps to simulate larger batch")
    # 训练轮数
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    # 学习率
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    # 是否使用所有摄像头
    parser.add_argument("--use_all_cameras", action="store_true", default=True, help="Use all three cameras (center, left, right)")
    parser.add_argument("--run_name", type=str, default="carla_steering", help="Run name for tensorboard")
    # 线程数
    parser.add_argument("--num_workers", type=int, default=24, help="Number of data loader workers")

    # 解析命令行参数
    args = parser.parse_args()

    # 更新全局配置
    config.learning_rate = args.lr

    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建数据加载器
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers, use_all_cameras=args.use_all_cameras
    )

    # 构建模型
    print("Creating model...")
    # 选择具体的模型
    model = NvidiaModel()

    # 打印模型参数信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 创建训练器（核心部分）
    trainer = Trainer(model, train_loader, val_loader, device, accumulate_steps=args.accumulate_steps)

    # 创建TensorBoard 日志写入器，日志目录基于 run_name 参数
    writer = SummaryWriter(f'logs/{args.run_name}')

    # 创建保存目录
    save_dir = Path("checkpoints_weathers") # 模型的保存目录
    save_dir.mkdir(exist_ok=True)

    # 打印训练信息
    print(f"\nStarting training...")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Use all cameras: {args.use_all_cameras}")

    start_time = time.time()
    epoch = 0
    val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # 训练一个 epoch
        train_loss = trainer.train_epoch()

        # 验证
        val_loss = trainer.validate_epoch()

        # 更新调度器
        trainer.scheduler.step(val_loss)

        # 把训练和验证损失以及学习率变化写入 TensorBoard
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalar('Learning_Rate', trainer.optimizer.param_groups[0]['lr'], epoch)

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"LR: {trainer.optimizer.param_groups[0]['lr']:.2e}")

        # 保存最佳模型
        if val_loss < trainer.best_val_loss:
            best_model_path = save_dir / f"{args.run_name}_best.pt"
            trainer.save_checkpoint(epoch, val_loss, best_model_path)
            print(f"New best model saved: {val_loss:.6f}")

        # 每隔 10 个 epoch 保存一次检查点
        if epoch % 10 == 0:
            checkpoint_path = save_dir / f"{args.run_name}_epoch_{epoch}.pt"
            trainer.save_checkpoint(epoch, val_loss, checkpoint_path)

        # 早停检查（多轮验证损失没有改善则停止训练）
        if trainer.early_stop_check(val_loss):
            print(f"Early stopping triggered after {epoch} epochs")
            break

    # 保存最终模型
    final_model_path = save_dir / f"{args.run_name}_final.pt"
    trainer.save_checkpoint(epoch, val_loss, final_model_path)

    writer.close()

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")


if __name__ == '__main__':
    main()