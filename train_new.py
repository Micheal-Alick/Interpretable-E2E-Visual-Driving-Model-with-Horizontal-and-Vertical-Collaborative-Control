"""
train_new.py

多任务训练入口：共享主干 + 三个任务头，预测 [steer, throttle, brake]
可选输入上一时刻速度（prev_speed_kmh）。
"""

import os
import warnings
import multiprocessing as mp
import time
import argparse # 用于解析命令行参数
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm   # 进度条库，训练/验证循环中可视化进度和损失

from config import config
from model_multitask import MultiTaskNvidiaModel
from dataset_loader_multitask import CarlaMultiTaskDataset


# 运行期环境配置：减少无关告警并控制部分内存行为。
warnings.filterwarnings("ignore")
os.environ["NO_ALBUMENTATIONS_UPDATE_CHECK"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TORCH_USE_CUDA_DSA"] = "0"


# 训练器：封装多任务损失计算、训练/验证循环、检查点与早停逻辑。
class MultiTaskTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        accumulate_steps=1,
        # 多任务Loss权重系数：根据实际需求调整不同任务的损失贡献度
        # 没修改之前的参数[1,1,1.2,0.35,1,1]
        # 削弱分类增强回归重视程度和刹车安全的参数[1,1,1.35,0.3,0.5,0.6]
        lambda_steer=1.0,
        lambda_throttle=1.0,
        lambda_brake=1.35,   # 更重视刹车预测，因其安全相关性更高
        lambda_conflict=0.30,    # 约束项权重，抑制油门和刹车同时输出较大值的情况
        lambda_tl=0.5,
        lambda_stop=0.5,
        tl_class_weight=None,   # traffic-light 多分类类别权重
        stop_pos_weight=None,   # is_stopped 正类权重（BCE pos_weight）
        use_speed_input=True,   # 使用上一时刻速度输入
        use_amp=True,           # CUDA 上默认开启 FP16 混合精度
    ):
        # 1) 基础状态与超参数。
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.accumulate_steps = max(1, int(accumulate_steps))
        self.use_speed_input = use_speed_input

        self.lambda_steer = float(lambda_steer)
        self.lambda_throttle = float(lambda_throttle)
        self.lambda_brake = float(lambda_brake)
        self.lambda_conflict = float(lambda_conflict)
        self.lambda_tl = float(lambda_tl)
        self.lambda_stop = float(lambda_stop)

        # 分类任务权重：用于缓解类别不平衡，权重张量需放在训练设备上。
        self.tl_class_weight = None
        if tl_class_weight is not None:
            tl_w = np.asarray(tl_class_weight, dtype=np.float32)
            if tl_w.ndim == 1 and tl_w.size > 0:
                self.tl_class_weight = torch.tensor(tl_w, dtype=torch.float32, device=device)

        self.stop_pos_weight = None
        if stop_pos_weight is not None:
            spw = float(stop_pos_weight)
            if spw > 0.0:
                self.stop_pos_weight = torch.tensor([spw], dtype=torch.float32, device=device)

        self.use_amp = bool(use_amp and device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # 2) 损失函数：回归任务使用 MSE。
        self.criterion = nn.MSELoss()
        # 分类任务损失：显式使用不平衡权重（若可用）。
        # 交通灯分类使用 CrossEntropyLoss，类别权重通过 tl_class_weight 传入。
        self.ce_loss = nn.CrossEntropyLoss(weight=self.tl_class_weight)
        # 停驶二分类使用 BCEWithLogitsLoss，正类权重通过 stop_pos_weight 传入。
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.stop_pos_weight)

        # 3) 按模块分组参数，给各自回归头、共享回归层、特征层设置不同学习率。
        backbone, shared_head, task_heads = [], [], []
        for name, param in model.named_parameters():
            if name.startswith("conv_layers"):
                backbone.append(param)
            elif name.startswith("shared_head"):
                shared_head.append(param)
            else:
                task_heads.append(param)

        # 优化器：任务头学习率最高，主干最低，符合迁移学习常见策略。
        self.optimizer = optim.Adam(
            [
                {"params": backbone, "lr": 1e-4},   # 各自回归头
                {"params": shared_head, "lr": 5e-4},    # 共享回归层
                {"params": task_heads, "lr": 1e-3}, # 特征提取层
            ],
            weight_decay=1e-4,  # L2 正则化
        )

        # 调度器：验证总损失长期不下降时自动降学习率。
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )

        # 早停状态。
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.patience = config.early_stopping_patience

    def _compute_losses(self, outputs, targets):
        # 新模型返回字典 outputs，targets 为回归标签 tensor
        steer_pred = outputs["steer"]
        throttle_pred = outputs["throttle"]
        brake_pred = outputs["brake"]

        steer_gt = targets[:, 0]
        throttle_gt = targets[:, 1]
        brake_gt = targets[:, 2]

        # 回归损失
        steer_loss = self.criterion(steer_pred, steer_gt)
        throttle_loss = self.criterion(throttle_pred, throttle_gt)
        brake_loss = self.criterion(brake_pred, brake_gt)

        # 交通灯分类与停驶二分类标签由外部传入（在调用处）
        tl_logits = outputs.get("tl_logits")
        stop_logit = outputs.get("stop_logit")

        # conflict loss: 抑制油门与刹车同时较大
        conflict_loss = torch.mean(throttle_pred * brake_pred)

        # classification losses will be computed by caller supplying labels
        # 这里返回回归和分类子损失占位，实际分类损失由外部合入
        total_loss = (
            self.lambda_steer * steer_loss
            + self.lambda_throttle * throttle_loss
            + self.lambda_brake * brake_loss
            + self.lambda_conflict * conflict_loss
        )

        losses = {
            "total": total_loss,
            "steer": steer_loss.detach(),
            "throttle": throttle_loss.detach(),
            "brake": brake_loss.detach(),
            "conflict": conflict_loss.detach(),
        }
        return total_loss, losses

    def train_epoch(self):
        # 训练阶段：启用梯度与参数更新。
        self.model.train()

        total_stats = {"total": 0.0, "steer": 0.0, "throttle": 0.0, "brake": 0.0, "conflict": 0.0}

        self.optimizer.zero_grad()
        last_idx = -1

        # 主训练循环：读取 batch、前向、损失、反向、梯度累积更新。
        with tqdm(self.train_loader, desc="Training", leave=False) as pbar:
            for batch_idx, (images, targets, prev_speed, tl_labels, is_stopped) in enumerate(pbar):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                prev_speed = prev_speed.to(self.device, non_blocking=True)
                tl_labels = tl_labels.to(self.device, non_blocking=True)
                is_stopped = is_stopped.to(self.device, non_blocking=True)

                # AMP 前向：在 CUDA 上使用 FP16 自动混合精度。
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    if self.use_speed_input:
                        outputs = self.model(images, prev_speed_kmh=prev_speed)
                    else:
                        outputs = self.model(images)

                    # 计算回归与分类损失
                    total_loss, losses = self._compute_losses(outputs, targets)

                    # traffic-light 分类损失
                    tl_loss = self.ce_loss(outputs["tl_logits"], tl_labels.long())
                    stop_loss = self.bce_loss(outputs["stop_logit"].view(-1), is_stopped.float())

                    # 把分类损失按权重合入总损失
                    total_loss = (
                        total_loss
                        + self.lambda_tl * tl_loss
                        + self.lambda_stop * stop_loss
                    )

                # 梯度累积：将 loss 缩放后再反向传播。
                self.scaler.scale(total_loss / self.accumulate_steps).backward()

                if (batch_idx + 1) % self.accumulate_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                total_stats["total"] += float(total_loss.item())
                total_stats["steer"] += float(losses["steer"].item())
                total_stats["throttle"] += float(losses["throttle"].item())
                total_stats["brake"] += float(losses["brake"].item())
                total_stats["conflict"] += float(losses["conflict"].item())
                # 分类子损失统计（detach）
                total_stats.setdefault("tl", 0.0)
                total_stats.setdefault("stop", 0.0)
                total_stats["tl"] += float(tl_loss.detach().item())
                total_stats["stop"] += float(stop_loss.detach().item())

                pbar.set_postfix(
                    {
                        "L": f"{total_loss.item():.4f}",
                        "S": f"{losses['steer'].item():.4f}",
                        "T": f"{losses['throttle'].item():.4f}",
                        "B": f"{losses['brake'].item():.4f}",
                        "TL": f"{tl_loss.item():.4f}",
                        "ST": f"{stop_loss.item():.4f}",
                    }
                )
                last_idx = batch_idx

        # 若最后不足一个累积步，补一次参数更新。
        if (last_idx + 1) % self.accumulate_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        # 返回 epoch 平均统计。
        n = max(1, len(self.train_loader))
        return {k: v / n for k, v in total_stats.items()}

    def validate_epoch(self):
        # 验证阶段：关闭梯度，仅做前向与统计。
        self.model.eval()

        total_stats = {"total": 0.0, "steer": 0.0, "throttle": 0.0, "brake": 0.0, "conflict": 0.0}

        with torch.no_grad():
            # 验证循环：与训练一致，但无反向传播与参数更新。
            with tqdm(self.val_loader, desc="Validation", leave=False) as pbar:
                for images, targets, prev_speed, tl_labels, is_stopped in pbar:
                    images = images.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    prev_speed = prev_speed.to(self.device, non_blocking=True)
                    tl_labels = tl_labels.to(self.device, non_blocking=True)
                    is_stopped = is_stopped.to(self.device, non_blocking=True)

                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        if self.use_speed_input:
                            outputs = self.model(images, prev_speed_kmh=prev_speed)
                        else:
                            outputs = self.model(images)

                        total_loss, losses = self._compute_losses(outputs, targets)

                        tl_loss = self.ce_loss(outputs["tl_logits"], tl_labels.long())
                        stop_loss = self.bce_loss(outputs["stop_logit"].view(-1), is_stopped.float())

                        total_loss = total_loss + self.lambda_tl * tl_loss + self.lambda_stop * stop_loss

                    total_stats["total"] += float(total_loss.item())
                    total_stats["steer"] += float(losses["steer"].item())
                    total_stats["throttle"] += float(losses["throttle"].item())
                    total_stats["brake"] += float(losses["brake"].item())
                    total_stats["conflict"] += float(losses["conflict"].item())
                    total_stats.setdefault("tl", 0.0)
                    total_stats.setdefault("stop", 0.0)
                    total_stats["tl"] += float(tl_loss.detach().item())
                    total_stats["stop"] += float(stop_loss.detach().item())

                    pbar.set_postfix(
                        {
                            "L": f"{total_loss.item():.4f}",
                            "S": f"{losses['steer'].item():.4f}",
                            "T": f"{losses['throttle'].item():.4f}",
                            "B": f"{losses['brake'].item():.4f}",
                            "TL": f"{tl_loss.item():.4f}",
                            "ST": f"{stop_loss.item():.4f}",
                        }
                    )

        # 返回 epoch 平均统计。
        n = max(1, len(self.val_loader))
        return {k: v / n for k, v in total_stats.items()}

    def save_checkpoint(self, epoch, val_stats, filepath):
        # 保存完整训练状态，便于断点恢复和离线分析。
        checkpoint = {
            "epoch": epoch,
            # 模型参数状态字典，包含所有可学习参数的当前值。
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            # 验证统计数据，包含总损失和各子损失的数值，便于后续分析和可视化。
            "val_total_loss": val_stats["total"],
            "val_steer_loss": val_stats["steer"],
            "val_throttle_loss": val_stats["throttle"],
            "val_brake_loss": val_stats["brake"],
            "val_conflict_loss": val_stats["conflict"],
            "val_tl_loss": val_stats.get("tl", 0.0),
            "val_stop_loss": val_stats.get("stop", 0.0),
        }
        torch.save(checkpoint, filepath)    # 保存模型参数到指定路径（.pt 文件）

    def early_stop_check(self, val_total_loss):
        # 早停判定：验证总损失若改进则清零计数，否则累计。
        if val_total_loss < self.best_val_loss:
            self.best_val_loss = val_total_loss
            self.patience_counter = 0
            return False
        self.patience_counter += 1
        return self.patience_counter >= self.patience


def create_data_loaders(batch_size=16, num_workers=4, use_all_cameras=True, dataset_folder_name="dataset_carla_visual_Town01", datasets_base="data_weathers"):
    # 1) 预定义多个 Town 数据目录并逐个加载。
    # 使用命令行传入的基目录（默认为 data_weather），若不存在则兼容回退到原来的 data_weathers
    datasets_base_path = Path(datasets_base)
    if not datasets_base_path.exists():
        alt_base = Path("data_weathers")
        if alt_base.exists():
            print(f"Warning: base datasets folder '{datasets_base}' not found, falling back to 'data_weathers'")
            datasets_base_path = alt_base

    town_folders = [dataset_folder_name]

    all_datasets = []
    all_samplers = []
    combined_tl_labels = []
    combined_stop_labels = []

    for town_folder in town_folders:
        town_path = datasets_base_path / town_folder
        if town_path.exists():
            print(f"从 {town_folder} 载入数据")
            dataset = CarlaMultiTaskDataset(
                root_dir=str(town_path),
                use_all_cameras=use_all_cameras,
            )
            all_datasets.append(dataset)
            all_samplers.append(dataset.sampler)
            # 记录分类标签，用于后续按训练子集统计类别权重。
            combined_tl_labels.extend(dataset.data["tl_label"].astype(int).tolist())
            combined_stop_labels.extend((dataset.data["is_stopped"].astype(float).values > 0.5).astype(np.int64).tolist())
            print(f" 载入了 {len(dataset)} 个样本数据")
        else:
            print(f"Warning: {town_folder} not found (checked {town_path})")

    if not all_datasets:
        raise ValueError("No datasets found!")

    # 2) 合并多城市场景数据。
    combined_dataset = ConcatDataset(all_datasets)
    print(f"合并后总样本数: {len(combined_dataset)}")

    # 3) 合并各子数据集权重，用于后续子集重采样。
    combined_weights = []
    for sampler in all_samplers:
        combined_weights.extend(sampler.weights)

    # 4) 划分训练/验证集（固定随机种子保证可复现）。
    train_size = int(config.train_split_size * len(combined_dataset))
    val_size = len(combined_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        combined_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # 5) 训练集使用加权采样以缓解 steering 分布偏斜。
    train_indices = train_dataset.indices
    train_weights = [combined_weights[i] for i in train_indices]
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True,
    )

    # 6) 基于训练子集统计分类任务权重，缓解红灯/停驶等长尾样本影响。
    train_tl_labels = np.asarray([combined_tl_labels[i] for i in train_indices], dtype=np.int64)
    train_stop_labels = np.asarray([combined_stop_labels[i] for i in train_indices], dtype=np.int64)

    tl_counts = np.bincount(train_tl_labels, minlength=4).astype(np.float32)
    tl_class_weight = np.zeros(4, dtype=np.float32)
    valid_cls = tl_counts > 0
    if np.any(valid_cls):
        tl_class_weight[valid_cls] = float(train_tl_labels.size) / (float(np.sum(valid_cls)) * tl_counts[valid_cls])
        # 归一化到均值 1，避免整体 loss 量纲被放大。
        tl_class_weight[valid_cls] = tl_class_weight[valid_cls] / float(np.mean(tl_class_weight[valid_cls]))

    stop_pos = int(np.sum(train_stop_labels == 1))
    stop_neg = int(np.sum(train_stop_labels == 0))
    stop_pos_weight = 1.0
    if stop_pos > 0 and stop_neg > 0:
        stop_pos_weight = float(stop_neg) / float(stop_pos)

    # 7) 构建 DataLoader。
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

    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")
    print(f"TL类别计数(训练集): {tl_counts.astype(np.int64).tolist()}")
    print(f"TL类别权重(训练集): {[round(float(x), 4) for x in tl_class_weight.tolist()]}")
    print(f"Stop样本计数(训练集): neg={stop_neg}, pos={stop_pos}, pos_weight={stop_pos_weight:.4f}")

    loss_weight_info = {
        "tl_class_weight": tl_class_weight.tolist(),
        "stop_pos_weight": float(stop_pos_weight),
        "tl_counts": tl_counts.astype(np.int64).tolist(),
        "stop_counts": {"neg": stop_neg, "pos": stop_pos},
    }
    return train_loader, val_loader, loss_weight_info


def main():
    # 1) 命令行参数：训练超参与任务权重。
    parser = argparse.ArgumentParser(description="Train CARLA Multi-Task Driving Model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--accumulate_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for task heads")
    parser.add_argument("--use_all_cameras", action="store_true", default=True, help="Use center/left/right cameras")
    parser.add_argument("--run_name", type=str, default="carla_multitask", help="Run name for tensorboard/checkpoints")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--use_speed_input", action="store_true", help="Use prev_speed_kmh as extra model input")
    parser.add_argument("--no_amp", action="store_true", help="Disable FP16 mixed precision training")

    parser.add_argument("--lambda_steer", type=float, default=1.0)
    parser.add_argument("--lambda_throttle", type=float, default=1.0)
    parser.add_argument("--lambda_brake", type=float, default=1.35)
    parser.add_argument("--lambda_conflict", type=float, default=0.25)
    parser.add_argument("--lambda_tl", type=float, default=0.5, help="Traffic-light classification loss weight")
    parser.add_argument("--lambda_stop", type=float, default=0.5, help="Is-stopped loss weight")
    parser.add_argument("--dataset_path", type=str, default="dataset_carla_visual_Town01", help="Dataset folder name under data_weathers")
    parser.add_argument("--modelsave_path", type=str, default=None, help="Model save folder name under checkpoints_weathers_new (default: run_name)")
    
    # 把命令行参数解析为 args 对象，后续通过 args.参数名 访问各个参数值。
    args = parser.parse_args()
    if args.modelsave_path is None:
        args.modelsave_path = args.run_name

    # 2) 设备与数据。
    config.learning_rate = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用硬件: {device}")
    print(f"是否使用速度输入: {args.use_speed_input}")
    print(f"是否启用AMP(FP16): {not args.no_amp and device.type == 'cuda'}")

    train_loader, val_loader, loss_weight_info = create_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_all_cameras=args.use_all_cameras,
        dataset_folder_name=args.dataset_path,
    )

    # 3) 构建模型并打印参数规模。
    model = MultiTaskNvidiaModel(
        pretrained=True,
        freeze_features=False,
        use_speed_input=args.use_speed_input,   # 通过命令行读入
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 4) 初始化训练器与日志系统。
    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        accumulate_steps=args.accumulate_steps,
        lambda_steer=args.lambda_steer,
        lambda_throttle=args.lambda_throttle,
        lambda_brake=args.lambda_brake,
        lambda_conflict=args.lambda_conflict,
        lambda_tl=args.lambda_tl,
        lambda_stop=args.lambda_stop,
        tl_class_weight=loss_weight_info.get("tl_class_weight"),
        stop_pos_weight=loss_weight_info.get("stop_pos_weight"),
        use_speed_input=args.use_speed_input,
        use_amp=(not args.no_amp),
    )

    # 用命令行 lr 覆盖默认 task-head 学习率
    trainer.optimizer.param_groups[2]["lr"] = args.lr

    writer = SummaryWriter(f"logs/{args.run_name}")

    # 5) 准备模型保存目录。
    save_dir = Path("checkpoints_weathers_new") / args.modelsave_path
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n开始多任务训练...")
    print(f"总训练轮次: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"特征层学习率: {args.lr}")
    print(f"使用全部摄像头数据: {args.use_all_cameras}")
    print(f"数据集路径: data_weathers/{args.dataset_path}")
    print(f"模型保存路径: checkpoints_weathers_new/{args.modelsave_path}")
    print(
        "Loss权重: "
        f"steer={trainer.lambda_steer:.3f}, "
        f"throttle={trainer.lambda_throttle:.3f}, "
        f"brake={trainer.lambda_brake:.3f}, "
        f"conflict={trainer.lambda_conflict:.3f}, "
        f"tl={trainer.lambda_tl:.3f}, "
        f"stop={trainer.lambda_stop:.3f}"
    )
    print(f"TL类别权重(用于CE): {[round(float(x), 4) for x in loss_weight_info.get('tl_class_weight', [])]}")
    print(f"Stop pos_weight(用于BCE): {float(loss_weight_info.get('stop_pos_weight', 1.0)):.4f}")

    start_time = time.time()
    epoch = 0
    best_stats = None

    # 6) 主训练循环：训练 -> 验证 -> 调度 -> 记录 -> 保存 -> 早停。
    for epoch in range(1, args.epochs + 1):
        print(f"\n当前轮次： {epoch}/{args.epochs}")

        train_stats = trainer.train_epoch()
        val_stats = trainer.validate_epoch()

        trainer.scheduler.step(val_stats["total"])

        # 训练/验证损失写入 TensorBoard
        writer.add_scalars(
            "Loss/Train",
            {
                "total": train_stats["total"],
                "steer": train_stats["steer"],
                "throttle": train_stats["throttle"],
                "brake": train_stats["brake"],
                "conflict": train_stats["conflict"],
                "tl": train_stats.get("tl", 0.0),
                "stop": train_stats.get("stop", 0.0),
            },
            epoch,
        )
        writer.add_scalars(
            "Loss/Val",
            {
                "total": val_stats["total"],
                "steer": val_stats["steer"],
                "throttle": val_stats["throttle"],
                "brake": val_stats["brake"],
                "conflict": val_stats["conflict"],
                "tl": val_stats.get("tl", 0.0),
                "stop": val_stats.get("stop", 0.0),
            },
            epoch,
        )
        
        # 写入学习率
        writer.add_scalar("Learning_Rate/backbone", trainer.optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("Learning_Rate/shared", trainer.optimizer.param_groups[1]["lr"], epoch)
        writer.add_scalar("Learning_Rate/heads", trainer.optimizer.param_groups[2]["lr"], epoch)

        print(
            "Train: "
            f"total={train_stats['total']:.6f}, "
            f"steer={train_stats['steer']:.6f}, "
            f"throttle={train_stats['throttle']:.6f}, "
            f"\nbrake={train_stats['brake']:.6f}, "
            f"conflict={train_stats['conflict']:.6f}, "
            f"tl={train_stats.get('tl', 0.0):.6f}, "
            f"stop={train_stats.get('stop', 0.0):.6f}"
        )
        print(
            "Val:   "
            f"total={val_stats['total']:.6f}, "
            f"steer={val_stats['steer']:.6f}, "
            f"throttle={val_stats['throttle']:.6f}, "
            f"\nbrake={val_stats['brake']:.6f}, "
            f"conflict={val_stats['conflict']:.6f}, "
            f"tl={val_stats.get('tl', 0.0):.6f}, "
            f"stop={val_stats.get('stop', 0.0):.6f}"
        )
        print(f"LR(backbone/shared/heads): {trainer.optimizer.param_groups[0]['lr']:.2e} / {trainer.optimizer.param_groups[1]['lr']:.2e} / {trainer.optimizer.param_groups[2]['lr']:.2e}")

        if val_stats["total"] < trainer.best_val_loss:
            best_path = save_dir / f"{args.run_name}_best.pt"
            trainer.save_checkpoint(epoch, val_stats, best_path)
            best_stats = val_stats
            print(f"New best model saved: total={val_stats['total']:.6f}")

        if epoch % 10 == 0:
            ckpt_path = save_dir / f"{args.run_name}_epoch_{epoch}.pt"
            trainer.save_checkpoint(epoch, val_stats, ckpt_path)

        if trainer.early_stop_check(val_stats["total"]):
            print(f"Early stopping triggered after {epoch} epochs")
            break

    # 7) 训练结束：保存 final 并输出总结。
    final_path = save_dir / f"{args.run_name}_final.pt"
    trainer.save_checkpoint(epoch, val_stats, final_path)

    writer.close()

    elapsed = time.time() - start_time
    print(f"\n训练完成，耗时 {elapsed:.2f} 秒")
    print(f"最佳验证总损失: {trainer.best_val_loss:.6f}")
    if best_stats is not None:
        print(
            "Best val breakdown: "
            f"steer={best_stats['steer']:.6f}, "
            f"throttle={best_stats['throttle']:.6f}, "
            f"brake={best_stats['brake']:.6f}, "
            f"conflict={best_stats['conflict']:.6f}"
        )


if __name__ == "__main__":
    # Windows 下使用 spawn，避免 DataLoader 多进程兼容问题。
    try:
        import platform
        if platform.system() == "Windows":
            mp.set_start_method("spawn", force=True)
        else:
            mp.set_start_method("forkserver", force=True)
    except RuntimeError:
        pass

    # 控制线程数，避免 CPU 线程过多导致资源争用。
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)

    main()
