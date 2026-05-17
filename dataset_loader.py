"""
dataset_loader.py
功能：定义用于训练/推理的数据集类与 DataLoader 工具，支持 CARLA 仿真数据与真实世界数据。
添加内容：
 - `RealWorldDataset`：用于从 driving_dataset/data.txt 加载单张图片与转向角（度->弧度）。
 - `CarlaDataset`：用于加载 CARLA 导出的 CSV（包含三路相机）并提供数据增强与平衡采样。
 - 辅助函数：`get_inference_dataset`, `get_full_dataset_loader`。
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
import os
from pathlib import Path
import albumentations as A  # 数据增强库
from albumentations.pytorch import ToTensorV2

# 相机位置对应的转向角偏移（如果使用左/右摄像头时可用来矫正）
CAM_OFFSET = {"left": 0.15, "center": 0.0, "right": -0.15}

class RealWorldDataset(Dataset):
    """
    实际道路数据集（用于推理或用真实数据评估模型）

    结构假设：root_dir 下存在一个 `data.txt`，每行记录：`image_filename steering_angle_in_degrees`
    - `__init__`: 解析 data.txt，将存在的图片记录加载到内存（路径与弧度化后的转向角）。
    - `__len__`: 返回样本数量。
    - `__getitem__`: 读取图片、转换为 RGB、按网络输入尺寸缩放与归一化，返回 (image_tensor, steering_angle_tensor)。

    设计说明：模型期望转向角以弧度为单位，因此在加载阶段将度转换为弧度；推理时仅做最小的数据处理（Resize + Normalize + ToTensor）。
    """

    def __init__(self, root_dir="driving_dataset"):
        self.root_dir = Path(root_dir)  # 读取数据集路径

        # 解析 data.txt，将每条存在的图片加入到 self.data 列表
        data_file = self.root_dir / "data.txt"
        self.data = []

        if data_file.exists():
            with open(data_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        filename = parts[0]
                        steering_angle = float(parts[1])
                        # 将转向角从度 -> 弧度（训练/推理中模型使用弧度）
                        steering_angle_rad = np.radians(steering_angle)

                        image_path = self.root_dir / filename
                        # 仅当图片文件存在时才加入列表，避免后续加载异常
                        if image_path.exists():
                            self.data.append({
                                'filename': filename,
                                'steering_angle': steering_angle_rad,
                                'image_path': image_path
                            })

        # 推理时的最小预处理：调整到 Nvidia 模型要求的 (66, 200)、归一化并转为 Tensor
        self.transform = A.Compose([
            A.Resize(66, 200),  # Nvidia 模型输入尺寸
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # ImageNet 预训练模型常用的归一化参数，rgb三通道各自的均值和标准差
            ToTensorV2()
        ])

        print(f"Loaded {len(self.data)} real-world samples from {root_dir}")
        # 读入steering_angle的范围以便观察数据分布（以度为单位更直观）
        if len(self.data) > 0:
            steering_angles = [item['steering_angle'] for item in self.data]
            print(f"  Steering angle range: [{np.degrees(min(steering_angles)):.1f}°, {np.degrees(max(steering_angles)):.1f}°]")

    def __len__(self):
        """返回数据集样本数"""
        return len(self.data)

    def __getitem__(self, idx):
        """按索引读取样本并返回 (image_tensor, steering_angle_tensor)

        实现步骤：
        1. 读取图片（cv2），若失败抛出异常以便定位缺失文件。
        2. BGR->RGB 转换（PyTorch 常用），应用定义好的 transform。
        3. 将转向角包装为 float32 的 torch.tensor 返回。
        """
        item = self.data[idx]

        image = cv2.imread(str(item['image_path']))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {item['image_path']}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.transform(image=image)   # albumentations库接口
        image = transformed['image']

        steering_angle = item['steering_angle']

        return image, torch.tensor(steering_angle, dtype=torch.float32)

class CarlaDataset(Dataset):
    """
    CARLA 仿真数据集类

    用途：从指定目录加载 CARLA 导出的 CSV（如 `steering_data.csv`）和对应的三路摄像头图片。
    - 支持按相机位置筛选（center/left/right）或使用全部相机。
    - 提供训练时的增强（ReplayCompose，可检测是否发生 HorizontalFlip）以及简单的验证变换。
    - 提供 `_create_balanced_sampler` 方法用于生成按转向角分布加权的采样器，缓解数据不平衡。
    """

    def __init__(self, root_dir, csv_file="steering_data.csv", use_all_cameras=True):
        self.root_dir = Path(root_dir)
        self.use_all_cameras = use_all_cameras  # 是否shy使用全部相机数据（默认True），否则仅使用center摄像头数据以简化训练与验证过程
        
        # 从CSV表格加载数据，CSV应包含至少以下列：camera_position, frame_filename, steering_angle
        csv_path = self.root_dir / csv_file
        self.data = pd.read_csv(csv_path)   # panda库读入
        
        # Filter data if needed
        if not use_all_cameras:
            self.data = self.data[self.data['camera_position'] == 'center'].reset_index(drop=True)
        
        # 图片路径字典
        self.image_dirs = {
            'center': self.root_dir / 'images_center',
            'left': self.root_dir / 'images_left', 
            'right': self.root_dir / 'images_right'        }
        
        # 数据增强流程 - using ReplayCompose to track transformations
        self.transform = A.ReplayCompose([
            A.Resize(66, 200),  # 修改模型输入像素尺寸为 (66, 200)
            A.OneOf([   # 按一定概率对输入图片执行随机数据增强变化
                # 随机调整亮度、对比度
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                # 随机改变色相、饱和度与明度
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
                # 随机调整gamma值
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            ], p=0.3),
            # 0.5概率图像做水平翻转
            A.HorizontalFlip(p=0.5),
            # 对每个通道归一化
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # 转换为PyTorch Tensor张量
            ToTensorV2()
        ]) # type: ignore
        
        # 为验证 / 测试 / 推理阶段提供“确定性”的预处理管线（无随机增强），保证输入分布一致且标签不被改变。
        self.transform_val = A.Compose([
            A.Resize(66, 200),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()        ])
          # 创建了一个按权重随机采样的采样器，训练时缓解转向角分布不平衡
        self.sampler = self._create_balanced_sampler()
        
        print(f"Loaded {len(self.data)} samples from {root_dir}")
    
    def _create_balanced_sampler(self):
        """
        基于转向角直方图创建加权采样器（WeightedRandomSampler）

        实现要点：
        1. 将所有样本的 `steering_angle` 提取为 numpy 数组并构建直方图（bins=41，范围为 -0.4 至 0.4）。
        2. 对每个样本找到其在直方图中的箱索引，并用该箱的频率计算该样本的权重（取逆频率）。
        3. 归一化权重并创建 PyTorch 的 `WeightedRandomSampler`，用于 DataLoader 的采样以缓解数据偏斜。

        返回：一个 `torch.utils.data.WeightedRandomSampler` 实例。
        """
        steering_angles = np.array(self.data['steering_angle'].values, dtype=np.float32)

        # 将角度分箱以估算每个区间的样本数量（频率）
        hist, bins = np.histogram(steering_angles, bins=41, range=(-0.4, 0.4))

        # 计算每个样本对应的箱索引（0-based）
        bin_indices = np.digitize(steering_angles, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(hist) - 1)

        # 以箱频率的逆作为权重（频率低的样本权重高），并避免除零
        weights = 1.0 / (hist[bin_indices] + 1e-6)

        # 归一化权重（可选，使权重和等于样本数）
        weights = weights / weights.sum() * len(weights)

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )

        # 打印统计信息以便调试与观察分布
        print(f"  Steering angle distribution:")
        print(f"    Range: [{steering_angles.min():.3f}, {steering_angles.max():.3f}]")
        print(f"    Mean: {steering_angles.mean():.3f}, Std: {steering_angles.std():.3f}")
        print(f"    Histogram (bins={len(hist)}): min={hist.min()}, max={hist.max()}")
        print(f"    Bin indices range: [{bin_indices.min()}, {bin_indices.max()}]")

        return sampler
    
    def __len__(self):
        """返回数据集样本数（行数）"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """按索引读取 CARLA 数据并返回 (image_tensor, steering_angle_tensor)

        实现要点：
        - 从 CSV 行获取相机位置与帧文件名，构建图片路径并读取。
        - 将 BGR->RGB，应用 ReplayCompose 变换并检测是否发生水平翻转以调整转向角符号。
        """
        row = self.data.iloc[idx]
        
        # Get image path
        camera_pos = row['camera_position']
        filename = row['frame_filename']
        image_path = self.image_dirs[camera_pos] / filename
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get steering angle with camera-specific correction
        steering_angle = float(row['steering_angle'])
        # Apply camera offset correction
        #steering_angle += CAM_OFFSET[camera_pos]
        
        # 应用带有 replay 追踪的变换（便于判断哪些随机变换被应用）
        transformed = self.transform(image=image)
        image = transformed['image']

        # 如果发生水平翻转，转向角需要取反以保持标签与图像一致
        replay_data = transformed.get('replay', {})
        if replay_data:
            for transform_info in replay_data.get('transforms', []):
                if transform_info['__class_fullname__'] == 'HorizontalFlip' and transform_info['applied']:
                    steering_angle = -steering_angle
                    break
        
        return image, torch.tensor(steering_angle, dtype=torch.float32)
    


def get_inference_dataset(dataset_type='carla_001'):
    """根据字符串标识返回相应的数据集实例（用于推理或快速取样）。

    支持的 `dataset_type`:
    - 'carla_001': 返回内置路径下的 CarlaDataset（data_weathers/dataset_carla_001_Town01）。
    - 'real_world': 返回 RealWorldDataset（driving_dataset）。
    - 'real_dataset': 将 driving_dataset 当作 CarlaDataset 加载（用于兼容已导出的格式）。
    """
    if dataset_type == 'carla_001':
        return CarlaDataset(
            root_dir="data_weathers/dataset_carla_001_Town01",
            use_all_cameras=True
        )
    elif dataset_type == 'real_world':
        return RealWorldDataset(root_dir="driving_dataset")
    elif dataset_type == 'real_dataset':
        return CarlaDataset(
            root_dir="driving_dataset",
            use_all_cameras=True
        )
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}. Valid options: 'carla_001', 'real_world', 'real_dataset'")

def get_full_dataset_loader(dataset_type='carla_001') -> DataLoader:
    """快捷构造 DataLoader 的工厂函数。

    - 默认 batch_size=64, shuffle=False, num_workers=2（可按需在调用处修改）。
    - 适用于快速获取用于推理或验证的 DataLoader。
    """
    ds = get_inference_dataset(dataset_type)
    return DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

