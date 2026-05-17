import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


# 多任务数据集：读取单帧图像并返回三任务标签 + 上一时刻速度。
class CarlaMultiTaskDataset(Dataset):
    """CARLA 多任务数据集。

    返回:
    - image: (3, 66, 200)
    - target: [steer, throttle, brake]
    - prev_speed: 上一时刻速度(km/h)
    """

    def __init__(self, root_dir, csv_file="steering_data.csv", use_all_cameras=True):
        # 1) 路径与基础配置。
        self.root_dir = Path(root_dir)
        self.use_all_cameras = use_all_cameras  # 使用三路相机数据

        # 2) 读取标注 CSV（包含 steer/throttle/brake/speed/camera/frame）。
        csv_path = self.root_dir / csv_file
        self.data = pd.read_csv(csv_path)   # 读取 CSV 文件到 self.data DataFrame 中

        # 可选仅使用中置相机，便于与某些单相机训练设定保持一致。
        if not use_all_cameras:
            self.data = self.data[self.data["camera_position"] == "center"].reset_index(drop=True)

        """
        为每个 camera_position 计算上一时刻速度
        若为该相机首帧，则用当前速度补齐，避免产生 NaN
        """
        # 把数据按摄像头 + 帧号 正确排序，保证时序不乱。
        self.data = self.data.sort_values(["camera_position", "frame_number"]).reset_index(drop=True)
        # 计算上一时刻速度，按摄像头分组后 shift(1) 把speed_kmh列下移一行实现错位对齐。
        self.data["prev_speed_kmh"] = (
            self.data.groupby("camera_position")["speed_kmh"].shift(1)
        )
        # 替换首帧的 NaN 值为当前速度，确保每行都有一个有效的上一时刻速度值。
        self.data["prev_speed_kmh"] = self.data["prev_speed_kmh"].fillna(self.data["speed_kmh"])

        # 3) 三路相机图像目录映射。
        self.image_dirs = {
            "center": self.root_dir / "images_center",
            "left": self.root_dir / "images_left",
            "right": self.root_dir / "images_right",
        }

        # 4) 解析新增的 traffic_light_state 与 is_stopped 字段（若不存在则补齐默认值）。
        if "traffic_light_state" not in self.data.columns:
            # 若 CSV 中未包含此列，则默认全部 unknown
            self.data["traffic_light_state"] = "unknown"
        else:
            self.data["traffic_light_state"] = self.data["traffic_light_state"].fillna("unknown").astype(str)

        if "is_stopped" not in self.data.columns:
            self.data["is_stopped"] = 0
        else:
            # 将可能的字符串 '0'/'1' 转为数值
            self.data["is_stopped"] = self.data["is_stopped"].fillna(0).astype(float)

        # 将 traffic_light_state 文本映射为类别索引：0=red,1=green,2=yellow,3=unknown
        def _tl_to_idx(v):
            s = str(v).lower()
            if "red" in s:
                return 0
            if "green" in s:
                return 1
            if "yellow" in s:
                return 2
            return 3

        self.data["tl_label"] = self.data["traffic_light_state"].apply(_tl_to_idx).astype(int)

        # 4) 训练增强与标准化：使用 ReplayCompose 以便追踪是否执行了翻转。
        self.transform = A.ReplayCompose([
            # 缩放到模型输入尺寸
            A.Resize(66, 200),
            A.OneOf([
                # 随机调整亮度/对比度，模拟不同光照条件
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                # 轻微色调/饱和度变化，增加颜色多样性
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
                # 随机 gamma 变换，模拟不同摄像头响应特性
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            ], p=0.3),
            # 水平翻转，模拟左右转弯，注意后续标签调整
            A.HorizontalFlip(p=0.5),
            # 三通道归一化
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # 转换为张量
            ToTensorV2(),
        ])

        # 5) 根据 steering 分布构造加权采样器，缓解长尾不平衡。
        self.sampler = self._create_balanced_sampler()

    def _create_balanced_sampler(self):
        # 将 steering 分箱后按逆频率赋权，低频角度会被更高概率采样。
        steering_angles = np.array(self.data["steering_angle"].values, dtype=np.float32)
        hist, bins = np.histogram(steering_angles, bins=41, range=(-0.4, 0.4))
        bin_indices = np.digitize(steering_angles, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(hist) - 1)
        weights = 1.0 / (hist[bin_indices] + 1e-6)
        weights = weights / weights.sum() * len(weights)

        return torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1) 读取当前样本元数据并定位图像路径。
        row = self.data.iloc[idx]

        camera_pos = row["camera_position"]
        filename = row["frame_filename"]
        image_path = self.image_dirs[camera_pos] / filename

        # 2) 读图并转换为 RGB（与训练预处理一致）。
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3) 读取多任务标签与上一时刻速度。
        steer = float(row["steering_angle"])
        throttle = float(row["throttle"])
        brake = float(row["brake"])
        prev_speed = float(row["prev_speed_kmh"])
        # 读取新增标签：traffic light 类别索引与 is_stopped（0/1）
        tl_label = int(row.get("tl_label", 3))
        is_stopped = float(row.get("is_stopped", 0.0))

        # 4) 执行增强与归一化。
        transformed = self.transform(image=image)
        image = transformed["image"]

        # 水平翻转时只改变横向控制符号
        replay_data = transformed.get("replay", {})
        if replay_data:
            for transform_info in replay_data.get("transforms", []):
                if transform_info.get("__class_fullname__") == "HorizontalFlip" and transform_info.get("applied"):
                    steer = -steer
                    break

        # 5) 组装返回张量：图像、三任务目标、上一时刻速度、traffic-light 类别、是否停驶
        target = torch.tensor([steer, throttle, brake], dtype=torch.float32)
        prev_speed_tensor = torch.tensor(prev_speed, dtype=torch.float32)
        tl_label_tensor = torch.tensor(tl_label, dtype=torch.long)
        is_stopped_tensor = torch.tensor(is_stopped, dtype=torch.float32)
        return image, target, prev_speed_tensor, tl_label_tensor, is_stopped_tensor
