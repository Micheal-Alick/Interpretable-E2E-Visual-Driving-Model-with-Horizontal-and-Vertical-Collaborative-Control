"""
grad_cam_mask.py

使用 Grad-CAM 算法生成输入显著性 mask，
用于解释端到端转向模型在当前帧关注了哪些区域。

核心思路 (4 步实现):
1. 前向传播：获取模型最后卷积层的特征图
2. 梯度计算：计算目标类别分数对该特征图的梯度
3. 通道权重：对梯度进行全局平均池化，得到各通道的类别重要性权重
4. 热力图生成：将权重与特征图加权求和并通过 ReLU 激活，生成类别判别性的定位热图
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAMGenerator:
    """Grad-CAM 可解释性 mask 生成器。"""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        input_size: Tuple[int, int] = (200, 66),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        """
        初始化 Grad-CAM 生成器。

        参数:
            model: 神经网络模型（需要是单输出标量的回归模型）
            device: 计算设备 (cuda/cpu)
            input_size: 输入图片维度 (width, height)
            mean: 图像标准化的均值 (RGB 顺序)
            std: 图像标准化的标准差 (RGB 顺序)
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size

        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.mean_t = torch.as_tensor(mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std_t = torch.as_tensor(std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

        self.model.to(self.device)
        self.model.eval()

        # 可配置的后处理/层选择参数（便于调参以获得更精细热区）
        # - min_spatial: 选择目标卷积层时要求的最小空间维度（height/width 的最小值）
        # - postprocess_*: 一组用于锐化/聚焦热力图的超参数
        """这里是配置grad-cam生成和后处理的参数，便于调试和优化热力图质量："""
        self.min_spatial = 8    # 这是选择卷积层时要求的最小空间维度，避免选到过早的层导致热力图过于模糊
        self.postprocess_enabled = True # 是否启用后处理步骤（如高斯平滑、百分位裁剪、gamma校正等）来增强热力图的可视化效果
        self.post_percentile = 99.9 # 在后处理步骤中使用的百分位数，用于裁剪热力图的高值区域，保留上位响应以增强对关键区域的关注
        self.post_gamma = 1.6   # gamma 校正指数，用于放大热力图中的高响应区域并压缩低响应区域，使得重要区域更突出
        self.post_gaussian_sigma = 1  # 用于高斯平滑的 sigma 值，适度平滑可以减少热力图中的噪点并增强连续区域的可视化，但过度平滑可能会模糊细节
        self.post_threshold = 0.9  # 最后归一化后的阈值，低于该值的热力图区域将被置零，以进一步突出重要区域并减少视觉干扰

        # 查找最后的卷积层，用于获取特征图
        self.target_conv_layer = self._find_last_conv_layer()

    def _find_last_conv_layer(self) -> Optional[nn.Module]:
        """
        更鲁棒地查找用于 Grad-CAM 的卷积层：

        - 对模型中的所有 Conv2d 注册前向 hook，使用一个 dummy 输入做一次前向，记录每个 Conv 的输出尺寸
        - 从后向前选择第一个具有合理空间分辨率（min(H,W) >= 4）的 Conv2d 作为目标
        - 若未找到满足条件的层，则回退到最后一个 Conv2d

        这样可以兼容 ResNet 风格（layer3/layer4）以及自定义主干结构。
        """
        # 收集模型中所有 Conv2d 模块
        conv_modules = [m for m in self.model.modules() if isinstance(m, nn.Conv2d)]
        if not conv_modules:
            print("[Grad-CAM] WARNING: No Conv2d layer found in model!")
            return None

        # 注册前向 hook，记录每个 conv 的输出 shape（使用 detach 的 cpu 副本以节省显存）
        activations = []
        hooks = []

        def make_hook(module):
            def _hook(mod, inp, out):
                try:
                    activations.append((mod, out.detach().cpu().shape))
                except Exception:
                    activations.append((mod, None))
            return _hook

        for cm in conv_modules:
            try:
                hooks.append(cm.register_forward_hook(make_hook(cm)))
            except Exception:
                pass

        # 准备 dummy 输入（注意 input_size 存储为 (width, height)）
        dummy = torch.zeros((1, 3, self.input_size[1], self.input_size[0]), device=self.device)
        try:
            with torch.no_grad():
                # 兼容 adapter 可能需要 prev_speed_kmh 的情况
                try:
                    _ = self.model(dummy)
                except TypeError:
                    try:
                        _ = self.model(dummy, prev_speed_kmh=torch.zeros((1,), device=self.device))
                    except Exception:
                        _ = self.model(dummy)
        except Exception as e:
            print(f"[Grad-CAM] Dummy forward failed: {e}")
        finally:
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass

        # 从后向前选择第一个满足空间分辨率的 conv
        min_spatial = int(getattr(self, "min_spatial", 4))
        for mod, shape in reversed(activations):
            if shape is None:
                continue
            if len(shape) >= 4:
                _, c, h, w = shape
                if min(h, w) >= min_spatial:
                    print(f"[Grad-CAM] Selected conv layer {mod} with activation shape {shape}")
                    return mod

        # 回退到最后一个 Conv2d
        last_conv = conv_modules[-1]
        print(f"[Grad-CAM] Fallback selected last conv layer: {last_conv}")
        return last_conv

    def preprocess_bgr_image(self, image_bgr: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """
        将 BGR 图像转换为模型输入张量。

        参数:
            image_bgr: OpenCV 格式的 BGR 图像

        返回:
            (input_tensor, resized_rgb): 模型输入张量和 resize 后的 RGB 图像
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, self.input_size)

        image_normalized = image_resized.astype(np.float32) / 255.0
        image_normalized = (image_normalized - self.mean) / self.std

        image_tensor = torch.from_numpy(image_normalized).float().permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        return image_tensor, image_resized

    def generate_mask_from_tensor(
        self,
        model_input_tensor: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """
        从模型输入张量生成 Grad-CAM mask。

        参数:
            model_input_tensor: 形状为 (1, C, H, W) 的归一化输入张量

        返回:
            dict 包含:
            - mask: float32, [0, 1] 的 Grad-CAM 热力图
            - raw_saliency: 未归一化前的 Grad-CAM 值
            - steering: 当前输出标量
        """
        input_tensor = model_input_tensor.detach().clone().to(self.device)
        input_tensor.requires_grad = True

        if self.target_conv_layer is None:
            raise RuntimeError("Could not find a Conv2d layer in the model for Grad-CAM.")

        # 步骤1: 注册前向钩子捕获目标卷积层的输出特征图
        feature_maps = []

        def hook_features(module, input, output):
            feature_maps.append(output)

        hook_features_handle = self.target_conv_layer.register_forward_hook(hook_features)

        try:
            # 执行前向传播，获取模型输出和特征图
            self.model.zero_grad()
            with torch.enable_grad():
                use_speed_input = getattr(self.model, "use_speed_input", False)
                if use_speed_input:
                    speed_tensor = torch.zeros((input_tensor.shape[0],), dtype=torch.float32, device=self.device)
                    output = self.model(input_tensor, prev_speed_kmh=speed_tensor)
                else:
                    output = self.model(input_tensor)

                # 验证特征图是否成功捕获
                if not feature_maps:
                    raise RuntimeError("Failed to capture feature maps from target conv layer.")

                # 获取第一个特征图用于后续计算
                features = feature_maps[0]  # (1, C, H, W)

                # 更可靠地捕获特征图处的梯度：对特征张量本身注册 tensor hook
                grad_features = []

                def _save_grad(grad):
                    grad_features.append(grad.detach())

                features.register_hook(_save_grad)

                # 获取输出值（优先使用 pre-activation steer_logit，如果可用）
                if isinstance(output, dict):
                    target = output.get("steer_logit", output.get("steer", output.get("steering")))
                    if target is None:
                        raise RuntimeError("Model output dict doesn't contain 'steer'/'steering' or 'steer_logit' key")
                    if target.dim() > 0:
                        target = target.squeeze()[0] if target.numel() > 1 else target.squeeze()
                else:
                    target = output.reshape(-1)[0]

                steering_value = float(target.cpu().item())

                # 计算梯度：反向传播
                self.model.zero_grad()
                target.backward()

                # 验证梯度是否成功计算
                if not grad_features:
                    raise RuntimeError("Failed to compute gradients for Grad-CAM (no grad on features captured).")

                gradients = grad_features[0]  # (1, C, H, W)

                # 步骤3: 对梯度在空间维度进行全局平均池化，得到每个通道的重要性权重
                weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

                # 步骤4: 使用权重加权特征图并通过 ReLU 生成热力图
                grad_cam = (weights * features).sum(dim=1, keepdim=True)  # (1, 1, H, W)
                grad_cam = F.relu(grad_cam)

                saliency = grad_cam[0, 0].detach().cpu().numpy().astype(np.float32)
                mask = self._normalize_mask(saliency)

                return {
                    "mask": mask,
                    "raw_saliency": saliency,
                    "steering": np.array([steering_value], dtype=np.float32),
                }

        except Exception as e:
            raise RuntimeError(f"Grad-CAM generation failed: {str(e)}")
        finally:
            # 清理前向钩子
            hook_features_handle.remove()

    def generate_mask_from_bgr_image(self, image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        """
        从原始 BGR 图像直接生成 Grad-CAM mask。

        参数:
            image_bgr: OpenCV 格式的 BGR 图像

        返回:
            包含 mask, raw_saliency, steering, resized_rgb 的字典
        """
        input_tensor, resized_rgb = self.preprocess_bgr_image(image_bgr)
        result = self.generate_mask_from_tensor(input_tensor)
        result["resized_rgb"] = resized_rgb
        return result

    def _normalize_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        将原始 Grad-CAM 热力图归一化到 [0, 1]。
        
        Grad-CAM 热力图通过累积梯度和特征图生成，可能包含以下问题：
        - NaN 值：由于数值计算误差
        - 无穷大：由于除以零或极端值
        - 负值：虽然 ReLU 应该已经移除，但浮点计算可能产生
        
        本函数通过以下步骤处理：
        1. 清理异常值（NaN、Inf）
        2. 将所有负值剪裁为 0
        3. Min-Max 归一化到 [0, 1] 范围
        4. 二次剪裁确保值域正确
        
        参数:
            mask: 原始 Grad-CAM 值 (可能包含 nan 或 inf)

        返回:
            归一化后的 mask，值域 [0, 1]
        """
        # 清理异常值：将 NaN 替换为 0.0，正无穷和负无穷也替换为 0.0
        mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        # 确保所有值非负
        mask = np.maximum(mask, 0.0)

        # 基本的 min-max 检查
        mask_min = float(np.min(mask))
        mask_max = float(np.max(mask))
        if mask_max - mask_min <= 1e-12:
            return np.zeros_like(mask, dtype=np.float32)

        # 若未启用后处理，使用简单的 min-max 归一化
        if not getattr(self, "postprocess_enabled", False):
            norm = (mask - mask_min) / (mask_max - mask_min)
            return np.clip(norm, 0.0, 1.0).astype(np.float32)

        # 获取实例化时设置的后处理参数
        post_percentile = float(getattr(self, "post_percentile", 98.0))
        post_gamma = float(getattr(self, "post_gamma", 1.6))
        post_sigma = float(getattr(self, "post_gaussian_sigma", 1.0))
        post_threshold = float(getattr(self, "post_threshold", 0.04))

        # Gaussian smoothing
        try:
            if post_sigma is not None and post_sigma > 0.0:
                mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=float(post_sigma))
        except Exception:
            pass

        # 使用百分位裁剪来削弱扩散的高值区间，只保留上位响应
        try:
            pval = float(np.percentile(mask, float(post_percentile))) if post_percentile < 100.0 else float(np.max(mask))
        except Exception:
            pval = float(np.max(mask))

        if pval <= 1e-12:
            return np.zeros_like(mask, dtype=np.float32)

        # 剪裁并归一化到 [0,1]
        mask = np.clip(mask, 0.0, pval) / (pval + 1e-12)

        # gamma 校正以放大高响应并压缩低响应
        if post_gamma is not None and post_gamma != 1.0:
            mask = np.power(mask, float(post_gamma))

        # 最后归一化与阈值过滤
        mask = mask / (np.max(mask) + 1e-12)
        mask[mask < float(post_threshold)] = 0.0
        mask = np.clip(mask, 0.0, 1.0)

        return mask.astype(np.float32)

    @staticmethod
    def create_heatmap(mask: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        将 [0,1] 范围的灰度热力图映射为彩色热力图。
        
        热力图可视化的步骤：
        1. 将 [0, 1] 的浮点值转换为 [0, 255] 的整型像素值
        2. 应用 OpenCV 颜色映射将灰度值映射到彩色图像
        
        常见的颜色映射方案：
        - cv2.COLORMAP_JET: 蓝→青→绿→黄→红（推荐）
        - cv2.COLORMAP_VIRIDIS: 深紫→蓝→青→绿→黄
        - cv2.COLORMAP_HOT: 黑→红→黄→白

        参数:
            mask: [0, 1] 范围的单通道 mask
            colormap: OpenCV 颜色映射

        返回:
            BGR 格式的彩色热力图
        """
        # 将 [0, 1] 的浮点值缩放到 [0, 255] 并转换为 uint8
        mask_uint8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
        # 应用颜色映射
        heatmap = cv2.applyColorMap(mask_uint8, colormap)
        return heatmap

    @staticmethod
    def overlay_mask_on_rgb(
        image_rgb: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        将 Grad-CAM 热力图与原始图像合成，便于可视化模型的注意力区域。
        
        合成方法：
        - 原始图像和热力图按权重加权平均
        - alpha 参数控制热力图的透明度
        - 较大的 alpha 使热力图更显眼；较小的 alpha 更突出原始图像细节
        
        公式: Output = (1 - alpha) * Original + alpha * Heatmap

        参数:
            image_rgb: RGB 格式的原始图像
            mask: [0, 1] 范围的 Grad-CAM mask
            alpha: 热力图的透明度 (0-1)，默认 0.45
            colormap: OpenCV 颜色映射

        返回:
            BGR 格式的叠加图像
        """
        # 从mask生成彩色热力图
        heatmap_bgr = GradCAMGenerator.create_heatmap(mask, colormap=colormap)
        
        # 将RGB原始图像转换为BGR格式
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # 如果热力图尺寸与原图不匹配，调整热力图大小
        if heatmap_bgr.shape[:2] != image_bgr.shape[:2]:
            heatmap_bgr = cv2.resize(
                heatmap_bgr,
                (image_bgr.shape[1], image_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        # 加权融合：(1-alpha)*原图 + alpha*热力图
        overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0.0)
        return overlay
