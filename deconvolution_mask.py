"""
deconvolution_mask.py

使用论文中的逐层反卷积掩膜算法生成输入显著性 mask，
用于解释端到端转向模型在当前帧关注了哪些区域。

核心思路：
1. 对每个卷积层输出在通道维做平均，得到单通道激活图。
2. 从最顶层平均图开始，按对应卷积层参数做单位核反卷积上采样。
3. 与下一层（更低层）平均图逐层相乘，得到中间掩膜并继续向下传播。
4. 传播到输入尺寸后，与输入平均图相乘并归一化到 [0, 1]。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import NvidiaModel


class DeconvolutionMaskGenerator:
    """论文版逐层反卷积可解释性 mask 生成器。"""

    def __init__(
        self,
        model: nn.Module,   # 需要是单输出标量的回归模型，且包含卷积层
        device: Optional[torch.device] = None,
        input_size: Tuple[int, int] = (200, 66),    # 输入图片维度
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),   # 平均值（RGB 顺序）
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),    # 标准差（RGB 顺序）
    ) -> None:
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size  # (width, height)

        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.mean_t = torch.as_tensor(mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std_t = torch.as_tensor(std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

        self.model.to(self.device)
        self.model.eval()

    def preprocess_bgr_image(self, image_bgr: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """将 BGR 图像转换为模型输入张量，并返回 resize 后的 RGB 图像。"""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, self.input_size)

        image_normalized = image_resized.astype(np.float32) / 255.0
        image_normalized = (image_normalized - self.mean) / self.std

        image_tensor = torch.from_numpy(image_normalized).float().permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        return image_tensor, image_resized

    def _register_conv_forward_hooks(
        self,
    ) -> Tuple[List[torch.utils.hooks.RemovableHandle], List[nn.Conv2d], List[torch.Tensor]]:
        """回退路径：注册所有卷积层前向钩子并收集平均图。"""
        handles: List[torch.utils.hooks.RemovableHandle] = []
        conv_modules: List[nn.Conv2d] = []
        avg_maps: List[torch.Tensor] = []

        def conv_forward_hook(module, _input, output):
            if not isinstance(output, torch.Tensor) or output.ndim != 4:
                return
            conv_modules.append(module)
            # 论文步骤1：每层 feature map 在通道维取平均。
            avg_map = output.detach().mean(dim=1, keepdim=True)
            # 保留正激活证据，避免负值在逐层相乘中引入符号翻转。
            avg_maps.append(torch.clamp(avg_map, min=0.0))

        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                handles.append(module.register_forward_hook(conv_forward_hook))

        return handles, conv_modules, avg_maps

    @staticmethod
    def _extract_stage_transition_conv(stage_module: nn.Module) -> Optional[nn.Conv2d]:
        """提取 stage 中负责降采样的卷积层；若不存在则返回该 stage 首个卷积层。"""
        first_conv: Optional[nn.Conv2d] = None
        stride_conv: Optional[nn.Conv2d] = None

        for module in stage_module.modules():
            if isinstance(module, nn.Conv2d):
                if first_conv is None:
                    first_conv = module
                if module.stride[0] > 1 or module.stride[1] > 1:
                    stride_conv = module
                    break

        return stride_conv or first_conv

    def _register_stage_forward_hooks(
        self,
    ) -> Tuple[List[torch.utils.hooks.RemovableHandle], List[nn.Conv2d], List[torch.Tensor]]:
        """
        优先使用 ResNet 主干 stage 输出（layer1~layer4）作为论文中的层级激活图。
        若模型结构不匹配，则回退到全卷积层路径。
        """
        conv_layers = getattr(self.model, "conv_layers", None)
        if not isinstance(conv_layers, nn.Sequential) or len(conv_layers) < 8:
            return self._register_conv_forward_hooks()

        conv1 = conv_layers[0] if isinstance(conv_layers[0], nn.Conv2d) else None
        layer1 = conv_layers[4] if isinstance(conv_layers[4], nn.Module) else None
        layer2 = conv_layers[5] if isinstance(conv_layers[5], nn.Module) else None
        layer3 = conv_layers[6] if isinstance(conv_layers[6], nn.Module) else None
        layer4 = conv_layers[7] if isinstance(conv_layers[7], nn.Module) else None

        if conv1 is None or layer1 is None or layer2 is None or layer3 is None or layer4 is None:
            return self._register_conv_forward_hooks()

        t2 = self._extract_stage_transition_conv(layer2)
        t3 = self._extract_stage_transition_conv(layer3)
        t4 = self._extract_stage_transition_conv(layer4)
        if t2 is None or t3 is None or t4 is None:
            return self._register_conv_forward_hooks()

        handles: List[torch.utils.hooks.RemovableHandle] = []
        # 与 stage map 一一对应：layer1、layer2、layer3、layer4。
        # 其中第 0 个 conv1 用于最后一步传播到输入分辨率。
        conv_modules: List[nn.Conv2d] = [conv1, t2, t3, t4]
        avg_maps: List[torch.Tensor] = []

        def stage_forward_hook(module, _input, output):
            if not isinstance(output, torch.Tensor) or output.ndim != 4:
                return
            avg_map = output.detach().mean(dim=1, keepdim=True)
            avg_maps.append(torch.clamp(avg_map, min=0.0))

        for stage_module in [layer1, layer2, layer3, layer4]:
            handles.append(stage_module.register_forward_hook(stage_forward_hook))

        return handles, conv_modules, avg_maps

    @staticmethod
    def _renormalize_positive_map(mask: torch.Tensor) -> torch.Tensor:
        """层间传播后做重标定，抑制连乘数值塌缩。"""
        mask = torch.clamp(mask, min=0.0)
        max_val = mask.amax(dim=(-2, -1), keepdim=True)
        return mask / (max_val + 1e-8)

    @staticmethod
    def _compute_output_padding(
        in_size: int,
        target_size: int,
        kernel: int,
        stride: int,
        padding: int,
        dilation: int,
    ) -> Optional[int]:
        """根据目标尺寸计算 conv_transpose2d 的 output_padding。"""
        base = (in_size - 1) * stride - 2 * padding + dilation * (kernel - 1) + 1
        out_pad = int(target_size - base)
        if 0 <= out_pad < stride:
            return out_pad
        return None

    def _deconv_upsample_with_unit_kernel(
        self,
        mask: torch.Tensor,
        conv_layer: nn.Conv2d,
        target_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """
        使用与对应卷积层相同参数进行反卷积上采样。
        反卷积核权重固定为 1.0，bias 为 0.0（论文步骤2/4）。
        """
        kh, kw = conv_layer.kernel_size
        sh, sw = conv_layer.stride
        ph, pw = conv_layer.padding
        dh, dw = conv_layer.dilation

        in_h, in_w = int(mask.shape[-2]), int(mask.shape[-1])
        target_h, target_w = int(target_hw[0]), int(target_hw[1])

        out_pad_h = self._compute_output_padding(in_h, target_h, kh, sh, ph, dh)
        out_pad_w = self._compute_output_padding(in_w, target_w, kw, sw, pw, dw)

        if out_pad_h is not None and out_pad_w is not None:
            weight = torch.ones((1, 1, kh, kw), dtype=mask.dtype, device=mask.device)
            up = F.conv_transpose2d(
                mask,
                weight,
                bias=None,
                stride=(sh, sw),
                padding=(ph, pw),
                output_padding=(out_pad_h, out_pad_w),
                dilation=(dh, dw),
            )
        else:
            # 兼容复杂网络拓扑：若无法由 output_padding 精确对齐，则退化到插值对齐尺寸。
            up = F.interpolate(mask, size=(target_h, target_w), mode="bilinear", align_corners=False)

        if int(up.shape[-2]) != target_h or int(up.shape[-1]) != target_w:
            up = F.interpolate(up, size=(target_h, target_w), mode="bilinear", align_corners=False)

        return up

    @staticmethod
    def _normalize_mask(mask: np.ndarray) -> np.ndarray:
        """将最终 mask 做 min-max 归一化到 [0, 1]（论文步骤6）。"""
        mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.maximum(mask, 0.0)

        mask_min = float(np.min(mask))
        mask_max = float(np.max(mask))

        if mask_max - mask_min <= 1e-12:
            return np.zeros_like(mask, dtype=np.float32)

        norm = (mask - mask_min) / (mask_max - mask_min)
        norm = np.clip(norm, 0.0, 1.0)

        return norm.astype(np.float32)

    def generate_mask_from_tensor(
        self,
        model_input_tensor: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """
        从模型输入张量生成论文版逐层反卷积 mask。

        参数:
            model_input_tensor: 形状为 (1, C, H, W) 的归一化输入张量。

        返回:
            dict 包含:
            - mask: float32, [0, 1]
            - raw_saliency: 未归一化前的逐层传播掩膜
            - steering: 当前输出标量
        """
        input_tensor = model_input_tensor.detach().clone().to(self.device)
        hooks, conv_modules, avg_maps = self._register_stage_forward_hooks()

        try:
            with torch.no_grad():
                output = self.model(input_tensor)

            output_flat = output.reshape(-1)
            target = output_flat[0]
            steering_value = float(target.cpu().item())

            if not conv_modules or not avg_maps:
                raise RuntimeError("No Conv2d activations were captured for paper-style mask generation.")

            # 输入端采用反归一化后的亮度均值，避免标准化空间大面积被截断为 0。
            input_rgb = torch.clamp(input_tensor * self.std_t + self.mean_t, 0.0, 1.0)
            input_avg = input_rgb.mean(dim=1, keepdim=True)
            input_avg = self._renormalize_positive_map(input_avg)

            # 论文步骤2-5：从顶层开始逐层反卷积上采样，并与下层平均图相乘。
            current_mask = self._renormalize_positive_map(avg_maps[-1])
            for idx in range(len(avg_maps) - 1, 0, -1):
                lower_map = self._renormalize_positive_map(avg_maps[idx - 1])
                current_mask = self._deconv_upsample_with_unit_kernel(
                    current_mask,
                    conv_modules[idx],
                    target_hw=(int(lower_map.shape[-2]), int(lower_map.shape[-1])),
                )
                current_mask = current_mask * lower_map
                current_mask = self._renormalize_positive_map(current_mask)

            # 继续从最底卷积层传播到输入分辨率，并与输入平均图相乘。
            current_mask = self._deconv_upsample_with_unit_kernel(
                current_mask,
                conv_modules[0],
                target_hw=(int(input_avg.shape[-2]), int(input_avg.shape[-1])),
            )
            current_mask = current_mask * input_avg
            current_mask = self._renormalize_positive_map(current_mask)

            saliency = current_mask[0, 0].detach().cpu().numpy().astype(np.float32)

            mask = self._normalize_mask(saliency)

            return {
                "mask": mask,
                "raw_saliency": saliency.astype(np.float32),
                "steering": np.array([steering_value], dtype=np.float32),
            }
        finally:
            for handle in hooks:
                handle.remove()

    def generate_mask_from_bgr_image(self, image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        """从原始 BGR 图像直接生成 Deconv mask。"""
        input_tensor, resized_rgb = self.preprocess_bgr_image(image_bgr)
        result = self.generate_mask_from_tensor(input_tensor)
        result["resized_rgb"] = resized_rgb
        return result

    @staticmethod
    def create_heatmap(mask: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """把 [0,1] mask 映射为 BGR 热力图。"""
        mask_uint8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(mask_uint8, colormap)
        return heatmap

    @staticmethod
    def overlay_mask_on_rgb(
        image_rgb: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """将 mask 热力图覆盖到 RGB 图像，返回 BGR 结果，便于 OpenCV 显示。"""
        heatmap_bgr = DeconvolutionMaskGenerator.create_heatmap(mask, colormap=colormap)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0.0)
        return overlay


def _load_model_for_demo(model_path: Path, device: torch.device) -> nn.Module:
    model = NvidiaModel(pretrained=False, freeze_features=False)
    checkpoint = torch.load(str(model_path), map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deconvolution mask for one image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (.pt)")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--save_dir", type=str, default="analysis_output/deconv_demo", help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    image_path = Path(args.image_path)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model_for_demo(model_path, device)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    generator = DeconvolutionMaskGenerator(model=model, device=device)
    result = generator.generate_mask_from_bgr_image(image_bgr)

    resized_rgb = result["resized_rgb"]
    mask = result["mask"]
    overlay = generator.overlay_mask_on_rgb(resized_rgb, mask, alpha=float(args.alpha))

    mask_uint8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(save_dir / "mask.png"), mask_uint8)
    cv2.imwrite(str(save_dir / "overlay.png"), overlay)

    cv2.imshow("Deconvolution Mask Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
