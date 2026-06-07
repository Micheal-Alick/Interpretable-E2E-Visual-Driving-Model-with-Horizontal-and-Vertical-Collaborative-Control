"""
mask_visualization.py

封装实时 mask 可视化流程：
- 接收当前用于推理的模型输入张量
- 调用 deconvolution_mask.py 生成可解释性 mask
- 将 mask 覆盖到输入图像并生成 OpenCV 可显示面板
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from deconvolution_mask import DeconvolutionMaskGenerator
from model import NvidiaModel


class MaskVisualizer:
    """实时可解释性遮罩可视化器。"""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        overlay_alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
        roi_keep_ratio: float = 0.80,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.overlay_alpha = float(np.clip(overlay_alpha, 0.0, 1.0))
        self.colormap = colormap
        self.roi_keep_ratio = float(np.clip(roi_keep_ratio, 0.0, 1.0))
        self.mask_generator = DeconvolutionMaskGenerator(model=model, device=self.device)

    def build_overlay_from_model_input(
        self,
        model_input_tensor: torch.Tensor,
        resized_rgb: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        根据当前推理输入直接生成遮罩可视化。

        返回字段：
        - mask: [0,1] 单通道 mask
        - heatmap_bgr: 伪彩热力图
        - overlay_bgr: 叠加效果图
        """
        # 1) 先根据当前模型输入生成单通道 mask。
        result = self.mask_generator.generate_mask_from_tensor(model_input_tensor)
        mask = result["mask"]

        # ---- ROI 区域裁剪：仅保留图像中下方区域供注意力遮罩 ----
        # 将 mask 顶部（天空等无关区域）置零，只保留自底向上的 keep_ratio 比例区域。
        if self.roi_keep_ratio < 1.0:
            mask_h = mask.shape[0]
            cutoff_row = int(mask_h * (1.0 - self.roi_keep_ratio))
            if cutoff_row > 0:
                mask[:cutoff_row, :] = 0.0

        # 2) 再派生热力图与 overlay，便于调试和展示复用。
        heatmap_bgr = self.mask_generator.create_heatmap(mask, colormap=self.colormap)
        overlay_bgr = self.mask_generator.overlay_mask_on_rgb(
            resized_rgb,
            mask,
            alpha=self.overlay_alpha,
            colormap=self.colormap,
        )

        return {
            "mask": mask,
            "heatmap_bgr": heatmap_bgr,
            "overlay_bgr": overlay_bgr,
        }

    def create_visualization_panel(
        self,
        resized_rgb: np.ndarray,
        overlay_bgr: np.ndarray,
        steering_value: Optional[float] = None,
    ) -> np.ndarray:
        """构造 side-by-side 可视化面板，便于实时观察输入与遮罩覆盖效果。"""
        input_bgr = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
        # 左侧原图，右侧 mask overlay，保持同尺度便于逐像素对比。
        panel = np.hstack([input_bgr, overlay_bgr])

        h, w = panel.shape[:2]
        title_bg_h = 28
        canvas = np.zeros((h + title_bg_h, w, 3), dtype=np.uint8)
        canvas[title_bg_h:, :, :] = panel
        canvas[:title_bg_h, :, :] = (25, 25, 25)

        cv2.putText(
            canvas,
            "Input (left) | Deconvolution Mask Overlay (right)",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

        if steering_value is not None:
            cv2.putText(
                canvas,
                f"Steering: {steering_value:+.3f}",
                (w - 170, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return canvas


def _load_model_for_demo(model_path: Path, device: torch.device) -> nn.Module:
    # Demo 入口仅负责加载权重并切到 eval，不改变训练图结构。
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
    parser = argparse.ArgumentParser(description="Visualize deconvolution mask on one image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (.pt)")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask overlay alpha")
    args = parser.parse_args()

    # 复用主流程同款设备选择逻辑，保证本地单图调试与在线推理一致。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model_for_demo(Path(args.model_path), device)
    visualizer = MaskVisualizer(
        model=model,
        device=device,
        overlay_alpha=args.alpha,
    )

    image_bgr = cv2.imread(args.image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    # 为演示构造模型输入（resize + normalize），与推理路径保持一致。
    input_tensor, resized_rgb = visualizer.mask_generator.preprocess_bgr_image(image_bgr)

    with torch.no_grad():
        steering_pred = float(model(input_tensor).reshape(-1)[0].item())

    output = visualizer.build_overlay_from_model_input(input_tensor, resized_rgb)
    panel = visualizer.create_visualization_panel(
        resized_rgb=resized_rgb,
        overlay_bgr=output["overlay_bgr"],
        steering_value=steering_pred,
    )

    cv2.imshow("Mask Visualization", panel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
