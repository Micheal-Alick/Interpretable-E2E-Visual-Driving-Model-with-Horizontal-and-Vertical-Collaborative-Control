"""
gradcam_visualization.py

封装实时 Grad-CAM 可视化流程：
- 接收当前用于推理的模型输入张量
- 调用 grad_cam_mask.py 生成可解释性 mask
- 将 mask 覆盖到输入图像并生成 OpenCV 可显示面板
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from grad_cam_mask import GradCAMGenerator


class GradCAMMaskVisualizer:
    """实时 Grad-CAM 可解释性遮罩可视化器。"""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        overlay_alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.overlay_alpha = float(np.clip(overlay_alpha, 0.0, 1.0))
        self.colormap = colormap
        self.mask_generator = GradCAMGenerator(model=model, device=self.device)

    def build_overlay_from_model_input(
        self,
        model_input_tensor: torch.Tensor,
        resized_rgb: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        根据当前推理输入直接生成 Grad-CAM 遮罩可视化。

        返回字段：
        - mask: [0,1] 单通道 mask
        - heatmap_bgr: 伪彩热力图
        - overlay_bgr: 叠加效果图
        """
        # 1) 先根据当前模型输入生成单通道 Grad-CAM mask。
        result = self.mask_generator.generate_mask_from_tensor(model_input_tensor)
        mask = result["mask"]
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
        """构造 side-by-side 可视化面板，便于实时观察输入与 Grad-CAM 覆盖效果。"""
        input_bgr = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
        # 左侧原图，右侧 Grad-CAM overlay，保持同尺度便于逐像素对比。
        panel = np.hstack([input_bgr, overlay_bgr])

        h, w = panel.shape[:2]
        title_bg_h = 28
        canvas = np.zeros((h + title_bg_h, w, 3), dtype=np.uint8)
        canvas[title_bg_h:, :, :] = panel
        canvas[:title_bg_h, :, :] = (25, 25, 25)

        cv2.putText(
            canvas,
            "Input (left) | Grad-CAM Mask Overlay (right)",
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
