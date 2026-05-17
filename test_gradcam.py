#!/usr/bin/env python3
"""
test_gradcam.py

诊断脚本：验证 Grad-CAM 实现是否正常工作
用于调试热力图生成问题

使用方法:
    python test_gradcam.py --model_path <path_to_model>
"""

import argparse
import torch
import cv2
import numpy as np
from pathlib import Path

from model_multitask import MultiTaskNvidiaModel
from grad_cam_mask import GradCAMGenerator


def test_gradcam_basic():
    """测试 Grad-CAM 基础功能"""
    print("=" * 70)
    print("Grad-CAM 诊断测试")
    print("=" * 70)
    
    # 1. 检查 PyTorch 和 GPU
    print("\n[1] 环境检查:")
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU 设备: {torch.cuda.get_device_name(0)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  使用设备: {device}")
    
    return device


def test_model_loading(model_path, device):
    """测试模型加载"""
    print("\n[2] 模型加载:")
    try:
        model = MultiTaskNvidiaModel(pretrained=False, freeze_features=False)
        checkpoint = torch.load(str(model_path), map_location=device)
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        print(f"  [OK] 模型成功加载: {model_path}")
        return model
    except Exception as e:
        print(f"  [FAIL] 模型加载失败: {e}")
        return None


def test_gradcam_initialization(model, device):
    """测试 Grad-CAM 初始化"""
    print("\n[3] Grad-CAM 初始化:")
    try:
        generator = GradCAMGenerator(model=model, device=device)
        
        if generator.target_conv_layer is None:
            print(f"  [WARN] 未找到卷积层!")
            return None
        
        print(f"  [OK] Grad-CAM 生成器初始化成功")
        print(f"  [OK] 找到目标卷积层")
        return generator
    except Exception as e:
        print(f"  [FAIL] Grad-CAM 初始化失败: {e}")
        return None


def test_image_preprocessing(generator):
    """测试图像预处理"""
    print("\n[4] 图像预处理:")
    try:
        # 创建一个随机测试图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        input_tensor, resized_rgb = generator.preprocess_bgr_image(test_image)
        
        print(f"  [OK] 图像预处理成功")
        print(f"    原始图像形状: {test_image.shape}")
        print(f"    缩放后形状: {resized_rgb.shape}")
        print(f"    输入张量形状: {input_tensor.shape}")
        print(f"    张量设备: {input_tensor.device}")
        print(f"    张量 requires_grad: {input_tensor.requires_grad}")
        
        return input_tensor, resized_rgb
    except Exception as e:
        print(f"  [FAIL] 图像预处理失败: {e}")
        return None, None


def test_gradcam_generation(generator, input_tensor, resized_rgb):
    """测试 Grad-CAM 生成"""
    print("\n[5] Grad-CAM 热力图生成:")
    try:
        result = generator.generate_mask_from_tensor(input_tensor)
        
        mask = result.get("mask")
        steering = result.get("steering")
        
        print(f"  [OK] Grad-CAM 生成成功")
        print(f"    Mask 形状: {mask.shape}")
        print(f"    Mask 值域: [{mask.min():.4f}, {mask.max():.4f}]")
        print(f"    Steering 值: {steering[0]:.4f}")
        
        return result
    except Exception as e:
        print(f"  [FAIL] Grad-CAM 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_heatmap_visualization(generator, result, resized_rgb):
    """测试热力图可视化"""
    print("\n[6] 热力图可视化:")
    try:
        mask = result.get("mask")
        
        # 创建热力图
        heatmap = generator.create_heatmap(mask)
        print(f"  [OK] 热力图生成成功")
        print(f"    热力图形状: {heatmap.shape}")
        
        # 创建叠加图
        overlay = generator.overlay_mask_on_rgb(resized_rgb, mask, alpha=0.45)
        print(f"  [OK] 热力图叠加成功")
        print(f"    叠加图形状: {overlay.shape}")
        
        return heatmap, overlay
    except Exception as e:
        print(f"  [FAIL] 热力图可视化失败: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM 诊断测试")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"[ERROR] 模型文件不存在: {model_path}")
        return
    
    # 执行诊断测试
    device = test_gradcam_basic()
    
    model = test_model_loading(model_path, device)
    if model is None:
        return
    
    generator = test_gradcam_initialization(model, device)
    if generator is None:
        return
    
    input_tensor, resized_rgb = test_image_preprocessing(generator)
    if input_tensor is None:
        return
    
    result = test_gradcam_generation(generator, input_tensor, resized_rgb)
    if result is None:
        return
    
    heatmap, overlay = test_heatmap_visualization(generator, result, resized_rgb)
    
    # 总结
    print("\n" + "=" * 70)
    print("诊断测试完成")
    print("=" * 70)
    
    if heatmap is not None and overlay is not None:
        print("\n[SUCCESS] 所有测试通过!")
        print("\n您可以安全地运行 predict_with_grad_cam.py")
    else:
        print("\n[FAILED] 部分测试失败，请检查上述错误信息")


if __name__ == "__main__":
    main()
