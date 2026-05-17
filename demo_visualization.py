"""
Quick Demo: Model Visualization for PhD Thesis
==============================================

This script demonstrates how to quickly generate publication-ready
visualizations of your PyTorch models for your PhD thesis.

Usage:
    python demo_visualization.py
"""

import os
import sys
import torch
import warnings
warnings.filterwarnings('ignore')

# Import your models
try:
    from model import NvidiaModel, NvidiaModelTransferLearning
    from model_visualization import ModelVisualizer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure model.py and model_visualization.py are in the same directory.")
    sys.exit(1)

def main():
    """Main demonstration function."""
    
    print("=" * 60)
    print("MODEL VISUALIZATION FOR PHD THESIS")
    print("=" * 60)
    
    # Create output directory
    output_dir = "thesis_figures"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load models
    print("\n1. Loading models...")
    try:
        resnet_model = NvidiaModel(pretrained=False)
        efficientnet_model = NvidiaModelTransferLearning(pretrained=False)
        print("✓ Models loaded successfully!")
        
        # Print basic model info
        resnet_params = sum(p.numel() for p in resnet_model.parameters())
        efficientnet_params = sum(p.numel() for p in efficientnet_model.parameters())
        
        print(f"  - ResNet model: {resnet_params:,} parameters")
        print(f"  - EfficientNet model: {efficientnet_params:,} parameters")
        
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return
    
    # Create visualizers
    print("\n2. Creating visualizers...")
    try:
        resnet_viz = ModelVisualizer(resnet_model, input_shape=(3, 224, 224))
        efficientnet_viz = ModelVisualizer(efficientnet_model, input_shape=(3, 224, 224))
        print("✓ Visualizers created successfully!")
    except Exception as e:
        print(f"✗ Error creating visualizers: {e}")
        return
    
    # Generate thesis figure packages
    print("\n3. Generating comprehensive figure packages...")
    
    # ResNet figures
    print("   a) Creating ResNet model figures...")
    try:
        resnet_dir = resnet_viz.create_thesis_figure_package(
            os.path.join(output_dir, "resnet_model")
        )
        print("   ✓ ResNet figures generated!")
    except Exception as e:
        print(f"   ✗ Error generating ResNet figures: {e}")
    
    # EfficientNet figures
    print("   b) Creating EfficientNet model figures...")
    try:
        efficientnet_dir = efficientnet_viz.create_thesis_figure_package(
            os.path.join(output_dir, "efficientnet_model")
        )
        print("   ✓ EfficientNet figures generated!")
    except Exception as e:
        print(f"   ✗ Error generating EfficientNet figures: {e}")
    
    # Model comparison
    print("\n4. Creating model comparison...")
    try:
        comparison_fig, comparison_stats = resnet_viz.compare_models(
            efficientnet_model,
            model_names=['ResNet-50 Based', 'EfficientNet-B0 Based'],
            save_path=os.path.join(output_dir, 'model_comparison.pdf')
        )
        print("✓ Model comparison generated!")
        
        # Print comparison summary
        print("\n   Model Comparison Summary:")
        print("   " + "-" * 40)
        for i, (name, stats) in enumerate(zip(['ResNet-50 Based', 'EfficientNet-B0 Based'], comparison_stats)):
            print(f"   {name}:")
            print(f"     - Parameters: {stats['Total Parameters']:,}")
            print(f"     - Size: {stats['Model Size (MB)']:.2f} MB")
            print(f"     - Layers: {stats['Layers']}")
        
    except Exception as e:
        print(f"✗ Error creating model comparison: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE!")
    print("=" * 60)
    print(f"\nAll figures have been saved to: {os.path.abspath(output_dir)}")
    print("\nGenerated files include:")
    print("📊 Architecture diagrams (PDF/PNG)")
    print("📈 Model summary statistics")
    print("🔍 Parameter analysis plots")
    print("⚖️  Model comparison charts")
    print("📋 CSV tables with layer details")
    print("📝 Text files with model statistics")
    
    print("\n💡 Tips for your PhD thesis:")
    print("   - Use PDF files for high-quality printing")
    print("   - Include model comparison in methodology section")
    print("   - Use architecture diagrams in model description")
    print("   - Cite parameter counts and model sizes")
    
    print(f"\n📁 Check the '{output_dir}' directory for all files!")

if __name__ == "__main__":
    main()
