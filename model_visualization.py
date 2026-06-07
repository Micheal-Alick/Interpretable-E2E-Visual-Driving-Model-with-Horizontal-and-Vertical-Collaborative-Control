"""
Model Visualization Tools for PhD Thesis
========================================

This module provides comprehensive visualization tools for PyTorch models,
specifically designed for academic documentation and PhD thesis requirements.

Features:
- Architecture diagrams with torchviz
- Model summary statistics
- Layer-wise parameter analysis
- Feature map visualizations
- Computational graph visualization
- Publication-ready plots
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from torchviz import make_dot
from torchsummary import summary
import torchinfo
from graphviz import Digraph
import io
import base64
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Set publication-ready style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelVisualizer:
    """
    Comprehensive model visualization toolkit for academic publications.
    """
    
    def __init__(self, model, input_shape=(3, 224, 224), device='cpu'):
        """
        Initialize the visualizer with a PyTorch model.
        
        Args:
            model: PyTorch model instance
            input_shape: Input tensor shape (C, H, W)
            device: Device to run model on
        """
        self.model = model.to(device)
        self.device = device
        self.input_shape = input_shape
        self.model.eval()
        
    def create_architecture_diagram(self, save_path=None, format='png', dpi=300):
        """
        Create a computational graph visualization of the model architecture.
        
        Args:
            save_path: Path to save the diagram
            format: Output format ('png', 'pdf', 'svg')
            dpi: Resolution for raster formats
        """
        # Create dummy input
        dummy_input = torch.randn(1, *self.input_shape).to(self.device)
        
        # Forward pass to create computational graph
        output = self.model(dummy_input)
        
        # Create visualization
        dot = make_dot(output, params=dict(self.model.named_parameters()))
        dot.attr(rankdir='TB')  # Top to Bottom layout
        dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')
        dot.attr('edge', fontname='Arial')
        
        if save_path:
            dot.render(save_path, format=format, cleanup=True)
            print(f"Architecture diagram saved to {save_path}.{format}")
        
        return dot
    
    def create_custom_architecture_diagram(self, save_path=None):
        """
        Create a custom, publication-ready architecture diagram.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Define model components based on the model type
        if hasattr(self.model, 'conv_layers') and hasattr(self.model, 'regressor'):
            # Draw backbone
            backbone_rect = plt.Rectangle((1, 6), 2, 2, 
                                        facecolor='lightblue', 
                                        edgecolor='black', linewidth=2)
            ax.add_patch(backbone_rect)
            ax.text(2, 7, 'Feature\nExtractor\n(EfficientNet/ResNet)', 
                   ha='center', va='center', fontsize=10, weight='bold')
            
            # Draw pooling
            pool_rect = plt.Rectangle((4, 6.5), 1.5, 1, 
                                    facecolor='lightgreen', 
                                    edgecolor='black', linewidth=2)
            ax.add_patch(pool_rect)
            ax.text(4.75, 7, 'Global\nAvgPool', ha='center', va='center', fontsize=9)
            
            # Draw regression head layers
            reg_layers = ['Flatten', 'FC(256)', 'FC(100)', 'FC(50)', 'FC(1)']
            y_positions = [5, 4, 3, 2, 1]
            
            for i, (layer, y_pos) in enumerate(zip(reg_layers, y_positions)):
                rect = plt.Rectangle((7, y_pos), 1.5, 0.8, 
                                   facecolor='orange', 
                                   edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                ax.text(7.75, y_pos + 0.4, layer, ha='center', va='center', fontsize=9)
                
                # Add arrows
                if i < len(reg_layers) - 1:
                    ax.arrow(7.75, y_pos - 0.1, 0, -0.2, 
                           head_width=0.1, head_length=0.05, 
                           fc='black', ec='black')
            
            # Add arrows between main components
            ax.arrow(3, 7, 0.8, 0, head_width=0.1, head_length=0.1, 
                   fc='black', ec='black', linewidth=2)
            ax.arrow(5.5, 7, 1.3, -1.5, head_width=0.1, head_length=0.1, 
                   fc='black', ec='black', linewidth=2)
            
            # Add input/output labels
            ax.text(0.5, 7, 'Input\nImage\n(224×224×3)', ha='center', va='center', 
                   fontsize=10, weight='bold', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            
            ax.text(9, 1.4, 'Steering\nAngle', ha='center', va='center', 
                   fontsize=10, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            
            # Add title
            ax.text(5, 9, 'Neural Network Architecture for Autonomous Driving', 
                   ha='center', va='center', fontsize=14, weight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Custom architecture diagram saved to {save_path}")
        
        plt.show()
        return fig
    
    def model_summary_table(self, save_path=None):
        """
        Create a detailed model summary table suitable for thesis.
        """
        # Get model info
        dummy_input = torch.randn(1, *self.input_shape).to(self.device)
        model_stats = torchinfo.summary(self.model, input_size=(1, *self.input_shape), 
                                      verbose=0, device=self.device)
        
        # Extract layer information
        layers_info = []
        total_params = 0
        trainable_params = 0
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                param_count = sum(p.numel() for p in module.parameters())
                trainable_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                layers_info.append({
                    'Layer Name': name if name else 'root',
                    'Layer Type': module.__class__.__name__,
                    'Parameters': f"{param_count:,}",
                    'Trainable': f"{trainable_count:,}",
                    'Shape': str(getattr(module, 'weight', 'N/A')).split('(')[1].split(',')[0] if hasattr(module, 'weight') else 'N/A'
                })
                
                total_params += param_count
                trainable_params += trainable_count
        
        df = pd.DataFrame(layers_info)
        
        # Create summary statistics
        summary_stats = {
            'Total Parameters': f"{total_params:,}",
            'Trainable Parameters': f"{trainable_params:,}",
            'Non-trainable Parameters': f"{total_params - trainable_params:,}",
            'Model Size (MB)': f"{(total_params * 4) / (1024**2):.2f}",
            'Input Shape': f"{self.input_shape}",
            'Output Shape': "1 (steering angle)"
        }
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Layer-wise parameter distribution
        layer_types = df['Layer Type'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(layer_types)))
        
        ax1.pie(layer_types.values, labels=layer_types.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax1.set_title('Layer Type Distribution', fontsize=14, weight='bold', pad=20)
        
        # Plot 2: Parameter count by layer type
        param_by_type = df.groupby('Layer Type')['Parameters'].apply(
            lambda x: sum(int(p.replace(',', '')) for p in x)
        ).sort_values(ascending=True)
        
        ax2.barh(range(len(param_by_type)), param_by_type.values, 
                color=plt.cm.viridis(np.linspace(0, 1, len(param_by_type))))
        ax2.set_yticks(range(len(param_by_type)))
        ax2.set_yticklabels(param_by_type.index)
        ax2.set_xlabel('Number of Parameters')
        ax2.set_title('Parameters by Layer Type', fontsize=14, weight='bold')
        
        # Add value labels on bars
        for i, v in enumerate(param_by_type.values):
            ax2.text(v + max(param_by_type.values) * 0.01, i, f'{v:,}', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model summary saved to {save_path}")
        
        plt.show()
        
        return df, summary_stats, fig
    
    def create_feature_maps_visualization(self, input_image, layer_names=None, save_path=None):
        """
        Visualize feature maps from specific layers.
        
        Args:
            input_image: Input tensor (1, C, H, W)
            layer_names: List of layer names to visualize
            save_path: Path to save visualization
        """
        if layer_names is None:
            # Get some default layers to visualize
            layer_names = []
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.ReLU)) and len(layer_names) < 4:
                    layer_names.append(name)
        
        # Register hooks to capture activations
        activations = {}
        hooks = []
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach().cpu()
            return hook
        
        for name, module in self.model.named_modules():
            if name in layer_names:
                hooks.append(module.register_forward_hook(get_activation(name)))
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_image.to(self.device))
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Create visualization
        n_layers = len(layer_names)
        fig, axes = plt.subplots(2, n_layers, figsize=(4*n_layers, 8))
        if n_layers == 1:
            axes = axes.reshape(2, 1)
        
        for i, layer_name in enumerate(layer_names):
            if layer_name in activations:
                feature_maps = activations[layer_name][0]  # Remove batch dimension
                
                # Show first feature map
                if len(feature_maps.shape) == 3:  # Conv layer
                    # Average across all channels
                    avg_feature_map = torch.mean(feature_maps, dim=0)
                    axes[0, i].imshow(avg_feature_map, cmap='viridis')
                    axes[0, i].set_title(f'{layer_name}\n(Avg of {feature_maps.shape[0]} channels)')
                    
                    # Show individual channels (first 6)
                    n_channels_to_show = min(6, feature_maps.shape[0])
                    channel_mosaic = torch.zeros((feature_maps.shape[1] * 2, 
                                               feature_maps.shape[2] * 3))
                    
                    for ch in range(n_channels_to_show):
                        row = ch // 3
                        col = ch % 3
                        start_row = row * feature_maps.shape[1]
                        end_row = start_row + feature_maps.shape[1]
                        start_col = col * feature_maps.shape[2]
                        end_col = start_col + feature_maps.shape[2]
                        channel_mosaic[start_row:end_row, start_col:end_col] = feature_maps[ch]
                    
                    axes[1, i].imshow(channel_mosaic, cmap='viridis')
                    axes[1, i].set_title(f'First {n_channels_to_show} channels')
                
                axes[0, i].axis('off')
                axes[1, i].axis('off')
        
        plt.suptitle('Feature Map Activations', fontsize=16, weight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature maps visualization saved to {save_path}")
        
        plt.show()
        return fig, activations
    
    def compare_models(self, other_model, model_names=None, save_path=None):
        """
        Compare two models side by side.
        
        Args:
            other_model: Second model to compare
            model_names: List of names for the models
            save_path: Path to save comparison
        """
        if model_names is None:
            model_names = ['Model 1', 'Model 2']
        
        models = [self.model, other_model]
        
        # Calculate model statistics
        stats = []
        for model in models:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_size = (total_params * 4) / (1024**2)  # MB
            
            stats.append({
                'Total Parameters': total_params,
                'Trainable Parameters': trainable_params,
                'Model Size (MB)': model_size,
                'Layers': len(list(model.modules()))
            })
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = list(stats[0].keys())
        x = np.arange(len(model_names))
        width = 0.35
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            values = [stat[metric] for stat in stats]
            
            bars = ax.bar(x, values, width, 
                         color=['skyblue', 'lightcoral'],
                         alpha=0.8, edgecolor='black')
            
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                if metric == 'Model Size (MB)':
                    label = f'{value:.2f}'
                else:
                    label = f'{value:,}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label, ha='center', va='bottom')
        
        plt.suptitle('Model Comparison', fontsize=16, weight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison saved to {save_path}")
        
        plt.show()
        return fig, stats
    
    def create_thesis_figure_package(self, save_dir='model_figures'):
        """
        Create a complete package of figures suitable for PhD thesis.
        
        Args:
            save_dir: Directory to save all figures
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("Creating comprehensive model visualization package...")
        
        # 1. Architecture diagram
        print("1. Creating architecture diagram...")
        self.create_architecture_diagram(
            save_path=os.path.join(save_dir, 'model_architecture'),
            format='pdf'
        )
        
        # 2. Custom architecture diagram
        print("2. Creating custom architecture diagram...")
        self.create_custom_architecture_diagram(
            save_path=os.path.join(save_dir, 'custom_architecture.pdf')
        )
        
        # 3. Model summary
        print("3. Creating model summary...")
        df, summary_stats, fig = self.model_summary_table(
            save_path=os.path.join(save_dir, 'model_summary.pdf')
        )
        
        # Save summary table as CSV
        df.to_csv(os.path.join(save_dir, 'model_layers_table.csv'), index=False)
        
        # Save summary stats as text
        with open(os.path.join(save_dir, 'model_statistics.txt'), 'w') as f:
            for key, value in summary_stats.items():
                f.write(f"{key}: {value}\n")
        
        # 4. Feature maps (if possible)
        print("4. Creating feature maps visualization...")
        try:
            dummy_input = torch.randn(1, *self.input_shape)
            self.create_feature_maps_visualization(
                dummy_input,
                save_path=os.path.join(save_dir, 'feature_maps.pdf')
            )
        except Exception as e:
            print(f"Feature maps visualization skipped: {e}")
        
        print(f"\nAll visualizations saved to '{save_dir}' directory!")
        print("Files created:")
        for file in os.listdir(save_dir):
            print(f"  - {file}")
        
        return save_dir


def demonstrate_model_visualization():
    """
    Demonstration function showing how to use the visualization tools.
    """
    from model import NvidiaModel, NvidiaModelTransferLearning
    
    print("=== Model Visualization Demo ===\n")
    
    # Create models
    print("Loading models...")
    model1 = NvidiaModel(pretrained=False)
    model2 = NvidiaModelTransferLearning(pretrained=False)
    
    # Create visualizers
    print("Creating visualizers...")
    vis1 = ModelVisualizer(model1, input_shape=(3, 224, 224))
    vis2 = ModelVisualizer(model2, input_shape=(3, 224, 224))
    
    # Create comprehensive visualization package
    print("Creating thesis figure package for ResNet model...")
    vis1.create_thesis_figure_package('resnet_model_figures')
    
    print("\nCreating thesis figure package for EfficientNet model...")
    vis2.create_thesis_figure_package('efficientnet_model_figures')
    
    # Compare models
    print("\nComparing models...")
    vis1.compare_models(model2, 
                       model_names=['ResNet-50 Based', 'EfficientNet-B0 Based'],
                       save_path='model_comparison.pdf')
    
    print("\n=== Visualization Complete! ===")
    print("Check the generated directories for all visualization files.")


if __name__ == "__main__":
    demonstrate_model_visualization()
