"""
Simplified Model Visualization for PhD Thesis
============================================

This script creates essential model visualizations that work without external dependencies
and are perfect for inclusion in your PhD thesis.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
import os
from model import NvidiaModel, NvidiaModelTransferLearning

# Set publication-ready style
plt.style.use('default')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['font.family'] = 'serif'

def analyze_model_layers(model, model_name):
    """Analyze model layers and return detailed statistics."""
    layer_info = []
    total_params = 0
    trainable_params = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            param_count = sum(p.numel() for p in module.parameters())
            trainable_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            layer_info.append({
                'Layer Name': name if name else f"{model_name}_root",
                'Layer Type': module.__class__.__name__,
                'Parameters': param_count,
                'Trainable': trainable_count,
                'Output Shape': getattr(module, 'out_features', 'N/A') if hasattr(module, 'out_features') else 'N/A'
            })
            
            total_params += param_count
            trainable_params += trainable_count
    
    model_stats = {
        'Total Parameters': total_params,
        'Trainable Parameters': trainable_params,
        'Non-trainable Parameters': total_params - trainable_params,
        'Model Size (MB)': (total_params * 4) / (1024**2),
        'Estimated Memory (MB)': (total_params * 4 * 4) / (1024**2)  # Forward + backward + optimizer
    }
    
    return layer_info, model_stats

def create_architecture_diagram(model, model_name, save_path=None):
    """Create a custom architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors
    colors = {
        'input': '#FFE6E6',
        'conv': '#E6F3FF', 
        'pool': '#E6FFE6',
        'fc': '#FFF0E6',
        'output': '#F0E6FF'
    }
    
    # Input
    input_rect = patches.Rectangle((0.5, 9), 2, 1.5, 
                                 facecolor=colors['input'], 
                                 edgecolor='black', linewidth=2)
    ax.add_patch(input_rect)
    ax.text(1.5, 9.75, 'Input\n(224×224×3)', ha='center', va='center', 
           fontsize=11, weight='bold')
    
    if 'efficientnet' in model_name.lower():
        # EfficientNet architecture
        components = [
            ('EfficientNet-B0\nBackbone', 3.5, 9, 2.5, 1.5, colors['conv']),
            ('Global\nAvgPool', 7, 9, 1.5, 1.5, colors['pool']),
            ('Flatten', 3, 6.5, 1.5, 1, colors['fc']),
            ('FC(256)\n+ BatchNorm\n+ ReLU', 5, 6.5, 2, 1, colors['fc']),
            ('FC(100)\n+ BatchNorm\n+ ReLU', 3, 4.5, 2, 1, colors['fc']),
            ('FC(50)\n+ ReLU', 5.5, 4.5, 1.5, 1, colors['fc']),
            ('FC(1)', 3.5, 2.5, 1.5, 1, colors['output'])
        ]
        
        # Connection paths for EfficientNet
        connections = [
            ((2.5, 9.75), (3.5, 9.75)),  # Input to backbone
            ((6, 9.75), (7, 9.75)),      # Backbone to pool
            ((7.75, 9), (4, 7.5)),       # Pool to flatten
            ((4.5, 6.5), (5, 6.5)),      # Flatten to FC256
            ((6, 6.5), (4, 5.5)),        # FC256 to FC100
            ((5, 4.5), (5.5, 4.5)),      # FC100 to FC50
            ((6.25, 4.5), (4.25, 3.5))   # FC50 to output
        ]
        
    else:
        # ResNet architecture
        components = [
            ('ResNet-50\nBackbone', 3.5, 9, 2.5, 1.5, colors['conv']),
            ('Global\nAvgPool', 7, 9, 1.5, 1.5, colors['pool']),
            ('Flatten', 3, 6.5, 1.5, 1, colors['fc']),
            ('FC(256)\n+ BatchNorm\n+ ReLU', 5, 6.5, 2, 1, colors['fc']),
            ('FC(100)\n+ BatchNorm\n+ ReLU', 3, 4.5, 2, 1, colors['fc']),
            ('FC(50)\n+ BatchNorm\n+ ReLU', 5.5, 4.5, 2, 1, colors['fc']),
            ('FC(10)\n+ ReLU', 3.5, 2.5, 1.5, 1, colors['fc']),
            ('FC(1)', 6, 2.5, 1.5, 1, colors['output'])
        ]
        
        # Connection paths for ResNet
        connections = [
            ((2.5, 9.75), (3.5, 9.75)),   # Input to backbone
            ((6, 9.75), (7, 9.75)),       # Backbone to pool
            ((7.75, 9), (4, 7.5)),        # Pool to flatten
            ((4.5, 6.5), (5, 6.5)),       # Flatten to FC256
            ((6, 6.5), (4, 5.5)),         # FC256 to FC100
            ((5, 4.5), (5.5, 4.5)),       # FC100 to FC50
            ((6.5, 4.5), (4.25, 3.5)),    # FC50 to FC10
            ((5.25, 2.5), (6, 2.5))       # FC10 to output
        ]
    
    # Draw components
    for text, x, y, w, h, color in components:
        rect = patches.Rectangle((x, y), w, h, 
                               facecolor=color, 
                               edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
               fontsize=10, weight='bold')
    
    # Draw connections
    for (x1, y1), (x2, y2) in connections:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add output label
    ax.text(9.5, 3.25, 'Steering\nAngle\nPrediction', ha='center', va='center',
           fontsize=11, weight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    # Title
    ax.text(6, 11.5, f'{model_name} Architecture for Autonomous Driving', 
           ha='center', va='center', fontsize=16, weight='bold')
    
    # Add parameter information
    total_params = sum(p.numel() for p in model.parameters())
    model_size = (total_params * 4) / (1024**2)
    
    info_text = f"Total Parameters: {total_params:,}\nModel Size: {model_size:.2f} MB"
    ax.text(10, 1, info_text, ha='left', va='bottom', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Architecture diagram saved to {save_path}")
    
    plt.show()
    return fig

def create_model_summary_table(models, model_names, save_path=None):
    """Create a comprehensive model comparison table."""
    # Analyze both models
    all_stats = []
    for model, name in zip(models, model_names):
        layer_info, model_stats = analyze_model_layers(model, name)
        all_stats.append(model_stats)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_stats, index=model_names)
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Parameter comparison bar chart
    metrics = ['Total Parameters', 'Trainable Parameters', 'Non-trainable Parameters']
    x = np.arange(len(model_names))
    width = 0.25
    
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    
    for i, metric in enumerate(metrics):
        values = [stats[metric] for stats in all_stats]
        ax1.bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Number of Parameters')
    ax1.set_title('Parameter Comparison', fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(model_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, metric in enumerate(metrics):
        values = [stats[metric] for stats in all_stats]
        for j, v in enumerate(values):
            ax1.text(j + i * width, v + max(values) * 0.01, f'{v:,}', 
                    ha='center', va='bottom', fontsize=8, rotation=45)
    
    # 2. Model size comparison
    sizes = [stats['Model Size (MB)'] for stats in all_stats]
    memory = [stats['Estimated Memory (MB)'] for stats in all_stats]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax2.bar(x - width/2, sizes, width, label='Model Size', color='skyblue', alpha=0.8)
    ax2.bar(x + width/2, memory, width, label='Training Memory', color='lightcoral', alpha=0.8)
    
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Size (MB)')
    ax2.set_title('Memory Requirements', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (s, m) in enumerate(zip(sizes, memory)):
        ax2.text(i - width/2, s + max(sizes) * 0.02, f'{s:.1f}', 
                ha='center', va='bottom', fontsize=10)
        ax2.text(i + width/2, m + max(memory) * 0.02, f'{m:.1f}', 
                ha='center', va='bottom', fontsize=10)
      # 3. Layer type distribution for first model
    layer_info_1, _ = analyze_model_layers(models[0], model_names[0])
    layer_types_1 = {}
    for layer in layer_info_1:
        layer_type = layer['Layer Type']
        layer_types_1[layer_type] = layer_types_1.get(layer_type, 0) + 1
    
    colors_1 = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99FFCC', '#CCFF99', '#FFCCFF']
    ax3.pie(layer_types_1.values(), labels=layer_types_1.keys(), autopct='%1.1f%%',
           startangle=90, colors=colors_1[:len(layer_types_1)])
    ax3.set_title(f'{model_names[0]} - Layer Distribution', fontweight='bold')
    
    # 4. Layer type distribution for second model
    layer_info_2, _ = analyze_model_layers(models[1], model_names[1])
    layer_types_2 = {}
    for layer in layer_info_2:
        layer_type = layer['Layer Type']
        layer_types_2[layer_type] = layer_types_2.get(layer_type, 0) + 1
    
    colors_2 = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99FFCC', '#CCFF99', '#FFCCFF']
    ax4.pie(layer_types_2.values(), labels=layer_types_2.keys(), autopct='%1.1f%%',
           startangle=90, colors=colors_2[:len(layer_types_2)])
    ax4.set_title(f'{model_names[1]} - Layer Distribution', fontweight='bold')
    
    plt.suptitle('Comprehensive Model Analysis', fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model analysis saved to {save_path}")
    
    plt.show()
    
    # Save detailed comparison table
    if save_path:
        csv_path = save_path.replace('.pdf', '_table.csv')
        comparison_df.to_csv(csv_path)
        print(f"Comparison table saved to {csv_path}")
    
    return fig, comparison_df

def create_parameter_analysis(models, model_names, save_path=None):
    """Create parameter distribution analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for idx, (model, name) in enumerate(zip(models, model_names)):
        # Extract weights
        weights = []
        layer_names = []
        
        for param_name, param in model.named_parameters():
            if param.requires_grad and 'weight' in param_name:
                weights.extend(param.data.flatten().cpu().numpy())
                layer_names.extend([param_name.split('.')[0]] * param.numel())
        
        # Plot weight distribution
        ax1 = axes[idx, 0]
        ax1.hist(weights, bins=50, alpha=0.7, color=f'C{idx}', edgecolor='black')
        ax1.set_xlabel('Weight Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{name} - Weight Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Plot parameter count by layer
        ax2 = axes[idx, 1]
        param_counts = {}
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                layer_name = param_name.split('.')[0]
                param_counts[layer_name] = param_counts.get(layer_name, 0) + param.numel()
        
        # Sort by parameter count
        sorted_layers = sorted(param_counts.items(), key=lambda x: x[1], reverse=True)
        layer_names = [item[0] for item in sorted_layers[:10]]  # Top 10 layers
        counts = [item[1] for item in sorted_layers[:10]]
        
        bars = ax2.barh(range(len(layer_names)), counts, color=f'C{idx}', alpha=0.8)
        ax2.set_yticks(range(len(layer_names)))
        ax2.set_yticklabels(layer_names)
        ax2.set_xlabel('Number of Parameters')
        ax2.set_title(f'{name} - Top 10 Layers by Parameters')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + max(counts) * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{int(width):,}', ha='left', va='center', fontsize=8)
    
    plt.suptitle('Parameter Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Parameter analysis saved to {save_path}")
    
    plt.show()
    return fig

def main():
    """Create comprehensive model visualizations for PhD thesis."""
    
    print("=" * 60)
    print("COMPREHENSIVE MODEL VISUALIZATION FOR PHD THESIS")
    print("=" * 60)
    
    # Create output directory
    output_dir = "thesis_figures_complete"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load models
    print("\n1. Loading models...")
    resnet_model = NvidiaModel(pretrained=False)
    efficientnet_model = NvidiaModelTransferLearning(pretrained=False)
    
    models = [resnet_model, efficientnet_model]
    model_names = ['ResNet-50 Based', 'EfficientNet-B0 Based']
    
    print(f"✓ Models loaded successfully!")
    
    # Print basic info
    for model, name in zip(models, model_names):
        params = sum(p.numel() for p in model.parameters())
        print(f"  - {name}: {params:,} parameters")
    
    # Create visualizations
    print("\n2. Creating architecture diagrams...")
    for model, name in zip(models, model_names):
        safe_name = name.replace('-', '_').replace(' ', '_').lower()
        create_architecture_diagram(
            model, name, 
            save_path=os.path.join(output_dir, f'{safe_name}_architecture.pdf')
        )
    
    print("\n3. Creating comprehensive model analysis...")
    create_model_summary_table(
        models, model_names,
        save_path=os.path.join(output_dir, 'comprehensive_model_analysis.pdf')
    )
    
    print("\n4. Creating parameter analysis...")
    create_parameter_analysis(
        models, model_names,
        save_path=os.path.join(output_dir, 'parameter_analysis.pdf')
    )
    
    # Create summary text file
    print("\n5. Creating summary report...")
    with open(os.path.join(output_dir, 'model_summary_report.txt'), 'w') as f:
        f.write("Model Comparison Summary for PhD Thesis\n")
        f.write("=" * 50 + "\n\n")
        
        for model, name in zip(models, model_names):
            layer_info, stats = analyze_model_layers(model, name)
            
            f.write(f"{name}:\n")
            f.write("-" * len(name) + "\n")
            for key, value in stats.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.2f}\n")
                else:
                    f.write(f"  {key}: {value:,}\n")
            f.write("\n")
        
        f.write("Files Generated:\n")
        f.write("- Architecture diagrams (PDF)\n")
        f.write("- Comprehensive model analysis (PDF)\n")
        f.write("- Parameter analysis (PDF)\n")
        f.write("- Model comparison table (CSV)\n")
        f.write("- This summary report (TXT)\n")
    
    print(f"\n✓ All visualizations created successfully!")
    print(f"📁 Check '{output_dir}' directory for all files")
    
    # List generated files
    print(f"\nGenerated files:")
    for file in os.listdir(output_dir):
        print(f"  - {file}")
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE!")
    print("All figures are ready for your PhD thesis!")
    print("=" * 60)

if __name__ == "__main__":
    main()
