import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import os

# Read the data
df = pd.read_csv('results/master_results.csv')

# Create output directory if it doesn't exist
os.makedirs('results/plots', exist_ok=True)

# Set style
sns.set_style("whitegrid")

# Create figure with 4 subplots (2x2 grid for 4 model-task combinations)
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.flatten()

# Define colors and markers
backend_colors = {'openvino': '#2E86AB', 'pytorch': '#A23B72'}
precision_markers = {'fp32': 'o', 'int8': 's', 'int4': '^'}

# Normalize accuracy to bubble size (scale between 100-1000 for visibility)
def accuracy_to_size(accuracy):
    # Scale accuracy (0.5-1.0 range) to bubble size (100-800)
    return (accuracy - 0.5) * 1400 + 100

# Get unique model-task combinations
model_task_combinations = df.groupby(['model', 'task']).size().reset_index()[['model', 'task']]

# Plot each model-task combination in its own subplot
for idx, (_, row) in enumerate(model_task_combinations.iterrows()):
    model = row['model']
    task = row['task']
    
    ax = axes[idx]
    
    # Filter data for this model-task combination
    subset = df[(df['model'] == model) & (df['task'] == task)]
    
    # Plot each point
    for _, data_row in subset.iterrows():
        ax.scatter(
            data_row['size_reduction_x'], 
            data_row['latency_speedup_x'],
            s=accuracy_to_size(data_row['accuracy']),
            alpha=0.6,
            marker=precision_markers[data_row['precision']],
            color=backend_colors[data_row['backend']],
            edgecolors='black',
            linewidth=2
        )
        
        # Add precision label on each point
        ax.annotate(
            data_row['precision'],
            (data_row['size_reduction_x'], data_row['latency_speedup_x']),
            fontsize=8,
            ha='center',
            va='center',
            fontweight='bold'
        )
    
    # Styling
    ax.set_xlabel('Size Reduction (x)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Latency Speedup (x)', fontsize=11, fontweight='bold')
    ax.set_title(f"{model.replace('-base-uncased', '').upper()}\n{task.upper()}", 
                 fontsize=12, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    
    # Add reference lines at 1.0 (baseline)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Set consistent axis limits for better comparison
    ax.set_xlim(0.5, 7)
    ax.set_ylim(0.5, 2.8)

# Create unified legend
legend_elements = [
    # Backend colors
    Line2D([0], [0], marker='o', color='w', markerfacecolor=backend_colors['openvino'], 
           markersize=12, label='OpenVINO', markeredgecolor='black', markeredgewidth=1.5),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=backend_colors['pytorch'], 
           markersize=12, label='PyTorch', markeredgecolor='black', markeredgewidth=1.5),
    # Spacer
    Line2D([0], [0], color='none', label=''),
    # Precision markers
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
           markersize=10, label='fp32', markeredgecolor='black', markeredgewidth=1.5),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
           markersize=10, label='int8', markeredgecolor='black', markeredgewidth=1.5),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', 
           markersize=10, label='int4', markeredgecolor='black', markeredgewidth=1.5),
    # Spacer
    Line2D([0], [0], color='none', label=''),
    # Bubble size legend
    Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
           markersize=8, label='Accuracy: 0.50', markeredgecolor='black', markeredgewidth=1),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
           markersize=14, label='Accuracy: 0.75', markeredgecolor='black', markeredgewidth=1),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
           markersize=20, label='Accuracy: 0.95', markeredgecolor='black', markeredgewidth=1),
]

# Add legend outside the subplots
fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.0, 0.5), 
           fontsize=11, frameon=True, fancybox=True, shadow=True)

# Add main title
fig.suptitle('Model Optimization Performance Dashboard\n' + 
             'Top-Right with Larger Bubbles = Best (High Compression + High Speed + High Accuracy)',
             fontsize=14, fontweight='bold', y=0.985)

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.35, wspace=0.30, left=0.08, right=0.85, top=0.94, bottom=0.06)

# Save plots
plt.savefig('results/plots/optimization_dashboard.png', dpi=300, bbox_inches='tight')
plt.savefig('results/plots/optimization_dashboard.pdf', bbox_inches='tight')

print("Dashboard plot saved successfully!")
print("- PNG: results/plots/optimization_dashboard.png")
print("- PDF: results/plots/optimization_dashboard.pdf")
print("\nReading guide:")
print("- X-axis: Higher = Better compression")
print("- Y-axis: Higher = Faster inference")
print("- Bubble size: Larger = Better accuracy")
print("- Top-right corner with large bubbles = Optimal configuration")

plt.show()