import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import os

# Read the data
df = pd.read_csv('results/master_results.csv')

# Create output directory if it doesn't exist
os.makedirs('results/plots', exist_ok=True)

# Create a combined identifier for better visualization
df['model_task'] = df['model'].str.replace('-base-uncased', '') + '\n' + df['task']
df['backend_precision'] = df['backend'] + '-' + df['precision']

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 5)

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Define colors and markers
backend_colors = {'openvino': 'blue', 'pytorch': 'red'}
precision_markers = {'fp32': 'o', 'int8': 's', 'int4': '^'}

# Plot 1: Latency vs Precision
ax1 = axes[0]
for (model, task), group in df.groupby(['model', 'task']):
    for backend in group['backend'].unique():
        subset = group[group['backend'] == backend]
        # Order by precision for line connection
        subset = subset.sort_values('precision', ascending=False)
        
        label = f"{model.replace('-base-uncased', '')} ({task}) - {backend}"
        ax1.plot(subset['precision'], subset['avg_latency_ms'], 
                marker='o', label=label, 
                color=backend_colors[backend],
                linestyle='-' if 'bert-base' in model else '--',
                linewidth=2, markersize=8)

ax1.set_xlabel('Precision', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average Latency (ms)', fontsize=12, fontweight='bold')
ax1.set_title('Latency vs Precision\n(Lower is Better)', fontsize=13, fontweight='bold')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Model Size vs Precision
ax2 = axes[1]
for (model, task), group in df.groupby(['model', 'task']):
    for backend in group['backend'].unique():
        subset = group[group['backend'] == backend]
        subset = subset.sort_values('precision', ascending=False)
        
        label = f"{model.replace('-base-uncased', '')} ({task}) - {backend}"
        ax2.plot(subset['precision'], subset['model_size_mb'], 
                marker='o', label=label,
                color=backend_colors[backend],
                linestyle='-' if 'bert-base' in model else '--',
                linewidth=2, markersize=8)

ax2.set_xlabel('Precision', fontsize=12, fontweight='bold')
ax2.set_ylabel('Model Size (MB)', fontsize=12, fontweight='bold')
ax2.set_title('Model Size vs Precision\n(Lower is Better)', fontsize=13, fontweight='bold')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 3: Accuracy vs Compression (scatter)
ax3 = axes[2]
for (model, task, backend), group in df.groupby(['model', 'task', 'backend']):
    for _, row in group.iterrows():
        ax3.scatter(row['size_reduction_x'], row['accuracy'], 
                   s=200, alpha=0.7,
                   marker=precision_markers[row['precision']],
                   color=backend_colors[backend],
                   edgecolors='black', linewidth=1.5,
                   label=f"{model.replace('-base-uncased', '')} ({task}) - {backend} - {row['precision']}")

# Remove duplicate labels
handles, labels = ax3.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax3.legend(by_label.values(), by_label.keys(), 
          bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

ax3.set_xlabel('Size Reduction Factor (x)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax3.set_title('Accuracy vs Compression\n(Top-Right is Best)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Add marker legend for precision types
precision_legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
           markersize=10, label='fp32'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
           markersize=10, label='int8'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', 
           markersize=10, label='int4')
]
ax3.add_artist(ax3.legend(handles=precision_legend_elements, 
                          loc='lower right', title='Precision', fontsize=9))

plt.tight_layout()

# Save plots
plt.savefig('results/plots/model_comparison_plots.png', dpi=300, bbox_inches='tight')
plt.savefig('results/plots/model_comparison_plots.pdf', bbox_inches='tight')

print("Plots saved successfully!")
print("- PNG: results/plots/model_comparison_plots.png")
print("- PDF: results/plots/model_comparison_plots.pdf")

plt.show()