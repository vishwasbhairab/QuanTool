"""
Updated Figure Generation for QuanTool Paper
Generate publication-quality figures for the 3 valid experiments only.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Load only valid experiments
valid_experiments = {
    'bert-base-uncased_sst2': {
        'model': 'BERT-base',
        'task': 'SST-2',
        'fp32_acc': 0.9243,
        'int8_acc': 0.9071,
        'fp32_lat': 4526.38,
        'int8_lat': 1976.33,
        'fp32_size': 417.72,
        'int8_size': 173.09
    },
    'distilbert-base-uncased_sst2': {
        'model': 'DistilBERT',
        'task': 'SST-2',
        'fp32_acc': 0.9106,
        'int8_acc': 0.8968,
        'fp32_lat': 3345.70,
        'int8_lat': 1438.38,
        'fp32_size': 255.45,
        'int8_size': 132.29
    },
    'distilbert-base-uncased_mrpc': {
        'model': 'DistilBERT',
        'task': 'MRPC',
        'fp32_acc': 0.8578,
        'int8_acc': 0.8064,
        'fp32_lat': 3115.00,
        'int8_lat': 1358.38,
        'fp32_size': 255.45,
        'int8_size': 132.29
    }
}

# Create DataFrame
data = []
for key, exp in valid_experiments.items():
    data.append({
        'Experiment': f"{exp['model']}\n{exp['task']}",
        'Model': exp['model'],
        'Task': exp['task'],
        'FP32 Accuracy': exp['fp32_acc'],
        'INT8 Accuracy': exp['int8_acc'],
        'Accuracy Change (%)': (exp['int8_acc'] - exp['fp32_acc']) * 100,
        'FP32 Latency': exp['fp32_lat'],
        'INT8 Latency': exp['int8_lat'],
        'Latency Speedup': exp['fp32_lat'] / exp['int8_lat'],
        'FP32 Size': exp['fp32_size'],
        'INT8 Size': exp['int8_size'],
        'Size Reduction': exp['fp32_size'] / exp['int8_size']
    })

df = pd.DataFrame(data)

# ============================================================================
# Figure 1: Comprehensive 4-Panel Analysis (Main Paper Figure)
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('QuanTool Benchmark Analysis: INT8 Dynamic Quantization', 
             fontsize=14, fontweight='bold', y=0.995)

experiments = df['Experiment'].values
x_pos = np.arange(len(experiments))
width = 0.35

# Color scheme
color_fp32 = '#3498db'  # Blue
color_int8 = '#e74c3c'  # Red
color_speedup = '#2ecc71'  # Green
color_reduction = '#9b59b6'  # Purple

# Panel 1: Accuracy Comparison
ax = axes[0, 0]
bars1 = ax.bar(x_pos - width/2, df['FP32 Accuracy'], width, 
               label='FP32', color=color_fp32, alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x_pos + width/2, df['INT8 Accuracy'], width, 
               label='INT8-Dynamic', color=color_int8, alpha=0.8, edgecolor='black', linewidth=0.5)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)

ax.set_ylabel('Accuracy', fontweight='bold')
ax.set_title('(a) Accuracy: FP32 vs INT8 Dynamic', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(experiments, rotation=0, ha='center')
ax.legend(loc='lower left', framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim([0.75, 0.95])

# Panel 2: Latency Speedup
ax = axes[0, 1]
bars = ax.bar(x_pos, df['Latency Speedup'], color=color_speedup, 
              alpha=0.8, edgecolor='black', linewidth=0.5)
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, 
           label='Baseline (1×)', alpha=0.7)

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}×',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Speedup Factor', fontweight='bold')
ax.set_title('(b) Latency Speedup with INT8 Dynamic', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(experiments, rotation=0, ha='center')
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim([0, 2.8])

# Panel 3: Model Size Reduction
ax = axes[1, 0]
bars = ax.bar(x_pos, df['Size Reduction'], color=color_reduction, 
              alpha=0.8, edgecolor='black', linewidth=0.5)
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, 
           label='Baseline (1×)', alpha=0.7)

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}×',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Compression Factor', fontweight='bold')
ax.set_title('(c) Model Size Reduction with INT8 Dynamic', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(experiments, rotation=0, ha='center')
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim([0, 2.8])

# Panel 4: Accuracy-Latency Trade-off Scatter
ax = axes[1, 1]
scatter = ax.scatter(df['Accuracy Change (%)'], df['Latency Speedup'],
                    s=df['Size Reduction'] * 200,  # Size proportional to compression
                    c=range(len(df)), cmap='viridis',
                    alpha=0.7, edgecolors='black', linewidth=1.5)

# Add labels for each point
for i, row in df.iterrows():
    ax.annotate(f"{row['Model']}\n{row['Task']}", 
               (row['Accuracy Change (%)'], row['Latency Speedup']),
               xytext=(5, 5), textcoords='offset points',
               fontsize=8, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        alpha=0.7, edgecolor='gray'))

ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.set_xlabel('Accuracy Change (%)', fontweight='bold')
ax.set_ylabel('Latency Speedup (×)', fontweight='bold')
ax.set_title('(d) Accuracy-Latency Trade-off\n(bubble size = compression ratio)', 
             fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim([-6, 0.5])
ax.set_ylim([2.1, 2.5])

# Add legend for bubble sizes
handles, labels = [], []
for size_val in [1.93, 2.41]:
    handles.append(plt.scatter([], [], s=size_val*200, c='gray', alpha=0.5, 
                              edgecolors='black'))
    labels.append(f'{size_val:.2f}× compression')
ax.legend(handles, labels, loc='lower right', title='Size Reduction', 
         framealpha=0.9, fontsize=8)

plt.tight_layout()
plt.savefig('figure1_comprehensive_analysis.png', bbox_inches='tight', dpi=300)
plt.savefig('figure1_comprehensive_analysis.pdf', bbox_inches='tight')
print("✓ Saved: figure1_comprehensive_analysis.png/pdf")
plt.close()

# ============================================================================
# Figure 2: Detailed Metrics Table Visualization
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = []
table_data.append(['Model-Task', 'Precision', 'Accuracy', 'Accuracy Δ', 
                   'Latency (ms)', 'Speedup', 'Size (MB)', 'Compression'])

for _, row in df.iterrows():
    # FP32 row
    table_data.append([
        f"{row['Model']}\n{row['Task']}",
        'FP32',
        f"{row['FP32 Accuracy']:.4f}",
        '—',
        f"{row['FP32 Latency']:.1f}",
        '1.00×',
        f"{row['FP32 Size']:.1f}",
        '1.00×'
    ])
    # INT8 row
    acc_change = row['Accuracy Change (%)']
    table_data.append([
        '',
        'INT8-Dyn',
        f"{row['INT8 Accuracy']:.4f}",
        f"{acc_change:+.2f}%",
        f"{row['INT8 Latency']:.1f}",
        f"{row['Latency Speedup']:.2f}×",
        f"{row['INT8 Size']:.1f}",
        f"{row['Size Reduction']:.2f}×"
    ])

# Create table
table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.15, 0.10, 0.10, 0.10, 0.13, 0.10, 0.12, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(len(table_data[0])):
    cell = table[(0, i)]
    cell.set_facecolor('#34495e')
    cell.set_text_props(weight='bold', color='white')

# Style data rows
for i in range(1, len(table_data)):
    for j in range(len(table_data[0])):
        cell = table[(i, j)]
        if i % 2 == 1:  # FP32 rows
            cell.set_facecolor('#ecf0f1')
        else:  # INT8 rows
            cell.set_facecolor('#d5dbdb')
        
        # Highlight improvements
        if j in [5, 7] and i % 2 == 0:  # Speedup and compression columns for INT8
            cell.set_text_props(weight='bold', color='#27ae60')

plt.title('Table I: Comprehensive Quantization Results on Verified Fine-Tuned Models',
         fontsize=14, fontweight='bold', pad=20)
plt.savefig('figure2_results_table.png', bbox_inches='tight', dpi=300)
print("✓ Saved: figure2_results_table.png")
plt.close()

# ============================================================================
# Figure 3: Task-Specific Analysis
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Task-Specific Quantization Analysis', fontsize=14, fontweight='bold')

# Panel 1: SST-2 Comparison (BERT vs DistilBERT)
sst2_data = df[df['Task'] == 'SST-2']
ax = axes[0]
x = np.arange(len(sst2_data))
width = 0.25

metrics = ['Accuracy', 'Speedup', 'Compression']
fp32_acc = sst2_data['FP32 Accuracy'].values
int8_acc = sst2_data['INT8 Accuracy'].values
speedup = sst2_data['Latency Speedup'].values
compression = sst2_data['Size Reduction'].values

# Normalize for visualization
norm_fp32_acc = fp32_acc / fp32_acc.max()
norm_int8_acc = int8_acc / int8_acc.max()
norm_speedup = speedup / 3.0  # Scale to similar range
norm_compression = compression / 3.0

x_pos = np.arange(len(sst2_data))
width = 0.2

ax.bar(x_pos - 1.5*width, norm_fp32_acc, width, label='FP32 Accuracy', 
       color='#3498db', alpha=0.8)
ax.bar(x_pos - 0.5*width, norm_int8_acc, width, label='INT8 Accuracy', 
       color='#e74c3c', alpha=0.8)
ax.bar(x_pos + 0.5*width, norm_speedup, width, label='Latency Speedup (÷3)', 
       color='#2ecc71', alpha=0.8)
ax.bar(x_pos + 1.5*width, norm_compression, width, label='Size Reduction (÷3)', 
       color='#9b59b6', alpha=0.8)

ax.set_ylabel('Normalized Value', fontweight='bold')
ax.set_title('(a) SST-2: BERT vs DistilBERT', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(sst2_data['Model'].values)
ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim([0, 1.1])

# Panel 2: DistilBERT: SST-2 vs MRPC
distilbert_data = df[df['Model'] == 'DistilBERT']
ax = axes[1]

categories = ['Accuracy\nPreservation', 'Latency\nSpeedup', 'Size\nReduction']
x_pos = np.arange(len(categories))
width = 0.35

sst2_vals = [
    1 - abs(distilbert_data[distilbert_data['Task'] == 'SST-2']['Accuracy Change (%)'].values[0] / 100),
    distilbert_data[distilbert_data['Task'] == 'SST-2']['Latency Speedup'].values[0] / 3,
    distilbert_data[distilbert_data['Task'] == 'SST-2']['Size Reduction'].values[0] / 3
]

mrpc_vals = [
    1 - abs(distilbert_data[distilbert_data['Task'] == 'MRPC']['Accuracy Change (%)'].values[0] / 100),
    distilbert_data[distilbert_data['Task'] == 'MRPC']['Latency Speedup'].values[0] / 3,
    distilbert_data[distilbert_data['Task'] == 'MRPC']['Size Reduction'].values[0] / 3
]

bars1 = ax.bar(x_pos - width/2, sst2_vals, width, label='SST-2 (Sentiment)', 
               color='#3498db', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, mrpc_vals, width, label='MRPC (Paraphrase)', 
               color='#e67e22', alpha=0.8)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)

ax.set_ylabel('Normalized Score', fontweight='bold')
ax.set_title('(b) DistilBERT: Task Comparison', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(categories)
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig('figure3_task_analysis.png', bbox_inches='tight', dpi=300)
print("✓ Saved: figure3_task_analysis.png")
plt.close()

# ============================================================================
# Figure 4: Summary Statistics for Paper
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

summary_stats = {
    'Metric': ['Accuracy Change', 'Latency Speedup', 'Size Reduction'],
    'Min': [
        f"{df['Accuracy Change (%)'].min():.2f}%",
        f"{df['Latency Speedup'].min():.2f}×",
        f"{df['Size Reduction'].min():.2f}×"
    ],
    'Max': [
        f"{df['Accuracy Change (%)'].max():.2f}%",
        f"{df['Latency Speedup'].max():.2f}×",
        f"{df['Size Reduction'].max():.2f}×"
    ],
    'Mean': [
        f"{df['Accuracy Change (%)'].mean():.2f}%",
        f"{df['Latency Speedup'].mean():.2f}×",
        f"{df['Size Reduction'].mean():.2f}×"
    ],
    'Std Dev': [
        f"{df['Accuracy Change (%)'].std():.2f}%",
        f"{df['Latency Speedup'].std():.3f}×",
        f"{df['Size Reduction'].std():.3f}×"
    ]
}

summary_df = pd.DataFrame(summary_stats)

ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=summary_df.values, 
                colLabels=summary_df.columns,
                cellLoc='center',
                loc='center',
                colWidths=[0.25, 0.18, 0.18, 0.18, 0.18])

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 3)

# Style header
for i in range(len(summary_df.columns)):
    cell = table[(0, i)]
    cell.set_facecolor('#34495e')
    cell.set_text_props(weight='bold', color='white', size=13)

# Style data rows
for i in range(1, len(summary_df) + 1):
    for j in range(len(summary_df.columns)):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#ecf0f1')
        else:
            cell.set_facecolor('#ffffff')
        
        if j == 0:  # Metric name column
            cell.set_text_props(weight='bold')

plt.title('Summary Statistics: INT8 Dynamic Quantization Performance',
         fontsize=14, fontweight='bold', pad=20)
plt.savefig('figure4_summary_stats.png', bbox_inches='tight', dpi=300)
print("✓ Saved: figure4_summary_stats.png")
plt.close()

# ============================================================================
# Generate Summary Report
# ============================================================================

print("\n" + "="*70)
print("FIGURE GENERATION COMPLETE")
print("="*70)
print("\nGenerated Files:")
print("  1. figure1_comprehensive_analysis.png/pdf - Main 4-panel figure")
print("  2. figure2_results_table.png - Detailed results table")
print("  3. figure3_task_analysis.png - Task-specific comparisons")
print("  4. figure4_summary_stats.png - Summary statistics")
print("\nKey Statistics:")
print(f"  • Experiments: {len(df)}")
print(f"  • Accuracy change: {df['Accuracy Change (%)'].min():.2f}% to {df['Accuracy Change (%)'].max():.2f}%")
print(f"  • Latency speedup: {df['Latency Speedup'].min():.2f}× to {df['Latency Speedup'].max():.2f}×")
print(f"  • Size reduction: {df['Size Reduction'].min():.2f}× to {df['Size Reduction'].max():.2f}×")
print("\n✓ All figures ready for paper submission!")
print("="*70)