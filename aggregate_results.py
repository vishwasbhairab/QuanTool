import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_all_results():
    """Aggregate results from all CSV files."""
    csv_files = glob.glob('benchmark_results_*.csv')

    if not csv_files:
        print("❌ No 'benchmark_results_*.csv' files found. Run the benchmarks first.")
        return

    all_data = []
    for csv_file in csv_files:
        # Parse filename to extract model and task
        # Assumes format like 'benchmark_results_distilbert-base-uncased_sst2.csv'
        try:
            parts = csv_file.replace('benchmark_results_', '').replace('.csv', '').rsplit('_', 1)
            model = parts[0]
            task = parts[1]

            df = pd.read_csv(csv_file, index_col=0)
            df['model'] = model.replace('textattack/', '').replace('-finetuned-sst-2-english', '') # Clean up names
            df['task'] = task
            df['quantization'] = df.index

            all_data.append(df)
        except IndexError:
            print(f"⚠️  Skipping malformed file: {csv_file}")
            continue

    if not all_data:
        print("❌ No valid benchmark files could be parsed.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # Save combined results
    combined_df.to_csv('all_benchmark_results.csv', index=False)
    print("✅ Combined results saved to all_benchmark_results.csv")

    # Create summary table for paper
    create_paper_table(combined_df)

    # Create comprehensive plots
    create_comprehensive_plots(combined_df)

    return combined_df

def create_paper_table(df):
    """Create LaTeX table for paper."""
    # Filter for the rows we need to calculate changes
    fp32_df = df[df['quantization'] == 'Float32'].set_index(['model', 'task'])
    dynamic_df = df[df['quantization'] == 'INT8-Dynamic'].set_index(['model', 'task'])

    # Reconstruct the paper table safely
    paper_table = dynamic_df.copy()
    paper_table['FP32 Accuracy'] = fp32_df['accuracy']
    paper_table = paper_table.rename(columns={'accuracy': 'INT8 Accuracy'})
    paper_table = paper_table.reset_index()

    paper_table = pd.DataFrame({
        'Model': paper_table['model'],
        'Task': paper_table['task'],
        'FP32 Accuracy': paper_table['FP32 Accuracy'],
        'INT8 Accuracy': paper_table['INT8 Accuracy'],
        'Accuracy Δ (%)': paper_table['Accuracy Change (%)'],
        'Latency Speedup': paper_table['Latency Speedup (x)'],
        'Size Reduction': paper_table['Size Reduction (x)']
    })

    # Save as LaTeX
    latex_table = paper_table.to_latex(
        index=False,
        float_format='%.3f',
        caption='Comprehensive quantization results across models and tasks.',
        label='tab:comprehensive_results',
        column_format='llrrrrr'
    )

    with open('paper_table.tex', 'w', encoding='utf-8') as f:
        f.write(latex_table)
    print("✅ LaTeX table saved to paper_table.tex")

def create_comprehensive_plots(df):
    """Create publication-quality plots."""
    sns.set_theme(style="whitegrid")

    # Filter for dynamic quantization and get FP32 baseline
    dynamic_df = df[df['quantization'] == 'INT8-Dynamic'].copy()
    fp32_df = df[df['quantization'] == 'Float32'].set_index(['model', 'task'])
    dynamic_df['fp32_accuracy'] = dynamic_df.apply(lambda row: fp32_df.loc[(row['model'], row['task'])]['accuracy'], axis=1)

    dynamic_df['label'] = dynamic_df['model'].str.replace('-base-uncased', '', regex=False) + '\n(' + dynamic_df['task'] + ')'

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('QuanTool Benchmark Analysis: INT8 Dynamic Quantization', fontsize=20)

    # Plot 1: Accuracy comparison
    ax = axes[0, 0]
    x = range(len(dynamic_df))
    width = 0.35
    ax.bar([i - width/2 for i in x], dynamic_df['fp32_accuracy'], width, label='FP32', color='cornflowerblue')
    ax.bar([i + width/2 for i in x], dynamic_df['accuracy'], width, label='INT8-Dynamic', color='salmon')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Comparison')
    ax.set_xticks(x, labels=dynamic_df['label'], rotation=45, ha='right')
    ax.legend()

    # Plot 2: Latency speedup
    ax = axes[0, 1]
    sns.barplot(x='label', y='Latency Speedup (x)', data=dynamic_df, ax=ax, palette='viridis')
    ax.axhline(y=1.0, color='r', linestyle='--', label='Baseline')
    ax.set_title('Inference Speedup')
    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Plot 3: Size reduction
    ax = axes[1, 0]
    sns.barplot(x='label', y='Size Reduction (x)', data=dynamic_df, ax=ax, palette='plasma')
    ax.axhline(y=1.0, color='r', linestyle='--')
    ax.set_title('Model Size Reduction')
    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Plot 4: Trade-off scatter
    ax = axes[1, 1]
    sns.scatterplot(
        data=dynamic_df,
        x='Accuracy Change (%)',
        y='Latency Speedup (x)',
        size='Size Reduction (x)',
        hue='model',
        sizes=(50, 500),
        alpha=0.7,
        ax=ax,
        palette='muted'
    )
    ax.set_title('Accuracy vs. Speedup Trade-off')
    ax.axvline(x=0, color='grey', linestyle='--', alpha=0.5)
    ax.grid(True)
    ax.legend(title='Model Architecture')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('comprehensive_analysis.png', dpi=300)
    print("✅ Comprehensive plot saved to comprehensive_analysis.png")

if __name__ == '__main__':
    df = aggregate_all_results()
    if df is not None:
        print("\n--- Summary Statistics ---")
        print(df.groupby(['quantization'])[['Accuracy Change (%)', 'Latency Speedup (x)', 'Size Reduction (x)']].describe())