import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def consolidate_metrics(base_dir='taguchi_runs_GRADCAM_RESULTS'):
    """
    Crawl through all runs and folds to consolidate metrics_summary.csv files
    """
    all_metrics = []
    
    # Get all run directories
    base_path = Path(base_dir)
    run_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('run_')])
    
    print(f"Found {len(run_dirs)} run directories")
    
    for run_dir in run_dirs:
        run_name = run_dir.name
        print(f"Processing {run_name}...")
        
        # Get all fold directories within this run
        fold_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith('fold_')])
        
        for fold_dir in fold_dirs:
            fold_name = fold_dir.name
            csv_path = fold_dir / 'csv' / 'metrics_summary.csv'
            
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    # Add run and fold information if not present
                    if 'run' not in df.columns:
                        df['run'] = run_name
                    if 'fold' not in df.columns:
                        df['fold'] = fold_name
                    
                    all_metrics.append(df)
                    print(f"  ✓ {run_name}/{fold_name}")
                except Exception as e:
                    print(f"  ✗ Error reading {csv_path}: {e}")
            else:
                print(f"  ✗ {csv_path} not found")
    
    # Consolidate all metrics
    if all_metrics:
        consolidated_df = pd.concat(all_metrics, ignore_index=True)
        
        # Save consolidated results
        output_dir = Path(base_dir) / 'CONSOLIDATED_RESULTS'
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / 'consolidated_metrics.csv'
        consolidated_df.to_csv(output_file, index=False)
        print(f"\n✓ Consolidated metrics saved to: {output_file}")
        print(f"Total records: {len(consolidated_df)}")
        
        return consolidated_df, output_dir
    else:
        print("No metrics found!")
        return None, None


def create_per_run_plots(df, output_dir):
    """
    Create plots for each run showing metrics across all folds
    X-axis: Fold numbers (0, 1, 2, 3, 4)
    Y-axis: Metric values
    One plot per run for each metric
    """
    # Extract fold number from fold column (e.g., 'fold_0' -> 0)
    if df['fold'].dtype == 'object':
        df['fold_num'] = df['fold'].str.extract(r'(\d+)').astype(int)
    else:
        df['fold_num'] = df['fold']
    
    # Extract run number for sorting
    if df['run'].dtype == 'object':
        df['run_num'] = df['run'].str.extract(r'(\d+)').astype(int)
    else:
        df['run_num'] = df['run']
    
    # Sort by run and fold
    df = df.sort_values(['run_num', 'fold_num'])
    
    # Define metrics to plot
    metrics = [
        ('opt_risky_recall', 'Risky Recall'),
        ('opt_accuracy', 'Accuracy'),
        ('opt_macro_f1', 'Macro F1')
    ]
    
    # Get all unique runs
    runs = sorted(df['run'].unique(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
    
    print(f"\nCreating plots for {len(runs)} runs...")
    
    # Create directory for each metric
    for metric_col, metric_name in metrics:
        if metric_col not in df.columns:
            print(f"Warning: {metric_col} not found in dataframe")
            continue
        
        metric_dir = output_dir / metric_name.lower().replace(' ', '_')
        metric_dir.mkdir(exist_ok=True)
        
        # Create a plot for each run
        for run in runs:
            run_data = df[df['run'] == run].sort_values('fold_num')
            
            if len(run_data) == 0:
                continue
            
            plt.figure(figsize=(10, 6))
            
            # Plot the metric across folds
            folds = run_data['fold_num'].values
            values = run_data[metric_col].values
            
            plt.plot(folds, values, marker='o', linewidth=2.5, 
                    markersize=10, color='#2E86AB', alpha=0.8)
            
            # Add value labels on points
            for fold, value in zip(folds, values):
                plt.text(fold, value, f'{value:.4f}', 
                        ha='center', va='bottom', fontsize=9)
            
            plt.xlabel('Fold', fontsize=12, fontweight='bold')
            plt.ylabel(metric_name, fontsize=12, fontweight='bold')
            plt.title(f'{metric_name} - {run}', fontsize=14, fontweight='bold')
            plt.xticks(range(5), fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            plot_path = metric_dir / f'{run}_{metric_name.lower().replace(" ", "_")}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✓ Created plots for {metric_name} in: {metric_dir}")
    
    # Create combined plots (all metrics for each run)
    combined_dir = output_dir / 'combined_per_run'
    combined_dir.mkdir(exist_ok=True)
    
    for run in runs:
        run_data = df[df['run'] == run].sort_values('fold_num')
        
        if len(run_data) == 0:
            continue
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (metric_col, metric_name) in enumerate(metrics):
            if metric_col not in df.columns:
                continue
            
            ax = axes[idx]
            folds = run_data['fold_num'].values
            values = run_data[metric_col].values
            
            ax.plot(folds, values, marker='o', linewidth=2.5, 
                   markersize=10, color='#2E86AB', alpha=0.8)
            
            # Add value labels
            for fold, value in zip(folds, values):
                ax.text(fold, value, f'{value:.3f}', 
                       ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Fold', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.set_xticks(range(5))
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'All Metrics - {run}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        combined_path = combined_dir / f'{run}_all_metrics.png'
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Created combined plots in: {combined_dir}")
    
    # Create summary statistics
    summary_stats = df.groupby('run')[['opt_risky_recall', 'opt_accuracy', 'opt_macro_f1']].agg(['mean', 'std', 'min', 'max'])
    summary_path = output_dir / 'run_summary_statistics.csv'
    summary_stats.to_csv(summary_path)
    print(f"\n✓ Saved summary statistics: {summary_path}")
    
    print("\n" + "="*60)
    print("Summary Statistics by Run:")
    print("="*60)
    print(summary_stats.round(4))


def main():
    """
    Main function to consolidate metrics and create plots
    """
    print("="*60)
    print("Starting Metrics Consolidation and Plotting")
    print("="*60 + "\n")
    
    # Step 1: Consolidate all metrics
    consolidated_df, output_dir = consolidate_metrics()
    
    if consolidated_df is not None:
        print("\n" + "="*60)
        print("Creating Per-Run Plots (X-axis: Folds)")
        print("="*60 + "\n")
        
        # Step 2: Create plots
        create_per_run_plots(consolidated_df, output_dir)
        
        print("\n" + "="*60)
        print("✓ All tasks completed successfully!")
        print("="*60)
    else:
        print("Failed to consolidate metrics.")


if __name__ == "__main__":
    main()
