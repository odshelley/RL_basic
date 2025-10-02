#!/usr/bin/env python3
"""
Ablation Study Analysis: Choquet Integral vs Hyperbolic Discounting Effects
Analyzes the individual and combined effects of Choquet expectation and hyperbolic discounting
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def extract_tensorboard_data(log_dir):
    """Extract data from tensorboard logs"""
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    data = {}
    
    # Extract scalar data
    scalar_tags = ea.Tags()['scalars']
    for tag in scalar_tags:
        scalar_events = ea.Scalars(tag)
        data[tag] = {
            'steps': [event.step for event in scalar_events],
            'values': [event.value for event in scalar_events]
        }
    
    return data

def analyze_ablation_experiments():
    """Analyze ablation study experiments"""
    runs_dir = "/home/george/projects/personal/RL_basic/runs"
    examples_runs_dir = "/home/george/projects/personal/RL_basic/examples/runs"
    
    # Define ablation experiment configurations
    ablation_experiments = {
        'Choquet + Hyperbolic': {
            'pattern': 'HalfCheetah-v4__choquet_hyperbolic_ablation*',
            'choquet': True,
            'hyperbolic': True,
            'color': '#2ca02c',
            'description': 'Both Choquet integral and hyperbolic discounting'
        },
        'Standard + Hyperbolic': {
            'pattern': 'HalfCheetah-v4__standard_hyperbolic_ablation*',
            'choquet': False,
            'hyperbolic': True,
            'color': '#ff7f0e',
            'description': 'Standard expectation with hyperbolic discounting'
        },
        'Choquet + Standard': {
            'pattern': 'HalfCheetah-v4__choquet_standard_ablation*',
            'choquet': True,
            'hyperbolic': False,
            'color': '#1f77b4',
            'description': 'Choquet integral with standard discounting'
        }
    }
    
    # Also include original experiments for comparison (trimmed to 200k steps)
    original_experiments = {
        'Standard SAC (Original)': {
            'pattern': 'HalfCheetah-v4__standard_sac_halfcheetah*',
            'choquet': False,
            'hyperbolic': False,
            'color': '#d62728',
            'description': 'Original standard SAC baseline'
        },
        'Inverse S-Curve (Original)': {
            'pattern': 'HalfCheetah-v4__inverse_s_curve_halfcheetah*',
            'choquet': True,
            'hyperbolic': False,
            'color': '#9467bd',
            'description': 'Original best-performing Choquet configuration'
        }
    }
    
    # Extract data from all experiments
    all_data = {}
    
    # Process ablation experiments (check both directories)
    for exp_name, config in ablation_experiments.items():
        matching_dirs = glob.glob(os.path.join(runs_dir, config['pattern']))
        if not matching_dirs:
            # Check examples/runs directory
            matching_dirs = glob.glob(os.path.join(examples_runs_dir, config['pattern']))
        
        if matching_dirs:
            log_dir = matching_dirs[0]  # Take first match
            print(f"Processing {exp_name}: {log_dir}")
            try:
                data = extract_tensorboard_data(log_dir)
                all_data[exp_name] = {
                    'data': data,
                    'config': config
                }
            except Exception as e:
                print(f"Error processing {exp_name}: {e}")
                continue
        else:
            print(f"No matching directory found for {exp_name}")
    
    # Process original experiments (trim to 200k steps for comparison)
    for exp_name, config in original_experiments.items():
        matching_dirs = glob.glob(os.path.join(runs_dir, config['pattern']))
        if matching_dirs:
            log_dir = matching_dirs[0]
            print(f"Processing {exp_name}: {log_dir}")
            try:
                data = extract_tensorboard_data(log_dir)
                # Trim data to 200k steps for fair comparison
                trimmed_data = {}
                for metric, values in data.items():
                    if 'steps' in values and 'values' in values:
                        # Keep only data up to 200k steps
                        mask = np.array(values['steps']) <= 200000
                        trimmed_data[metric] = {
                            'steps': np.array(values['steps'])[mask].tolist(),
                            'values': np.array(values['values'])[mask].tolist()
                        }
                    else:
                        trimmed_data[metric] = values
                
                all_data[exp_name] = {
                    'data': trimmed_data,
                    'config': config
                }
            except Exception as e:
                print(f"Error processing {exp_name}: {e}")
                continue
        else:
            print(f"No matching directory found for {exp_name}")
    
    return all_data

def create_ablation_performance_plot(all_data):
    """Create ablation study performance comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Learning curves
    for exp_name, exp_data in all_data.items():
        if 'charts/episodic_return' in exp_data['data']:
            steps = exp_data['data']['charts/episodic_return']['steps']
            returns = exp_data['data']['charts/episodic_return']['values']
            color = exp_data['config']['color']
            
            # Smooth the curve
            if len(returns) > 10:
                window = min(50, len(returns) // 10)
                smoothed = pd.Series(returns).rolling(window=window, center=True).mean()
                ax1.plot(steps, smoothed, label=exp_name, color=color, linewidth=2.5)
                
                # Add confidence bands
                if len(returns) > window:
                    std_dev = pd.Series(returns).rolling(window=window, center=True).std()
                    ax1.fill_between(steps, 
                                   smoothed - std_dev/2, 
                                   smoothed + std_dev/2,
                                   alpha=0.2, color=color)
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Episodic Return')
    ax1.set_title('Ablation Study: Learning Curves Comparison')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 200000)
    
    # Plot 2: Final performance comparison
    final_performance = {}
    for exp_name, exp_data in all_data.items():
        if 'charts/episodic_return' in exp_data['data']:
            returns = exp_data['data']['charts/episodic_return']['values']
            if len(returns) > 10:
                # Take last 20% of episodes for final performance
                final_performance[exp_name] = {
                    'mean': np.mean(returns[-len(returns)//5:]),
                    'std': np.std(returns[-len(returns)//5:]),
                    'color': exp_data['config']['color']
                }
    
    if final_performance:
        exp_names = list(final_performance.keys())
        means = [final_performance[name]['mean'] for name in exp_names]
        stds = [final_performance[name]['std'] for name in exp_names]
        colors = [final_performance[name]['color'] for name in exp_names]
        
        bars = ax2.bar(range(len(exp_names)), means, yerr=stds, 
                       color=colors, alpha=0.7, capsize=5)
        ax2.set_xticks(range(len(exp_names)))
        ax2.set_xticklabels(exp_names, rotation=45, ha='right')
        ax2.set_ylabel('Final Episodic Return')
        ax2.set_title('Final Performance Comparison (200k steps)')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'{mean:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/george/projects/personal/RL_basic/ablation_performance_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return final_performance

def create_ablation_effects_analysis(all_data):
    """Analyze individual effects of Choquet vs Hyperbolic"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Metrics to analyze
    behavioral_metrics = [
        'behavioral/distortion_bias',
        'behavioral/reward_std',
        'discounting/effective_gamma',
        'losses/qf_loss'
    ]
    
    titles = [
        'Distortion Bias (Choquet Effect)',
        'Reward Standard Deviation',
        'Effective Gamma (Hyperbolic Effect)',
        'Q-Function Loss'
    ]
    
    for idx, (metric, title) in enumerate(zip(behavioral_metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        for exp_name, exp_data in all_data.items():
            if metric in exp_data['data']:
                steps = exp_data['data'][metric]['steps']
                values = exp_data['data'][metric]['values']
                color = exp_data['config']['color']
                
                # Smooth the data
                if len(values) > 10:
                    window = min(20, len(values) // 5)
                    smoothed = pd.Series(values).rolling(window=window, center=True).mean()
                    ax.plot(steps, smoothed, label=exp_name, color=color, linewidth=2)
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel(metric.split('/')[-1].replace('_', ' ').title())
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 200000)
    
    plt.tight_layout()
    plt.savefig('/home/george/projects/personal/RL_basic/ablation_effects_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_ablation_contribution_matrix(final_performance):
    """Create matrix showing individual contributions"""
    # Calculate effect sizes
    effects_data = []
    
    # Find baseline and experimental conditions
    baseline_perf = None
    for name, perf in final_performance.items():
        if 'Standard SAC (Original)' in name:
            baseline_perf = perf['mean']
            break
    
    if baseline_perf is None:
        print("No baseline found, using first experiment as baseline")
        baseline_perf = list(final_performance.values())[0]['mean']
    
    # Calculate relative performance changes
    for exp_name, perf in final_performance.items():
        performance_change = ((perf['mean'] - baseline_perf) / abs(baseline_perf)) * 100
        
        # Determine components
        choquet_effect = 'Choquet' in exp_name or 'Inverse S-Curve' in exp_name
        hyperbolic_effect = 'Hyperbolic' in exp_name
        
        effects_data.append({
            'Experiment': exp_name,
            'Choquet': choquet_effect,
            'Hyperbolic': hyperbolic_effect,
            'Performance_Change': performance_change,
            'Final_Performance': perf['mean']
        })
    
    df = pd.DataFrame(effects_data)
    
    # Create contribution matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Matrix plot
    matrix_data = df.pivot_table(
        index='Choquet', 
        columns='Hyperbolic', 
        values='Performance_Change', 
        aggfunc='mean'
    )
    
    sns.heatmap(matrix_data, annot=True, cmap='RdYlGn', center=0, ax=ax1, 
                fmt='.1f', cbar_kws={'label': 'Performance Change (%)'})
    ax1.set_title('Performance Change Matrix\n(Relative to Baseline)')
    ax1.set_xlabel('Hyperbolic Discounting')
    ax1.set_ylabel('Choquet Integral')
    
    # Effect size plot
    choquet_only = df[(df['Choquet'] == True) & (df['Hyperbolic'] == False)]['Performance_Change'].mean()
    hyperbolic_only = df[(df['Choquet'] == False) & (df['Hyperbolic'] == True)]['Performance_Change'].mean()
    both_effects = df[(df['Choquet'] == True) & (df['Hyperbolic'] == True)]['Performance_Change'].mean()
    neither = df[(df['Choquet'] == False) & (df['Hyperbolic'] == False)]['Performance_Change'].mean()
    
    effects = ['Neither', 'Choquet Only', 'Hyperbolic Only', 'Both']
    values = [neither, choquet_only, hyperbolic_only, both_effects]
    colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c']
    
    bars = ax2.bar(effects, values, color=colors, alpha=0.7)
    ax2.set_ylabel('Performance Change (%)')
    ax2.set_title('Individual and Combined Effects')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -1),
                f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/george/projects/personal/RL_basic/ablation_contribution_matrix.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return df

def create_ablation_summary_table(df):
    """Create summary table for ablation results"""
    summary = df.groupby(['Choquet', 'Hyperbolic']).agg({
        'Final_Performance': 'mean',
        'Performance_Change': 'mean'
    }).round(2)
    
    # Save as CSV
    summary.to_csv('/home/george/projects/personal/RL_basic/ablation_summary.csv')
    
    # Create formatted table for the paper
    with open('/home/george/projects/personal/RL_basic/ablation_summary_table.md', 'w') as f:
        f.write("| Choquet Integral | Hyperbolic Discounting | Final Performance | Performance Change (%) |\n")
        f.write("|------------------|------------------------|-------------------|------------------------|\n")
        
        for idx, row in summary.iterrows():
            choquet, hyperbolic = idx
            choquet_str = "✓" if choquet else "✗"
            hyperbolic_str = "✓" if hyperbolic else "✗"
            f.write(f"| {choquet_str} | {hyperbolic_str} | ")
            f.write(f"{row['Final_Performance']:.1f} | {row['Performance_Change']:.1f}% |\n")
    
    return summary

if __name__ == "__main__":
    print("Analyzing ablation study experiments...")
    
    # Extract all experimental data
    all_data = analyze_ablation_experiments()
    
    if not all_data:
        print("No experimental data found!")
        exit(1)
    
    print(f"Found {len(all_data)} experiments to analyze")
    
    # Generate analysis plots
    print("Creating ablation performance comparison...")
    final_performance = create_ablation_performance_plot(all_data)
    
    print("Creating ablation effects analysis...")
    create_ablation_effects_analysis(all_data)
    
    print("Creating contribution matrix...")
    df = create_ablation_contribution_matrix(final_performance)
    
    print("Creating summary table...")
    summary = create_ablation_summary_table(df)
    
    print("Ablation analysis complete! Generated files:")
    print("- ablation_performance_comparison.png")
    print("- ablation_effects_analysis.png") 
    print("- ablation_contribution_matrix.png")
    print("- ablation_summary.csv")
    print("- ablation_summary_table.md")
    
    print("\nAblation Summary:")
    print(summary)
    
    print("\nIndividual Effects Analysis:")
    print("- Choquet Integral Effect:", df[df['Choquet'] == True]['Performance_Change'].mean() - df[df['Choquet'] == False]['Performance_Change'].mean(), "%")
    print("- Hyperbolic Discounting Effect:", df[df['Hyperbolic'] == True]['Performance_Change'].mean() - df[df['Hyperbolic'] == False]['Performance_Change'].mean(), "%")