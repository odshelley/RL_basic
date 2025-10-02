#!/usr/bin/env python3
"""
Analysis of HalfCheetah-v4 Experiments: Impact of Prelec Distortion Parameters on SAC Performance
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

def analyze_halfcheetah_experiments():
    """Analyze all HalfCheetah-v4 experiments"""
    runs_dir = "/home/george/projects/personal/RL_basic/runs"
    
    # Define experiment configurations
    experiments = {
        'Standard SAC': {
            'pattern': 'HalfCheetah-v4__standard_sac_halfcheetah*',
            'alpha': 1.0,
            'eta': 1.0,
            'use_choquet': False,
            'color': '#1f77b4'
        },
        'Highly Risk-Averse': {
            'pattern': 'HalfCheetah-v4__risk_averse_halfcheetah*',
            'alpha': 0.25,
            'eta': 0.7,
            'use_choquet': True,
            'color': '#ff7f0e'
        },
        'Extremely Risk-Averse': {
            'pattern': 'HalfCheetah-v4__extremely_risk_averse_halfcheetah*',
            'alpha': 0.15,
            'eta': 0.6,
            'use_choquet': True,
            'color': '#d62728'
        },
        'Inverse S-Curve': {
            'pattern': 'HalfCheetah-v4__inverse_s_curve_halfcheetah*',
            'alpha': 0.65,
            'eta': 0.4,
            'use_choquet': True,
            'color': '#2ca02c'
        },
        'Risk-Seeking': {
            'pattern': 'HalfCheetah-v4__risk_seeking_halfcheetah*',
            'alpha': 1.4,
            'eta': 1.2,
            'use_choquet': True,
            'color': '#9467bd'
        }
    }
    
    # Extract data from all experiments
    all_data = {}
    
    for exp_name, config in experiments.items():
        matching_dirs = glob.glob(os.path.join(runs_dir, config['pattern']))
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
    
    return all_data

def create_performance_comparison_plot(all_data):
    """Create episodic return comparison plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
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
                ax1.plot(steps, smoothed, label=exp_name, color=color, linewidth=2)
                ax1.fill_between(steps, 
                               pd.Series(returns).rolling(window=window, center=True).quantile(0.25),
                               pd.Series(returns).rolling(window=window, center=True).quantile(0.75),
                               alpha=0.2, color=color)
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Episodic Return')
    ax1.set_title('Learning Curves: Episodic Returns')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final performance distribution
    final_returns = {}
    for exp_name, exp_data in all_data.items():
        if 'charts/episodic_return' in exp_data['data']:
            returns = exp_data['data']['charts/episodic_return']['values']
            if len(returns) > 10:
                # Take last 20% of episodes for final performance
                final_returns[exp_name] = returns[-len(returns)//5:]
    
    if final_returns:
        box_data = []
        labels = []
        colors = []
        for exp_name, returns in final_returns.items():
            box_data.append(returns)
            labels.append(exp_name)
            colors.append(all_data[exp_name]['config']['color'])
        
        bp = ax2.boxplot(box_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax2.set_ylabel('Final Episodic Return')
    ax2.set_title('Final Performance Distribution')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/george/projects/personal/RL_basic/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_behavioral_metrics_analysis(all_data):
    """Analyze behavioral metrics across experiments"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    behavioral_metrics = [
        'behavioral/distortion_bias',
        'behavioral/reward_mean', 
        'behavioral/reward_std',
        'losses/qf_loss'
    ]
    
    titles = [
        'Distortion Bias Over Training',
        'Mean Reward Over Training',
        'Reward Standard Deviation',
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
    
    plt.tight_layout()
    plt.savefig('/home/george/projects/personal/RL_basic/behavioral_metrics_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_parameter_impact_heatmap(all_data):
    """Create heatmap showing impact of alpha and eta parameters"""
    # Extract final performance metrics
    results = []
    
    for exp_name, exp_data in all_data.items():
        config = exp_data['config']
        
        # Get final performance metrics
        final_return = None
        final_std = None
        final_bias = None
        
        if 'charts/episodic_return' in exp_data['data']:
            returns = exp_data['data']['charts/episodic_return']['values']
            if len(returns) > 10:
                final_return = np.mean(returns[-len(returns)//5:])
        
        if 'behavioral/reward_std' in exp_data['data']:
            stds = exp_data['data']['behavioral/reward_std']['values']
            if len(stds) > 10:
                final_std = np.mean(stds[-len(stds)//5:])
        
        if 'behavioral/distortion_bias' in exp_data['data']:
            biases = exp_data['data']['behavioral/distortion_bias']['values']
            if len(biases) > 10:
                final_bias = np.mean(biases[-len(biases)//5:])
        
        results.append({
            'Experiment': exp_name,
            'Alpha': config['alpha'],
            'Eta': config['eta'],
            'Use_Choquet': config['use_choquet'],
            'Final_Return': final_return,
            'Final_Std': final_std,
            'Final_Bias': final_bias
        })
    
    df = pd.DataFrame(results)
    
    # Create correlation matrix
    numeric_cols = ['Alpha', 'Eta', 'Final_Return', 'Final_Std', 'Final_Bias']
    corr_data = df[numeric_cols].dropna()
    
    if len(corr_data) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Correlation heatmap
        corr_matrix = corr_data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax1)
        ax1.set_title('Parameter Correlation Matrix')
        
        # Scatter plot: Alpha vs Final Return
        choquet_data = df[df['Use_Choquet'] == True]
        standard_data = df[df['Use_Choquet'] == False]
        
        if len(choquet_data) > 0:
            ax2.scatter(choquet_data['Alpha'], choquet_data['Final_Return'], 
                       c=choquet_data['Eta'], cmap='viridis', s=100, alpha=0.7, label='Choquet SAC')
        if len(standard_data) > 0:
            ax2.scatter(standard_data['Alpha'], standard_data['Final_Return'], 
                       c='red', s=100, marker='s', alpha=0.7, label='Standard SAC')
        
        ax2.set_xlabel('Alpha Parameter')
        ax2.set_ylabel('Final Return')
        ax2.set_title('Alpha vs Performance (Color = Eta)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.colorbar(ax2.collections[0], ax=ax2, label='Eta Parameter')
        
        plt.tight_layout()
        plt.savefig('/home/george/projects/personal/RL_basic/parameter_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return df

def create_summary_table(df):
    """Create summary statistics table"""
    summary = df.groupby('Experiment').agg({
        'Alpha': 'first',
        'Eta': 'first', 
        'Final_Return': 'mean',
        'Final_Std': 'mean',
        'Final_Bias': 'mean'
    }).round(3)
    
    # Save as CSV
    summary.to_csv('/home/george/projects/personal/RL_basic/experiment_summary.csv')
    
    # Create a formatted table for the paper
    with open('/home/george/projects/personal/RL_basic/experiment_summary_table.md', 'w') as f:
        f.write("| Experiment | α | η | Final Return | Reward Std | Distortion Bias |\n")
        f.write("|------------|---|---|--------------|------------|----------------|\n")
        
        for idx, row in summary.iterrows():
            f.write(f"| {idx} | {row['Alpha']:.2f} | {row['Eta']:.2f} | ")
            f.write(f"{row['Final_Return']:.1f} | {row['Final_Std']:.3f} | {row['Final_Bias']:.4f} |\n")
    
    return summary

if __name__ == "__main__":
    print("Analyzing HalfCheetah-v4 experiments...")
    
    # Extract all experimental data
    all_data = analyze_halfcheetah_experiments()
    
    if not all_data:
        print("No experimental data found!")
        exit(1)
    
    print(f"Found {len(all_data)} experiments to analyze")
    
    # Generate analysis plots
    print("Creating performance comparison plots...")
    create_performance_comparison_plot(all_data)
    
    print("Creating behavioral metrics analysis...")
    create_behavioral_metrics_analysis(all_data)
    
    print("Creating parameter impact analysis...")
    df = create_parameter_impact_heatmap(all_data)
    
    print("Creating summary table...")
    summary = create_summary_table(df)
    
    print("Analysis complete! Generated files:")
    print("- performance_comparison.png")
    print("- behavioral_metrics_analysis.png") 
    print("- parameter_impact_analysis.png")
    print("- experiment_summary.csv")
    print("- experiment_summary_table.md")
    
    print("\nSummary Statistics:")
    print(summary)