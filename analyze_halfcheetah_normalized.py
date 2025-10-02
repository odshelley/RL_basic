#!/usr/bin/env python3
"""
Analyze the 5 key HalfCheetah experiments with step normalization.
Focus on reward variance and distortion bias tracking.
"""

import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

# Define the 5 key HalfCheetah experiments
KEY_EXPERIMENTS = {
    # Original 5 behavioral experiments (runs/)
    'Standard SAC': 'HalfCheetah-v4__standard_sac_halfcheetah__42__1759257392',
    'Risk Averse': 'HalfCheetah-v4__risk_averse_halfcheetah__42__1759257362', 
    'Extremely Risk Averse': 'HalfCheetah-v4__extremely_risk_averse_halfcheetah__42__1759257418',
    'Risk Seeking': 'HalfCheetah-v4__risk_seeking_halfcheetah__42__1759259232',
    'Inverse S-Curve': 'HalfCheetah-v4__inverse_s_curve_halfcheetah__42__1759259198',
    
    # Ablation experiments (examples/runs/)
    'Choquet + Hyperbolic': 'HalfCheetah-v4__choquet_hyperbolic_ablation__42__1759345253',
    'Choquet + Standard': 'HalfCheetah-v4__choquet_standard_ablation__42__1759345279', 
    'Standard + Hyperbolic': 'HalfCheetah-v4__standard_hyperbolic_ablation__42__1759345266'
}

def load_experiment_data(exp_name, exp_dir_name, base_dirs):
    """Load data from a specific experiment."""
    data = {}
    
    # Find the experiment directory
    tfevents_file = None
    for base_dir in base_dirs:
        pattern = os.path.join(base_dir, exp_dir_name, "events.out.tfevents.*")
        files = glob.glob(pattern)
        if files:
            tfevents_file = files[0]
            break
    
    if not tfevents_file:
        print(f"Warning: Could not find tensorboard file for {exp_name}")
        return None
        
    try:
        ea = EventAccumulator(tfevents_file)
        ea.Reload()
        
        # Load all available metrics
        metrics = {}
        for key in ea.scalars.Keys():
            scalar_data = ea.scalars.Items(key)
            steps = [item.step for item in scalar_data]
            values = [item.value for item in scalar_data]
            metrics[key] = {'steps': steps, 'values': values}
        
        max_step = 0
        if metrics:
            max_step = max(max(m['steps']) for m in metrics.values() if m['steps'])
            
        data = {
            'name': exp_name,
            'directory': exp_dir_name,
            'max_step': max_step,
            'metrics': metrics,
            'file_path': tfevents_file
        }
        
        print(f"Loaded {exp_name}: {max_step} steps, {len(metrics)} metrics")
        return data
        
    except Exception as e:
        print(f"Error loading {exp_name}: {e}")
        return None

def normalize_to_common_timeframe(experiments_data, max_steps=None):
    """Normalize all experiments to a common timeframe."""
    
    # Find minimum steps across all experiments if not specified
    if max_steps is None:
        valid_experiments = [exp for exp in experiments_data.values() if exp is not None]
        if not valid_experiments:
            return {}, 0
        max_steps = min(exp['max_step'] for exp in valid_experiments)
    
    print(f"\nNormalizing all experiments to {max_steps} steps")
    
    normalized_data = {}
    
    for exp_name, exp_data in experiments_data.items():
        if exp_data is None:
            continue
            
        normalized_metrics = {}
        
        for metric_name, metric_data in exp_data['metrics'].items():
            steps = np.array(metric_data['steps'])
            values = np.array(metric_data['values'])
            
            # Filter to common timeframe
            mask = steps <= max_steps
            if np.any(mask):
                normalized_metrics[metric_name] = {
                    'steps': steps[mask],
                    'values': values[mask]
                }
        
        normalized_data[exp_name] = {
            'name': exp_name,
            'max_step': max_steps,
            'metrics': normalized_metrics
        }
    
    return normalized_data, max_steps

def calculate_reward_variance(experiments_data, window_size=1000):
    """Calculate rolling reward variance for each experiment."""
    variance_data = {}
    
    for exp_name, exp_data in experiments_data.items():
        if 'charts/episodic_return' in exp_data['metrics']:
            returns_data = exp_data['metrics']['charts/episodic_return']
            steps = np.array(returns_data['steps'])
            returns = np.array(returns_data['values'])
            
            # Calculate rolling variance
            rolling_variance = []
            rolling_steps = []
            
            for i in range(len(returns)):
                start_idx = max(0, i - window_size)
                window_returns = returns[start_idx:i+1]
                if len(window_returns) > 1:
                    variance = np.var(window_returns)
                    rolling_variance.append(variance)
                    rolling_steps.append(steps[i])
            
            variance_data[exp_name] = {
                'steps': rolling_steps,
                'variance': rolling_variance
            }
    
    return variance_data

def extract_distortion_bias(experiments_data):
    """Extract distortion bias data from experiments."""
    bias_data = {}
    
    for exp_name, exp_data in experiments_data.items():
        if 'behavioral/distortion_bias' in exp_data['metrics']:
            bias_metric = exp_data['metrics']['behavioral/distortion_bias']
            bias_data[exp_name] = {
                'steps': bias_metric['steps'],
                'bias': bias_metric['values']
            }
    
    return bias_data

def analyze_hyperbolic_effects(experiments_data):
    """Analyze the specific effects of hyperbolic discounting."""
    
    # Group experiments by hyperbolic vs standard discounting
    hyperbolic_experiments = []
    standard_experiments = []
    
    for exp_name in experiments_data.keys():
        if 'Hyperbolic' in exp_name or 'hyperbolic' in exp_name:
            hyperbolic_experiments.append(exp_name)
        else:
            standard_experiments.append(exp_name)
    
    print(f"\nHyperbolic experiments: {hyperbolic_experiments}")
    print(f"Standard experiments: {standard_experiments}")
    
    # Analyze performance differences
    analysis = {
        'hyperbolic_experiments': hyperbolic_experiments,
        'standard_experiments': standard_experiments,
        'performance_comparison': {},
        'variance_comparison': {}
    }
    
    # Calculate final performance for each group
    for exp_name, exp_data in experiments_data.items():
        if 'charts/episodic_return' in exp_data['metrics']:
            returns = exp_data['metrics']['charts/episodic_return']['values']
            if len(returns) > 0:
                final_performance = np.mean(returns[-10:])  # Average of last 10 episodes
                analysis['performance_comparison'][exp_name] = final_performance
    
    return analysis

def create_step_normalized_plots(experiments_data, variance_data, bias_data, max_steps):
    """Create comprehensive plots with step normalization."""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    colors = plt.cm.Set3(np.linspace(0, 1, len(experiments_data)))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'HalfCheetah Experiments: Step-Normalized Analysis (0 to {max_steps:,} steps)', fontsize=16, fontweight='bold')
    
    # Plot 1: Learning Curves
    ax1 = axes[0, 0]
    for i, (exp_name, exp_data) in enumerate(experiments_data.items()):
        if 'charts/episodic_return' in exp_data['metrics']:
            returns_data = exp_data['metrics']['charts/episodic_return']
            steps = np.array(returns_data['steps'])
            returns = np.array(returns_data['values'])
            
            # Smooth the curves
            window = min(50, len(returns) // 10)
            if window > 1:
                smooth_returns = pd.Series(returns).rolling(window=window, center=True).mean()
                ax1.plot(steps, smooth_returns, label=exp_name, color=colors[i], linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Episodic Return')
    ax1.set_title('Learning Curves (Step-Normalized)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max_steps)
    
    # Plot 2: Reward Variance Evolution
    ax2 = axes[0, 1]
    for i, (exp_name, var_data) in enumerate(variance_data.items()):
        if len(var_data['variance']) > 0:
            # Smooth variance data
            smooth_variance = pd.Series(var_data['variance']).rolling(window=10, center=True).mean()
            ax2.plot(var_data['steps'], smooth_variance, label=exp_name, color=colors[i], linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Reward Variance (Rolling)')
    ax2.set_title('Reward Variance Evolution')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_steps)
    
    # Plot 3: Distortion Bias Evolution
    ax3 = axes[1, 0]
    for i, (exp_name, bias_data_exp) in enumerate(bias_data.items()):
        if len(bias_data_exp['bias']) > 0:
            ax3.plot(bias_data_exp['steps'], bias_data_exp['bias'], 
                    label=exp_name, color=colors[i], linewidth=2, alpha=0.8)
    
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Distortion Bias')
    ax3.set_title('Behavioral Distortion Bias Evolution')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlim(0, max_steps)
    
    # Plot 4: Final Performance Comparison
    ax4 = axes[1, 1]
    final_performances = []
    exp_names = []
    
    for exp_name, exp_data in experiments_data.items():
        if 'charts/episodic_return' in exp_data['metrics']:
            returns = exp_data['metrics']['charts/episodic_return']['values']
            if len(returns) > 0:
                final_perf = np.mean(returns[-20:])  # Average of last 20 episodes
                final_performances.append(final_perf)
                exp_names.append(exp_name)
    
    bars = ax4.bar(range(len(final_performances)), final_performances, color=colors[:len(final_performances)])
    ax4.set_xlabel('Experiment')
    ax4.set_ylabel('Final Performance (Average Return)')
    ax4.set_title('Final Performance Comparison')
    ax4.set_xticks(range(len(exp_names)))
    ax4.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in exp_names], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, final_performances):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('halfcheetah_step_normalized_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """Main analysis function."""
    print("="*80)
    print("HALFCHEETAH STEP-NORMALIZED ANALYSIS")
    print("="*80)
    
    # Define base directories to search
    base_dirs = [
        "/home/george/projects/personal/RL_basic/runs",
        "/home/george/projects/personal/RL_basic/examples/runs"
    ]
    
    # Load all experiments
    print("\nLoading experiments...")
    experiments_data = {}
    
    for exp_name, exp_dir in KEY_EXPERIMENTS.items():
        data = load_experiment_data(exp_name, exp_dir, base_dirs)
        experiments_data[exp_name] = data
    
    # Filter out failed loads
    valid_experiments = {k: v for k, v in experiments_data.items() if v is not None}
    print(f"\nSuccessfully loaded {len(valid_experiments)} out of {len(KEY_EXPERIMENTS)} experiments")
    
    if not valid_experiments:
        print("No valid experiments found!")
        return
    
    # Show step counts before normalization
    print("\nStep counts before normalization:")
    for exp_name, exp_data in valid_experiments.items():
        print(f"  {exp_name:30} | {exp_data['max_step']:8} steps")
    
    # Normalize to common timeframe
    normalized_data, common_steps = normalize_to_common_timeframe(valid_experiments)
    
    print(f"\nNormalized to common timeframe: {common_steps:,} steps")
    
    # Calculate reward variance
    print("\nCalculating reward variance...")
    variance_data = calculate_reward_variance(normalized_data)
    
    # Extract distortion bias
    print("Extracting distortion bias...")
    bias_data = extract_distortion_bias(normalized_data)
    
    # Analyze hyperbolic effects
    print("Analyzing hyperbolic discounting effects...")
    hyperbolic_analysis = analyze_hyperbolic_effects(normalized_data)
    
    # Create visualizations
    print("Creating step-normalized plots...")
    fig = create_step_normalized_plots(normalized_data, variance_data, bias_data, common_steps)
    
    # Save analysis results
    results = {
        'common_steps': common_steps,
        'experiments_analyzed': list(normalized_data.keys()),
        'hyperbolic_analysis': hyperbolic_analysis,
        'variance_data_available': list(variance_data.keys()),
        'bias_data_available': list(bias_data.keys())
    }
    
    with open('halfcheetah_step_normalized_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to: halfcheetah_step_normalized_results.json")
    print(f"Plots saved to: halfcheetah_step_normalized_analysis.png")
    
    return results, normalized_data, variance_data, bias_data

if __name__ == "__main__":
    results, normalized_data, variance_data, bias_data = main()