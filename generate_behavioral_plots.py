#!/usr/bin/env python3
"""
Behavioral Policy Analysis: Visualizing Policy Characteristics Across Ablation Conditions
Generates plots showing policy behavior, action distributions, and decision-making patterns
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
from pathlib import Path
from scipy import stats

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def extract_tensorboard_data(log_dir):
    """Extract data from tensorboard logs"""
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    data = {}
    scalar_tags = ea.Tags()['scalars']
    for tag in scalar_tags:
        scalar_events = ea.Scalars(tag)
        data[tag] = {
            'steps': [event.step for event in scalar_events],
            'values': [event.value for event in scalar_events]
        }
    
    return data

def load_all_experiment_data():
    """Load data from all ablation experiments"""
    runs_dir = "/home/george/projects/personal/RL_basic/runs"
    examples_runs_dir = "/home/george/projects/personal/RL_basic/examples/runs"
    
    experiments = {
        'Standard SAC': {
            'pattern': 'HalfCheetah-v4__standard_sac_halfcheetah*',
            'choquet': False,
            'hyperbolic': False,
            'color': '#d62728',
            'label': 'Standard SAC'
        },
        'Choquet + Hyperbolic': {
            'pattern': 'HalfCheetah-v4__choquet_hyperbolic_ablation*',
            'choquet': True,
            'hyperbolic': True,
            'color': '#2ca02c',
            'label': 'Choquet + Hyperbolic'
        },
        'Standard + Hyperbolic': {
            'pattern': 'HalfCheetah-v4__standard_hyperbolic_ablation*',
            'choquet': False,
            'hyperbolic': True,
            'color': '#ff7f0e',
            'label': 'Standard + Hyperbolic'
        },
        'Choquet + Standard': {
            'pattern': 'HalfCheetah-v4__choquet_standard_ablation*',
            'choquet': True,
            'hyperbolic': False,
            'color': '#1f77b4',
            'label': 'Choquet + Standard'
        }
    }
    
    all_data = {}
    
    for exp_name, config in experiments.items():
        matching_dirs = glob.glob(os.path.join(runs_dir, config['pattern']))
        if not matching_dirs:
            matching_dirs = glob.glob(os.path.join(examples_runs_dir, config['pattern']))
        
        if matching_dirs:
            log_dir = matching_dirs[0]
            print(f"Loading {exp_name}: {log_dir}")
            try:
                data = extract_tensorboard_data(log_dir)
                all_data[exp_name] = {
                    'data': data,
                    'config': config
                }
            except Exception as e:
                print(f"Error loading {exp_name}: {e}")
    
    return all_data

def create_policy_entropy_analysis(all_data):
    """Analyze policy entropy and exploration characteristics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Alpha (entropy coefficient) evolution
    for exp_name, exp_data in all_data.items():
        if 'losses/alpha' in exp_data['data']:
            steps = exp_data['data']['losses/alpha']['steps']
            alpha_values = exp_data['data']['losses/alpha']['values']
            color = exp_data['config']['color']
            label = exp_data['config']['label']
            
            # Limit to 200k steps for fair comparison
            mask = np.array(steps) <= 200000
            if np.any(mask):
                steps_filtered = np.array(steps)[mask]
                alpha_filtered = np.array(alpha_values)[mask]
                
                if len(alpha_filtered) > 10:
                    window = min(50, len(alpha_filtered) // 10)
                    smoothed = pd.Series(alpha_filtered).rolling(window=window, center=True).mean()
                    ax1.plot(steps_filtered, smoothed, label=label, color=color, linewidth=2.5)
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Entropy Coefficient (α)')
    ax1.set_title('Policy Entropy Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 200000)
    
    # Plot 2: Q-value distributions (final values)
    q_stats = {}
    for exp_name, exp_data in all_data.items():
        if 'losses/qf1_values' in exp_data['data']:
            q_values = exp_data['data']['losses/qf1_values']['values']
            steps = exp_data['data']['losses/qf1_values']['steps']
            
            # Take final 20% of training for analysis
            mask = np.array(steps) >= max(steps) * 0.8
            if np.any(mask):
                final_q_values = np.array(q_values)[mask]
                q_stats[exp_data['config']['label']] = {
                    'mean': np.mean(final_q_values),
                    'std': np.std(final_q_values),
                    'color': exp_data['config']['color']
                }
    
    if q_stats:
        labels = list(q_stats.keys())
        means = [q_stats[label]['mean'] for label in labels]
        stds = [q_stats[label]['std'] for label in labels]
        colors = [q_stats[label]['color'] for label in labels]
        
        bars = ax2.bar(range(len(labels)), means, yerr=stds, 
                       color=colors, alpha=0.7, capsize=5)
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_ylabel('Q-Value Magnitude')
        ax2.set_title('Final Q-Value Distributions')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(stds) * 0.1,
                    f'{mean:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/george/projects/personal/RL_basic/policy_entropy_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_behavioral_decision_patterns(all_data):
    """Analyze behavioral decision-making patterns"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Distortion bias evolution (top-left)
    ax1 = axes[0, 0]
    for exp_name, exp_data in all_data.items():
        if 'behavioral/distortion_bias' in exp_data['data']:
            steps = exp_data['data']['behavioral/distortion_bias']['steps']
            bias_values = exp_data['data']['behavioral/distortion_bias']['values']
            color = exp_data['config']['color']
            label = exp_data['config']['label']
            
            mask = np.array(steps) <= 200000
            if np.any(mask):
                steps_filtered = np.array(steps)[mask]
                bias_filtered = np.array(bias_values)[mask]
                
                if len(bias_filtered) > 10:
                    window = min(30, len(bias_filtered) // 5)
                    smoothed = pd.Series(bias_filtered).rolling(window=window, center=True).mean()
                    ax1.plot(steps_filtered, smoothed, label=label, color=color, linewidth=2.5)
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Distortion Bias')
    ax1.set_title('Risk Perception Bias Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlim(0, 200000)
    
    # Plot 2: Effective gamma comparison (top-right)
    ax2 = axes[0, 1]
    gamma_data = {}
    for exp_name, exp_data in all_data.items():
        if 'charts/effective_gamma' in exp_data['data']:
            gamma_values = exp_data['data']['charts/effective_gamma']['values']
            if len(gamma_values) > 0:
                gamma_data[exp_data['config']['label']] = {
                    'gamma': np.mean(gamma_values),
                    'color': exp_data['config']['color']
                }
    
    if gamma_data:
        labels = list(gamma_data.keys())
        gammas = [gamma_data[label]['gamma'] for label in labels]
        colors = [gamma_data[label]['color'] for label in labels]
        
        bars = ax2.bar(range(len(labels)), gammas, color=colors, alpha=0.7)
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_ylabel('Effective Discount Factor (γ)')
        ax2.set_title('Temporal Discounting Comparison')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.95, 1.0)
        
        # Add value labels
        for bar, gamma in zip(bars, gammas):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{gamma:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Actor loss evolution (bottom-left)
    ax3 = axes[1, 0]
    for exp_name, exp_data in all_data.items():
        if 'losses/actor_loss' in exp_data['data']:
            steps = exp_data['data']['losses/actor_loss']['steps']
            actor_loss = exp_data['data']['losses/actor_loss']['values']
            color = exp_data['config']['color']
            label = exp_data['config']['label']
            
            mask = np.array(steps) <= 200000
            if np.any(mask):
                steps_filtered = np.array(steps)[mask]
                loss_filtered = np.array(actor_loss)[mask]
                
                if len(loss_filtered) > 10:
                    # Take absolute value and smooth
                    abs_loss = np.abs(loss_filtered)
                    window = min(50, len(abs_loss) // 10)
                    smoothed = pd.Series(abs_loss).rolling(window=window, center=True).mean()
                    ax3.plot(steps_filtered, smoothed, label=label, color=color, linewidth=2.5)
    
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Policy Loss Magnitude')
    ax3.set_title('Policy Optimization Dynamics')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 200000)
    ax3.set_yscale('log')
    
    # Plot 4: Reward variance analysis (bottom-right)
    ax4 = axes[1, 1]
    for exp_name, exp_data in all_data.items():
        if 'behavioral/reward_std' in exp_data['data']:
            steps = exp_data['data']['behavioral/reward_std']['steps']
            reward_std = exp_data['data']['behavioral/reward_std']['values']
            color = exp_data['config']['color']
            label = exp_data['config']['label']
            
            mask = np.array(steps) <= 200000
            if np.any(mask):
                steps_filtered = np.array(steps)[mask]
                std_filtered = np.array(reward_std)[mask]
                
                if len(std_filtered) > 10:
                    window = min(30, len(std_filtered) // 5)
                    smoothed = pd.Series(std_filtered).rolling(window=window, center=True).mean()
                    ax4.plot(steps_filtered, smoothed, label=label, color=color, linewidth=2.5)
    
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Reward Standard Deviation')
    ax4.set_title('Reward Variability Patterns')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 200000)
    
    plt.tight_layout()
    plt.savefig('/home/george/projects/personal/RL_basic/behavioral_decision_patterns.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_policy_comparison_radar(all_data):
    """Create radar chart comparing policy characteristics"""
    # Calculate metrics for each experiment
    metrics_data = {}
    
    for exp_name, exp_data in all_data.items():
        metrics = {}
        
        # Performance metric (normalized episodic return)
        if 'charts/episodic_return' in exp_data['data']:
            returns = exp_data['data']['charts/episodic_return']['values']
            steps = exp_data['data']['charts/episodic_return']['steps']
            mask = np.array(steps) <= 200000
            if np.any(mask):
                final_returns = np.array(returns)[mask][-len(returns)//5:]  # Last 20%
                metrics['Performance'] = np.mean(final_returns) / 10000  # Normalize to 0-1 scale
        
        # Stability metric (inverse of reward std)
        if 'behavioral/reward_std' in exp_data['data']:
            reward_stds = exp_data['data']['behavioral/reward_std']['values']
            if len(reward_stds) > 0:
                avg_std = np.mean(reward_stds[-len(reward_stds)//5:])  # Last 20%
                metrics['Stability'] = max(0, 1 - avg_std / 5.0)  # Normalize and invert
        
        # Exploration metric (entropy coefficient)
        if 'losses/alpha' in exp_data['data']:
            alpha_values = exp_data['data']['losses/alpha']['values']
            if len(alpha_values) > 0:
                avg_alpha = np.mean(alpha_values[-len(alpha_values)//5:])  # Last 20%
                metrics['Exploration'] = min(1.0, avg_alpha / 0.5)  # Normalize to 0-1
        
        # Risk sensitivity (absolute distortion bias)
        if 'behavioral/distortion_bias' in exp_data['data']:
            bias_values = exp_data['data']['behavioral/distortion_bias']['values']
            if len(bias_values) > 0:
                avg_bias = np.mean(np.abs(bias_values[-len(bias_values)//5:]))  # Last 20%
                metrics['Risk Sensitivity'] = min(1.0, avg_bias / 2.0)  # Normalize
        
        # Temporal focus (effective gamma)
        if 'charts/effective_gamma' in exp_data['data']:
            gamma_values = exp_data['data']['charts/effective_gamma']['values']
            if len(gamma_values) > 0:
                avg_gamma = np.mean(gamma_values)
                metrics['Long-term Focus'] = avg_gamma  # Already 0-1 scale
        
        # Q-value magnitude (learning efficiency)
        if 'losses/qf1_values' in exp_data['data']:
            q_values = exp_data['data']['losses/qf1_values']['values']
            if len(q_values) > 0:
                avg_q = np.abs(np.mean(q_values[-len(q_values)//5:]))  # Last 20%
                metrics['Value Accuracy'] = min(1.0, avg_q / 5000)  # Normalize
        
        if metrics:
            metrics_data[exp_data['config']['label']] = {
                'metrics': metrics,
                'color': exp_data['config']['color']
            }
    
    if not metrics_data:
        print("No data available for radar chart")
        return
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    # Get all unique metrics
    all_metrics = set()
    for exp_data in metrics_data.values():
        all_metrics.update(exp_data['metrics'].keys())
    all_metrics = sorted(list(all_metrics))
    
    # Calculate angles for each metric
    angles = np.linspace(0, 2 * np.pi, len(all_metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Plot each experiment
    for exp_name, exp_data in metrics_data.items():
        values = []
        for metric in all_metrics:
            values.append(exp_data['metrics'].get(metric, 0))
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2.5, label=exp_name, 
                color=exp_data['color'], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=exp_data['color'])
    
    # Customize the chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(all_metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    plt.title('Policy Behavioral Characteristics Comparison', 
              fontsize=16, fontweight='bold', pad=30)
    
    plt.tight_layout()
    plt.savefig('/home/george/projects/personal/RL_basic/policy_radar_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_learning_efficiency_analysis(all_data):
    """Analyze learning efficiency and convergence patterns"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Learning rate (episodic return improvement over time)
    for exp_name, exp_data in all_data.items():
        if 'charts/episodic_return' in exp_data['data']:
            steps = exp_data['data']['charts/episodic_return']['steps']
            returns = exp_data['data']['charts/episodic_return']['values']
            color = exp_data['config']['color']
            label = exp_data['config']['label']
            
            mask = np.array(steps) <= 200000
            if np.any(mask):
                steps_filtered = np.array(steps)[mask]
                returns_filtered = np.array(returns)[mask]
                
                if len(returns_filtered) > 20:
                    # Calculate learning rate as derivative of smoothed returns
                    window = min(50, len(returns_filtered) // 10)
                    smoothed = pd.Series(returns_filtered).rolling(window=window, center=True).mean()
                    
                    # Calculate learning rate (change per 1000 steps)
                    learning_rate = np.gradient(smoothed.dropna()) * 1000
                    valid_steps = steps_filtered[window//2:-window//2+1]
                    
                    ax1.plot(valid_steps, learning_rate[:len(valid_steps)], 
                            label=label, color=color, linewidth=2.5, alpha=0.8)
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Learning Rate (Return Change / 1000 steps)')
    ax1.set_title('Learning Efficiency Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlim(0, 200000)
    
    # Plot 2: Sample efficiency comparison (time to reach milestones)
    milestones = [1000, 2000, 3000, 4000, 5000]
    efficiency_data = {}
    
    for exp_name, exp_data in all_data.items():
        if 'charts/episodic_return' in exp_data['data']:
            steps = np.array(exp_data['data']['charts/episodic_return']['steps'])
            returns = np.array(exp_data['data']['charts/episodic_return']['values'])
            
            mask = steps <= 200000
            steps_filtered = steps[mask]
            returns_filtered = returns[mask]
            
            milestone_steps = []
            for milestone in milestones:
                # Find first step where return exceeds milestone
                exceeds = np.where(returns_filtered >= milestone)[0]
                if len(exceeds) > 0:
                    milestone_steps.append(steps_filtered[exceeds[0]])
                else:
                    milestone_steps.append(200000)  # Didn't reach milestone
            
            efficiency_data[exp_data['config']['label']] = {
                'steps': milestone_steps,
                'color': exp_data['config']['color']
            }
    
    if efficiency_data:
        x = np.arange(len(milestones))
        width = 0.2
        
        for i, (exp_name, exp_data) in enumerate(efficiency_data.items()):
            bars = ax2.bar(x + i * width, exp_data['steps'], width, 
                          label=exp_name, color=exp_data['color'], alpha=0.7)
            
            # Add value labels on bars
            for bar, steps in zip(bars, exp_data['steps']):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 5000,
                        f'{steps//1000}k' if steps < 200000 else 'N/A',
                        ha='center', va='bottom', fontsize=9, rotation=45)
        
        ax2.set_xlabel('Performance Milestone (Return)')
        ax2.set_ylabel('Steps to Reach Milestone')
        ax2.set_title('Sample Efficiency Comparison')
        ax2.set_xticks(x + width * 1.5)
        ax2.set_xticklabels([f'{m}' for m in milestones])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/george/projects/personal/RL_basic/learning_efficiency_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating behavioral policy analysis plots...")
    
    # Load all experimental data
    all_data = load_all_experiment_data()
    
    if not all_data:
        print("No experimental data found!")
        exit(1)
    
    print(f"Loaded data from {len(all_data)} experiments")
    
    # Generate behavioral analysis plots
    print("Creating policy entropy analysis...")
    create_policy_entropy_analysis(all_data)
    
    print("Creating behavioral decision patterns...")
    create_behavioral_decision_patterns(all_data)
    
    print("Creating policy radar comparison...")
    create_policy_comparison_radar(all_data)
    
    print("Creating learning efficiency analysis...")
    create_learning_efficiency_analysis(all_data)
    
    print("Behavioral policy analysis complete! Generated files:")
    print("- policy_entropy_analysis.png")
    print("- behavioral_decision_patterns.png") 
    print("- policy_radar_comparison.png")
    print("- learning_efficiency_analysis.png")