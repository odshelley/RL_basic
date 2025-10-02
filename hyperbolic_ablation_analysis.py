#!/usr/bin/env python3
"""
Focused hyperbolic discounting ablation analysis on HalfCheetah experiments.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def analyze_hyperbolic_effects():
    """Detailed analysis of hyperbolic discounting effects."""
    
    # Load the step-normalized results
    with open('halfcheetah_step_normalized_results.json', 'r') as f:
        results = json.load(f)
    
    performance_data = results['hyperbolic_analysis']['performance_comparison']
    hyperbolic_experiments = results['hyperbolic_analysis']['hyperbolic_experiments']
    standard_experiments = results['hyperbolic_analysis']['standard_experiments']
    
    print("="*80)
    print("HYPERBOLIC DISCOUNTING ABLATION ANALYSIS")
    print("="*80)
    
    # Separate performance by discounting type
    hyperbolic_performance = {exp: performance_data[exp] for exp in hyperbolic_experiments if exp in performance_data}
    standard_performance = {exp: performance_data[exp] for exp in standard_experiments if exp in performance_data}
    
    print(f"\nHyperbolic Discounting Experiments:")
    for exp, perf in hyperbolic_performance.items():
        print(f"  {exp:30} | {perf:8.1f}")
    
    print(f"\nStandard Discounting Experiments:")
    for exp, perf in standard_performance.items():
        print(f"  {exp:30} | {perf:8.1f}")
    
    # Statistical analysis
    hyperbolic_values = list(hyperbolic_performance.values())
    standard_values = list(standard_performance.values())
    
    hyperbolic_mean = np.mean(hyperbolic_values)
    standard_mean = np.mean(standard_values)
    
    hyperbolic_std = np.std(hyperbolic_values)
    standard_std = np.std(standard_values)
    
    print(f"\nStatistical Summary:")
    print(f"Hyperbolic Discounting:")
    print(f"  Mean: {hyperbolic_mean:8.1f} ± {hyperbolic_std:6.1f}")
    print(f"  Range: {min(hyperbolic_values):8.1f} to {max(hyperbolic_values):8.1f}")
    
    print(f"Standard Discounting:")
    print(f"  Mean: {standard_mean:8.1f} ± {standard_std:6.1f}")
    print(f"  Range: {min(standard_values):8.1f} to {max(standard_values):8.1f}")
    
    # Effect size calculation
    effect_difference = hyperbolic_mean - standard_mean
    effect_percentage = (effect_difference / standard_mean) * 100
    
    print(f"\nHyperbolic Effect:")
    print(f"  Absolute difference: {effect_difference:+8.1f}")
    print(f"  Percentage change: {effect_percentage:+6.1f}%")
    
    # Pair-wise comparisons for matched conditions
    paired_analysis = {}
    
    # Find matching pairs (same base algorithm, different discounting)
    for hyp_exp in hyperbolic_experiments:
        if 'Choquet' in hyp_exp:
            # Compare with Choquet + Standard
            matching_std = 'Choquet + Standard'
            if matching_std in standard_performance:
                hyp_perf = hyperbolic_performance[hyp_exp]
                std_perf = standard_performance[matching_std]
                improvement = hyp_perf - std_perf
                improvement_pct = (improvement / std_perf) * 100
                paired_analysis['Choquet'] = {
                    'hyperbolic': hyp_perf,
                    'standard': std_perf,
                    'improvement': improvement,
                    'improvement_pct': improvement_pct
                }
        elif 'Standard' in hyp_exp:
            # Compare with Standard SAC
            matching_std = 'Standard SAC'
            if matching_std in standard_performance:
                hyp_perf = hyperbolic_performance[hyp_exp]
                std_perf = standard_performance[matching_std]
                improvement = hyp_perf - std_perf
                improvement_pct = (improvement / std_perf) * 100
                paired_analysis['Standard'] = {
                    'hyperbolic': hyp_perf,
                    'standard': std_perf,
                    'improvement': improvement,
                    'improvement_pct': improvement_pct
                }
    
    print(f"\nPaired Comparisons (Hyperbolic vs Standard Discounting):")
    for base_algorithm, comparison in paired_analysis.items():
        print(f"  {base_algorithm} Algorithm:")
        print(f"    Hyperbolic: {comparison['hyperbolic']:8.1f}")
        print(f"    Standard:   {comparison['standard']:8.1f}")
        print(f"    Effect:     {comparison['improvement']:+8.1f} ({comparison['improvement_pct']:+6.1f}%)")
    
    return {
        'hyperbolic_performance': hyperbolic_performance,
        'standard_performance': standard_performance,
        'hyperbolic_mean': hyperbolic_mean,
        'standard_mean': standard_mean,
        'effect_difference': effect_difference,
        'effect_percentage': effect_percentage,
        'paired_analysis': paired_analysis
    }

def create_hyperbolic_analysis_plots(analysis_results):
    """Create focused plots for hyperbolic discounting analysis."""
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Hyperbolic Discounting Ablation Analysis - HalfCheetah (99,999 steps)', fontsize=16, fontweight='bold')
    
    # Plot 1: Performance Comparison by Discounting Type
    ax1 = axes[0, 0]
    
    hyperbolic_values = list(analysis_results['hyperbolic_performance'].values())
    standard_values = list(analysis_results['standard_performance'].values())
    
    positions = [1, 2]
    box_data = [standard_values, hyperbolic_values]
    box_labels = ['Standard\nDiscounting', 'Hyperbolic\nDiscounting']
    
    bp = ax1.boxplot(box_data, positions=positions, labels=box_labels, patch_artist=True)
    bp['boxes'][0].set(facecolor='lightblue', alpha=0.7)
    bp['boxes'][1].set(facecolor='lightcoral', alpha=0.7)
    
    # Add individual points
    for i, data in enumerate(box_data):
        x = positions[i]
        y = data
        ax1.scatter([x] * len(y), y, alpha=0.8, s=50, color='darkblue' if i == 0 else 'darkred')
    
    # Add mean lines
    ax1.axhline(y=analysis_results['standard_mean'], color='blue', linestyle='--', alpha=0.7, 
               label=f'Standard Mean: {analysis_results["standard_mean"]:.1f}')
    ax1.axhline(y=analysis_results['hyperbolic_mean'], color='red', linestyle='--', alpha=0.7,
               label=f'Hyperbolic Mean: {analysis_results["hyperbolic_mean"]:.1f}')
    
    ax1.set_ylabel('Final Performance (Episodic Return)')
    ax1.set_title('Performance Distribution by Discounting Type')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Paired Comparisons
    ax2 = axes[0, 1]
    
    paired_data = analysis_results['paired_analysis']
    algorithms = list(paired_data.keys())
    hyperbolic_values = [paired_data[alg]['hyperbolic'] for alg in algorithms]
    standard_values = [paired_data[alg]['standard'] for alg in algorithms]
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, standard_values, width, label='Standard Discounting', color='lightblue', alpha=0.7)
    bars2 = ax2.bar(x + width/2, hyperbolic_values, width, label='Hyperbolic Discounting', color='lightcoral', alpha=0.7)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_ylabel('Final Performance (Episodic Return)')
    ax2.set_title('Paired Algorithm Comparisons')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Effect Sizes
    ax3 = axes[1, 0]
    
    improvements = [paired_data[alg]['improvement'] for alg in algorithms]
    improvement_pcts = [paired_data[alg]['improvement_pct'] for alg in algorithms]
    
    bars = ax3.bar(algorithms, improvement_pcts, color=['green' if x > 0 else 'red' for x in improvement_pcts], alpha=0.7)
    
    # Add value labels
    for bar, improvement in zip(bars, improvements):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -0.5),
                f'{improvement:+.0f}\n({height:+.1f}%)', ha='center', 
                va='bottom' if height > 0 else 'top', fontweight='bold')
    
    ax3.set_ylabel('Performance Improvement (%)')
    ax3.set_title('Hyperbolic Discounting Effect Size')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: All Experiments Ranked
    ax4 = axes[1, 1]
    
    all_experiments = list(analysis_results['hyperbolic_performance'].keys()) + list(analysis_results['standard_performance'].keys())
    all_performances = list(analysis_results['hyperbolic_performance'].values()) + list(analysis_results['standard_performance'].values())
    
    # Sort by performance
    sorted_data = sorted(zip(all_experiments, all_performances), key=lambda x: x[1], reverse=True)
    sorted_names, sorted_perfs = zip(*sorted_data)
    
    # Color code by discounting type
    colors = ['lightcoral' if exp in analysis_results['hyperbolic_performance'] else 'lightblue' for exp in sorted_names]
    
    bars = ax4.barh(range(len(sorted_names)), sorted_perfs, color=colors, alpha=0.7)
    
    ax4.set_yticks(range(len(sorted_names)))
    ax4.set_yticklabels([name.replace(' ', '\n') for name in sorted_names])
    ax4.set_xlabel('Final Performance (Episodic Return)')
    ax4.set_title('All Experiments Ranked by Performance')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, perf) in enumerate(zip(bars, sorted_perfs)):
        width = bar.get_width()
        ax4.text(width + 50, bar.get_y() + bar.get_height()/2.,
                f'{perf:.0f}', ha='left', va='center', fontweight='bold')
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='lightcoral', alpha=0.7, label='Hyperbolic Discounting'),
                      Patch(facecolor='lightblue', alpha=0.7, label='Standard Discounting')]
    ax4.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('hyperbolic_ablation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """Main analysis function."""
    
    print("Running focused hyperbolic discounting ablation analysis...")
    
    # Perform analysis
    analysis_results = analyze_hyperbolic_effects()
    
    # Create plots
    fig = create_hyperbolic_analysis_plots(analysis_results)
    
    # Save detailed results
    with open('hyperbolic_ablation_detailed_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\nDetailed analysis complete!")
    print(f"Results saved to: hyperbolic_ablation_detailed_results.json")
    print(f"Plots saved to: hyperbolic_ablation_analysis.png")
    
    return analysis_results

if __name__ == "__main__":
    analysis_results = main()