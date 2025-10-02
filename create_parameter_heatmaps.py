#!/usr/bin/env python3
"""
Create heatmaps and comprehensive parameter analysis for HalfCheetah experiments
based on actual Prelec distortion parameters and discounting mechanisms.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle

def load_experimental_data():
    """Load experimental parameters and performance data."""
    
    # Load parameter data
    with open('experiment_parameters.json', 'r') as f:
        param_results = json.load(f)
    
    # Load performance data
    with open('halfcheetah_step_normalized_results.json', 'r') as f:
        perf_results = json.load(f)
    
    return param_results, perf_results

def create_parameter_performance_matrix(param_results, perf_results):
    """Create a comprehensive parameter-performance matrix."""
    
    parameter_data = param_results['parameter_data']
    performance_data = perf_results['hyperbolic_analysis']['performance_comparison']
    
    # Build comprehensive data matrix
    experiment_matrix = []
    
    for old_name, params in parameter_data.items():
        if old_name in performance_data:
            row = {
                'experiment': old_name,
                'new_name': param_results['new_names'][old_name],
                'alpha': params['alpha'],
                'eta': params['eta'],
                'gamma': params['gamma'],
                'hyperbolic': params['hyperbolic'],
                'choquet': params['choquet'],
                'distortion_bias': params['distortion_bias'],
                'performance': performance_data[old_name]
            }
            experiment_matrix.append(row)
    
    return pd.DataFrame(experiment_matrix)

def create_hyperbolic_ablation_heatmap(df):
    """Create heatmap for hyperbolic ablation analysis."""
    
    # Separate by discounting mechanism
    hyperbolic_exp = df[df['hyperbolic'] == True]
    standard_exp = df[df['hyperbolic'] == False]
    
    # Create parameter grid for heatmap
    # Use alpha and eta as primary dimensions
    alpha_values = sorted(df['alpha'].unique())
    eta_values = sorted(df['eta'].unique())
    
    # Create matrices for hyperbolic and standard
    hyp_matrix = np.full((len(eta_values), len(alpha_values)), np.nan)
    std_matrix = np.full((len(eta_values), len(alpha_values)), np.nan)
    
    # Fill matrices
    for _, row in hyperbolic_exp.iterrows():
        alpha_idx = alpha_values.index(row['alpha'])
        eta_idx = eta_values.index(row['eta'])
        hyp_matrix[eta_idx, alpha_idx] = row['performance']
    
    for _, row in standard_exp.iterrows():
        alpha_idx = alpha_values.index(row['alpha'])
        eta_idx = eta_values.index(row['eta'])
        std_matrix[eta_idx, alpha_idx] = row['performance']
    
    # Create the plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 1: Standard Discounting Heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(std_matrix, cmap='viridis', aspect='auto', origin='lower')
    ax1.set_title('Standard Discounting (γ=0.99)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Prelec Alpha (α)', fontsize=12)
    ax1.set_ylabel('Prelec Eta (η)', fontsize=12)
    ax1.set_xticks(range(len(alpha_values)))
    ax1.set_xticklabels([f'{a:.1f}' for a in alpha_values])
    ax1.set_yticks(range(len(eta_values)))
    ax1.set_yticklabels([f'{e:.1f}' for e in eta_values])
    
    # Add value annotations
    for i in range(len(eta_values)):
        for j in range(len(alpha_values)):
            if not np.isnan(std_matrix[i, j]):
                ax1.text(j, i, f'{std_matrix[i, j]:.0f}', 
                        ha='center', va='center', fontweight='bold', color='white')
    
    plt.colorbar(im1, ax=ax1, label='Performance')
    
    # Plot 2: Hyperbolic Discounting Heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(hyp_matrix, cmap='viridis', aspect='auto', origin='lower')
    ax2.set_title('Hyperbolic Discounting (γ_eff≈0.969)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Prelec Alpha (α)', fontsize=12)
    ax2.set_ylabel('Prelec Eta (η)', fontsize=12)
    ax2.set_xticks(range(len(alpha_values)))
    ax2.set_xticklabels([f'{a:.1f}' for a in alpha_values])
    ax2.set_yticks(range(len(eta_values)))
    ax2.set_yticklabels([f'{e:.1f}' for e in eta_values])
    
    # Add value annotations
    for i in range(len(eta_values)):
        for j in range(len(alpha_values)):
            if not np.isnan(hyp_matrix[i, j]):
                ax2.text(j, i, f'{hyp_matrix[i, j]:.0f}', 
                        ha='center', va='center', fontweight='bold', color='white')
    
    plt.colorbar(im2, ax=ax2, label='Performance')
    
    # Plot 3: Difference Matrix (Hyperbolic - Standard)
    ax3 = axes[2]
    diff_matrix = hyp_matrix - std_matrix
    
    # Use diverging colormap for differences
    vmax = np.nanmax(np.abs(diff_matrix))
    im3 = ax3.imshow(diff_matrix, cmap='RdBu_r', aspect='auto', origin='lower', 
                     vmin=-vmax, vmax=vmax)
    ax3.set_title('Hyperbolic Effect\n(Hyperbolic - Standard)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Prelec Alpha (α)', fontsize=12)
    ax3.set_ylabel('Prelec Eta (η)', fontsize=12)
    ax3.set_xticks(range(len(alpha_values)))
    ax3.set_xticklabels([f'{a:.1f}' for a in alpha_values])
    ax3.set_yticks(range(len(eta_values)))
    ax3.set_yticklabels([f'{e:.1f}' for e in eta_values])
    
    # Add value annotations
    for i in range(len(eta_values)):
        for j in range(len(alpha_values)):
            if not np.isnan(diff_matrix[i, j]):
                color = 'white' if abs(diff_matrix[i, j]) > vmax/2 else 'black'
                ax3.text(j, i, f'{diff_matrix[i, j]:+.0f}', 
                        ha='center', va='center', fontweight='bold', color=color)
    
    plt.colorbar(im3, ax=ax3, label='Performance Difference')
    
    plt.tight_layout()
    plt.savefig('hyperbolic_ablation_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig, (std_matrix, hyp_matrix, diff_matrix)

def create_prelec_impact_analysis(df):
    """Create comprehensive Prelec parameter impact analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Prelec Parameter Impact Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Alpha vs Performance (by discounting type)
    ax1 = axes[0, 0]
    
    standard_df = df[df['hyperbolic'] == False]
    hyperbolic_df = df[df['hyperbolic'] == True]
    
    # Plot standard experiments
    if not standard_df.empty:
        ax1.scatter(standard_df['alpha'], standard_df['performance'], 
                   c='blue', s=100, alpha=0.7, label='Standard Discounting', edgecolors='black')
        
        # Fit trend line for standard
        if len(standard_df) > 1:
            z = np.polyfit(standard_df['alpha'], standard_df['performance'], 1)
            p = np.poly1d(z)
            alpha_range = np.linspace(standard_df['alpha'].min(), standard_df['alpha'].max(), 100)
            ax1.plot(alpha_range, p(alpha_range), "b--", alpha=0.8)
    
    # Plot hyperbolic experiments
    if not hyperbolic_df.empty:
        ax1.scatter(hyperbolic_df['alpha'], hyperbolic_df['performance'], 
                   c='red', s=100, alpha=0.7, label='Hyperbolic Discounting', edgecolors='black')
        
        # Fit trend line for hyperbolic
        if len(hyperbolic_df) > 1:
            z = np.polyfit(hyperbolic_df['alpha'], hyperbolic_df['performance'], 1)
            p = np.poly1d(z)
            alpha_range = np.linspace(hyperbolic_df['alpha'].min(), hyperbolic_df['alpha'].max(), 100)
            ax1.plot(alpha_range, p(alpha_range), "r--", alpha=0.8)
    
    ax1.set_xlabel('Prelec Alpha (α)')
    ax1.set_ylabel('Performance')
    ax1.set_title('Alpha Parameter Impact')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Eta vs Performance (by discounting type)
    ax2 = axes[0, 1]
    
    if not standard_df.empty:
        ax2.scatter(standard_df['eta'], standard_df['performance'], 
                   c='blue', s=100, alpha=0.7, label='Standard Discounting', edgecolors='black')
    
    if not hyperbolic_df.empty:
        ax2.scatter(hyperbolic_df['eta'], hyperbolic_df['performance'], 
                   c='red', s=100, alpha=0.7, label='Hyperbolic Discounting', edgecolors='black')
    
    ax2.set_xlabel('Prelec Eta (η)')
    ax2.set_ylabel('Performance')
    ax2.set_title('Eta Parameter Impact')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distortion Bias vs Performance
    ax3 = axes[1, 0]
    
    # Filter out None values
    df_with_bias = df.dropna(subset=['distortion_bias'])
    
    if not df_with_bias.empty:
        standard_bias = df_with_bias[df_with_bias['hyperbolic'] == False]
        hyperbolic_bias = df_with_bias[df_with_bias['hyperbolic'] == True]
        
        if not standard_bias.empty:
            ax3.scatter(standard_bias['distortion_bias'], standard_bias['performance'], 
                       c='blue', s=100, alpha=0.7, label='Standard Discounting', edgecolors='black')
        
        if not hyperbolic_bias.empty:
            ax3.scatter(hyperbolic_bias['distortion_bias'], hyperbolic_bias['performance'], 
                       c='red', s=100, alpha=0.7, label='Hyperbolic Discounting', edgecolors='black')
    
    ax3.set_xlabel('Distortion Bias')
    ax3.set_ylabel('Performance')
    ax3.set_title('Distortion Bias Impact')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 4: Parameter Combination Effects
    ax4 = axes[1, 1]
    
    # Create alpha*eta interaction term
    df['alpha_eta_product'] = df['alpha'] * df['eta']
    standard_df['alpha_eta_product'] = standard_df['alpha'] * standard_df['eta']
    hyperbolic_df['alpha_eta_product'] = hyperbolic_df['alpha'] * hyperbolic_df['eta']
    
    if not standard_df.empty:
        ax4.scatter(standard_df['alpha_eta_product'], standard_df['performance'], 
                   c='blue', s=100, alpha=0.7, label='Standard Discounting', edgecolors='black')
    
    if not hyperbolic_df.empty:
        ax4.scatter(hyperbolic_df['alpha_eta_product'], hyperbolic_df['performance'], 
                   c='red', s=100, alpha=0.7, label='Hyperbolic Discounting', edgecolors='black')
    
    ax4.set_xlabel('Alpha × Eta Interaction')
    ax4.set_ylabel('Performance')
    ax4.set_title('Parameter Interaction Effects')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prelec_impact_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def analyze_parameter_correlations(df):
    """Analyze correlations between parameters and performance."""
    
    # Prepare data for correlation analysis
    numeric_columns = ['alpha', 'eta', 'gamma', 'distortion_bias', 'performance']
    correlation_data = df[numeric_columns].dropna()
    
    # Calculate correlation matrix
    corr_matrix = correlation_data.corr()
    
    # Create correlation heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Generate heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    
    ax.set_title('Parameter-Performance Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('parameter_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig, corr_matrix

def main():
    """Main analysis function."""
    
    print("="*80)
    print("PARAMETER-BASED HEATMAP AND IMPACT ANALYSIS")
    print("="*80)
    
    # Load data
    param_results, perf_results = load_experimental_data()
    
    # Create parameter-performance matrix
    df = create_parameter_performance_matrix(param_results, perf_results)
    
    print("\nExperiment Parameter Matrix:")
    print(df[['new_name', 'alpha', 'eta', 'gamma', 'hyperbolic', 'performance']].to_string(index=False))
    
    # Create hyperbolic ablation heatmaps
    print("\nCreating hyperbolic ablation heatmaps...")
    fig1, matrices = create_hyperbolic_ablation_heatmap(df)
    
    # Create Prelec impact analysis
    print("Creating Prelec parameter impact analysis...")
    fig2 = create_prelec_impact_analysis(df)
    
    # Analyze parameter correlations
    print("Analyzing parameter correlations...")
    fig3, corr_matrix = analyze_parameter_correlations(df)
    
    # Save results
    results = {
        'parameter_matrix': df.to_dict('records'),
        'correlation_matrix': corr_matrix.to_dict(),
        'heatmap_matrices': {
            'standard': matrices[0].tolist(),
            'hyperbolic': matrices[1].tolist(),
            'difference': matrices[2].tolist()
        }
    }
    
    with open('parameter_heatmap_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nAnalysis complete!")
    print(f"Heatmaps saved to: hyperbolic_ablation_heatmaps.png")
    print(f"Impact analysis saved to: prelec_impact_analysis.png") 
    print(f"Correlations saved to: parameter_correlation_matrix.png")
    print(f"Results saved to: parameter_heatmap_results.json")
    
    return df, results

if __name__ == "__main__":
    df, results = main()