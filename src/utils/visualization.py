"""
Visualization utilities for the Stochastic Gridworld and RL algorithms.

This module provides functions to visualize:
- The gridworld environment
- Value functions
- Policies
- Learning curves
- Algorithm comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional
import sys
import os

# Add the src directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gridworld.environment import StochasticGridworld, Action


def plot_gridworld(env: StochasticGridworld, title: str = "Stochastic Gridworld", 
                   figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
    """
    Plot the gridworld environment layout.
    
    Args:
        env: The gridworld environment
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create grid
    grid = np.zeros((env.size, env.size))
    
    # Color coding: 0=normal, 1=start, 2=goal, -1=pit
    grid[env.start_state] = 1
    grid[env.goal_state] = 2
    for pit in env.pit_states:
        grid[pit] = -1
    
    # Create custom colormap
    colors = ['red', 'white', 'lightblue', 'green']  # pit, normal, start, goal
    n_bins = 4
    cmap = plt.cm.colors.ListedColormap(colors[:n_bins])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    im = ax.imshow(grid, cmap=cmap, norm=norm)
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    
    # Add labels
    for i in range(env.size):
        for j in range(env.size):
            if (i, j) == env.start_state:
                ax.text(j, i, 'S', ha='center', va='center', fontsize=16, fontweight='bold')
            elif (i, j) == env.goal_state:
                ax.text(j, i, 'G', ha='center', va='center', fontsize=16, fontweight='bold')
            elif (i, j) in env.pit_states:
                ax.text(j, i, 'P', ha='center', va='center', fontsize=16, fontweight='bold', color='white')
            else:
                ax.text(j, i, f'({i},{j})', ha='center', va='center', fontsize=8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Column (j)')
    ax.set_ylabel('Row (i)')
    
    # Remove ticks
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    
    plt.tight_layout()
    return fig


def plot_value_function(env: StochasticGridworld, value_function: Dict[Tuple[int, int], float],
                       title: str = "Value Function", figsize: Tuple[int, int] = (10, 8),
                       show_values: bool = True) -> plt.Figure:
    """
    Plot the value function as a heatmap.
    
    Args:
        env: The gridworld environment
        value_function: Dictionary mapping states to values
        title: Plot title
        figsize: Figure size
        show_values: Whether to show numerical values in cells
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create value matrix
    value_matrix = np.zeros((env.size, env.size))
    for state in env.get_all_states():
        value_matrix[state] = value_function.get(state, 0.0)
    
    # Plot heatmap
    im = ax.imshow(value_matrix, cmap='RdYlBu_r', aspect='equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value', rotation=270, labelpad=20)
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    
    # Add value labels if requested
    if show_values:
        for i in range(env.size):
            for j in range(env.size):
                value = value_function.get((i, j), 0.0)
                color = 'white' if abs(value) > 0.3 else 'black'
                ax.text(j, i, f'{value:.3f}', ha='center', va='center', 
                       color=color, fontweight='bold', fontsize=10)
    
    # Mark special states
    ax.plot(env.start_state[1], env.start_state[0], 'go', markersize=15, markeredgecolor='black', markeredgewidth=2)
    ax.plot(env.goal_state[1], env.goal_state[0], 'g^', markersize=15, markeredgecolor='black', markeredgewidth=2)
    for pit in env.pit_states:
        ax.plot(pit[1], pit[0], 'rx', markersize=15, markeredgecolor='black', markeredgewidth=2)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Column (j)')
    ax.set_ylabel('Row (i)')
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    
    plt.tight_layout()
    return fig


def plot_policy(env: StochasticGridworld, policy: Dict[Tuple[int, int], Action],
               value_function: Optional[Dict[Tuple[int, int], float]] = None,
               title: str = "Policy", figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot the policy as arrows on the grid.
    
    Args:
        env: The gridworld environment
        policy: Dictionary mapping states to actions
        value_function: Optional value function for background coloring
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # If value function provided, use as background
    if value_function is not None:
        value_matrix = np.zeros((env.size, env.size))
        for state in env.get_all_states():
            value_matrix[state] = value_function.get(state, 0.0)
        
        im = ax.imshow(value_matrix, cmap='RdYlBu_r', alpha=0.3, aspect='equal')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Value', rotation=270, labelpad=20)
    
    # Arrow directions for each action
    arrow_directions = {
        Action.UP: (0, -0.3),
        Action.DOWN: (0, 0.3),
        Action.LEFT: (-0.3, 0),
        Action.RIGHT: (0.3, 0)
    }
    
    # Plot policy arrows
    for state in env.get_all_states():
        if env.is_terminal(state):
            continue
        
        action = policy.get(state)
        if action is None:
            continue
        
        i, j = state
        dx, dy = arrow_directions[action]
        
        ax.arrow(j, i, dx, dy, head_width=0.1, head_length=0.1, 
                fc='black', ec='black', linewidth=2)
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    
    # Mark special states
    ax.plot(env.start_state[1], env.start_state[0], 'go', markersize=15, 
           markeredgecolor='black', markeredgewidth=2, label='Start')
    ax.plot(env.goal_state[1], env.goal_state[0], 'g^', markersize=15, 
           markeredgecolor='black', markeredgewidth=2, label='Goal')
    for pit in env.pit_states:
        ax.plot(pit[1], pit[0], 'rx', markersize=15, 
               markeredgecolor='black', markeredgewidth=2)
    
    # Add legend for first pit only
    if env.pit_states:
        ax.plot([], [], 'rx', markersize=15, markeredgecolor='black', 
               markeredgewidth=2, label='Pit')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Column (j)')
    ax.set_ylabel('Row (i)')
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_learning_curve(value_history: List[Dict[Tuple[int, int], float]], 
                       env: StochasticGridworld,
                       title: str = "Value Function Convergence",
                       figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot the learning curve showing how the value function evolves.
    
    Args:
        value_history: List of value functions over time
        env: The gridworld environment
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Value function changes for selected states
    states_to_plot = [env.start_state, env.goal_state, (2, 2), (3, 1)]  # Representative states
    iterations = range(len(value_history))
    
    for state in states_to_plot:
        if state in env.get_all_states():
            values = [vh.get(state, 0.0) for vh in value_history]
            label = f"({state[0]},{state[1]})"
            if state == env.start_state:
                label += " (Start)"
            elif state == env.goal_state:
                label += " (Goal)"
            ax1.plot(iterations, values, marker='o', label=label)
    
    ax1.set_xlabel('Policy Iteration')
    ax1.set_ylabel('Value')
    ax1.set_title('Value Evolution for Selected States')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Maximum value change per iteration
    if len(value_history) > 1:
        max_changes = []
        for i in range(1, len(value_history)):
            max_change = 0
            for state in env.get_all_states():
                change = abs(value_history[i].get(state, 0.0) - value_history[i-1].get(state, 0.0))
                max_change = max(max_change, change)
            max_changes.append(max_change)
        
        ax2.semilogy(range(1, len(value_history)), max_changes, marker='o')
        ax2.set_xlabel('Policy Iteration')
        ax2.set_ylabel('Max Value Change (log scale)')
        ax2.set_title('Convergence Rate')
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_algorithm_comparison(results: Dict[str, Dict], 
                            title: str = "Algorithm Comparison",
                            figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Plot comparison between different algorithms.
    
    Args:
        results: Dictionary with algorithm names as keys and result dictionaries as values
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    algorithms = list(results.keys())
    
    # Plot 1: Convergence iterations
    iterations = [results[alg].get('iterations', 0) for alg in algorithms]
    ax1.bar(algorithms, iterations)
    ax1.set_ylabel('Iterations to Converge')
    ax1.set_title('Convergence Speed')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Final performance metrics
    if 'evaluation' in results[algorithms[0]]:
        mean_returns = [results[alg]['evaluation']['mean_return'] for alg in algorithms]
        success_rates = [results[alg]['evaluation']['success_rate'] for alg in algorithms]
        
        ax2_twin = ax2.twinx()
        
        bars1 = ax2.bar([i - 0.2 for i in range(len(algorithms))], mean_returns, 
                       width=0.4, label='Mean Return', alpha=0.7)
        bars2 = ax2_twin.bar([i + 0.2 for i in range(len(algorithms))], success_rates, 
                           width=0.4, label='Success Rate', alpha=0.7, color='orange')
        
        ax2.set_ylabel('Mean Return')
        ax2_twin.set_ylabel('Success Rate')
        ax2.set_title('Performance Metrics')
        ax2.set_xticks(range(len(algorithms)))
        ax2.set_xticklabels(algorithms, rotation=45)
        
        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Plot 3: Learning curves (if available)
    for alg in algorithms:
        if 'value_history' in results[alg] and results[alg]['value_history']:
            # Plot maximum value change over iterations
            value_history = results[alg]['value_history']
            if len(value_history) > 1:
                max_changes = []
                for i in range(1, len(value_history)):
                    max_change = 0
                    for state_values in value_history[i].values():
                        if i-1 < len(value_history):
                            for state in value_history[i]:
                                if state in value_history[i-1]:
                                    change = abs(value_history[i][state] - value_history[i-1][state])
                                    max_change = max(max_change, change)
                    max_changes.append(max_change)
                
                ax3.semilogy(range(1, len(value_history)), max_changes, 
                           marker='o', label=alg)
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Max Value Change (log scale)')
    ax3.set_title('Convergence Rate Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    stats_data = []
    for alg in algorithms:
        if 'evaluation' in results[alg]:
            eval_data = results[alg]['evaluation']
            stats_data.append([
                eval_data.get('mean_return', 0),
                eval_data.get('success_rate', 0),
                eval_data.get('mean_length', 0)
            ])
    
    if stats_data:
        stats_df = np.array(stats_data).T
        im = ax4.imshow(stats_df, cmap='RdYlBu_r', aspect='auto')
        
        ax4.set_xticks(range(len(algorithms)))
        ax4.set_xticklabels(algorithms, rotation=45)
        ax4.set_yticks(range(3))
        ax4.set_yticklabels(['Mean Return', 'Success Rate', 'Mean Episode Length'])
        ax4.set_title('Performance Heatmap')
        
        # Add text annotations
        for i in range(3):
            for j in range(len(algorithms)):
                text = ax4.text(j, i, f'{stats_df[i, j]:.3f}',
                              ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax4)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig
