"""
Comparison script for Q-learning (tabular) vs REINFORCE (function approximation).

This script runs both algorithms on the same gridworld environment
and compares their performance, learning curves, and final policies.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.qlearning import QLearning
from algorithms.reinforce import REINFORCEAgent, REINFORCEConfig
from algorithms.gymnasium_wrapper import make_gridworld_env
from gridworld.environment import StochasticGridworld


def evaluate_qlearning_agent(agent: QLearning, num_episodes: int = 100) -> Dict[str, float]:
    """
    Evaluate Q-learning agent performance.
    
    Args:
        agent: Trained Q-learning agent
        num_episodes: Number of evaluation episodes
        
    Returns:
        Evaluation statistics
    """
    print(f"Evaluating Q-learning agent for {num_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(num_episodes):
        state = agent.env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            # Use greedy policy (no exploration)
            action = agent.get_greedy_action(state)
            next_state, reward, done, info = agent.env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Check for success
            if next_state == agent.env.goal_state:
                success_count += 1
                break
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "success_rate": success_count / num_episodes,
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards)
    }
    
    print("Q-learning Evaluation Results:")
    print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Mean Length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
    print(f"  Success Rate: {stats['success_rate']:.1%}")
    print(f"  Reward Range: [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
    
    return stats


def train_qlearning_baseline(grid_size: int = 5, num_episodes: int = 2000) -> Tuple[QLearning, Dict]:
    """
    Train Q-learning agent as baseline.
    
    Args:
        grid_size: Size of the gridworld
        num_episodes: Number of training episodes
        
    Returns:
        Trained agent and training statistics
    """
    print("Training Q-learning baseline...")
    
    # Create environment and agent
    env = StochasticGridworld(size=grid_size, gamma=0.99, max_steps=200)
    agent = QLearning(
        env=env,
        alpha=0.1,
        epsilon=0.1,
        epsilon_decay=0.995,
        min_epsilon=0.01
    )
    
    # Training
    start_time = time.time()
    training_stats = agent.run(num_episodes=num_episodes, verbose=False)
    training_time = time.time() - start_time
    
    print(f"Q-learning training completed in {training_time:.1f} seconds")
    
    # Evaluation
    eval_stats = evaluate_qlearning_agent(agent, num_episodes=100)
    
    return agent, {
        "training_stats": training_stats,
        "eval_stats": eval_stats,
        "training_time": training_time
    }


def train_reinforce_agent(grid_size: int = 5, num_episodes: int = 2000, 
                         observation_type: str = "coordinates") -> Tuple[REINFORCEAgent, Dict]:
    """
    Train REINFORCE agent.
    
    Args:
        grid_size: Size of the gridworld
        num_episodes: Number of training episodes
        observation_type: Type of observation representation
        
    Returns:
        Trained agent and training statistics
    """
    print("Training REINFORCE agent...")
    
    # Configuration
    config = REINFORCEConfig(
        hidden_dims=(64, 32),
        learning_rate=0.001,
        gamma=0.99,
        grid_size=grid_size,
        max_steps_per_episode=200,
        observation_type=observation_type,
        num_episodes=num_episodes,
        log_interval=200,
        save_interval=500,
        log_dir=f"logs/comparison_reinforce_{observation_type}",
        gradient_clip=1.0  # Add gradient clipping for stability
    )
    
    # Create and train agent
    agent = REINFORCEAgent(config)
    
    start_time = time.time()
    training_stats = agent.train()
    training_time = time.time() - start_time
    
    print(f"REINFORCE training completed in {training_time:.1f} seconds")
    
    # Evaluation
    eval_stats = agent.evaluate(num_episodes=100, render=False)
    
    return agent, {
        "training_stats": training_stats,
        "eval_stats": eval_stats,
        "training_time": training_time
    }


def compare_learning_curves(qlearning_stats: Dict, reinforce_stats: Dict, 
                          save_path: str = "comparison_plots.png"):
    """
    Plot and compare learning curves of both algorithms.
    
    Args:
        qlearning_stats: Q-learning training statistics
        reinforce_stats: REINFORCE training statistics
        save_path: Path to save the comparison plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Q-learning vs REINFORCE Comparison", fontsize=16)
    
    # Episode rewards comparison
    q_rewards = qlearning_stats["training_stats"].get("episode_rewards", qlearning_stats["training_stats"].get("episode_returns", []))
    r_rewards = reinforce_stats["training_stats"].get("episode_rewards", reinforce_stats["training_stats"].get("episode_returns", []))
    
    axes[0, 0].plot(q_rewards, alpha=0.7, color='blue', label='Q-learning')
    axes[0, 0].plot(r_rewards, alpha=0.7, color='red', label='REINFORCE')
    axes[0, 0].set_title("Episode Rewards")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode lengths comparison
    q_lengths = qlearning_stats["training_stats"].get("episode_lengths", [])
    r_lengths = reinforce_stats["training_stats"].get("episode_lengths", [])
    
    axes[0, 1].plot(q_lengths, alpha=0.7, color='blue', label='Q-learning')
    axes[0, 1].plot(r_lengths, alpha=0.7, color='red', label='REINFORCE')
    axes[0, 1].set_title("Episode Lengths")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Steps")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Moving average rewards (smoothed)
    window_size = 100
    if len(q_rewards) >= window_size:
        q_smooth = np.convolve(q_rewards, np.ones(window_size)/window_size, mode='valid')
        axes[1, 0].plot(range(window_size-1, len(q_rewards)), 
                       q_smooth, color='blue', linewidth=2, label='Q-learning')
    
    if len(r_rewards) >= window_size:
        r_smooth = np.convolve(r_rewards, np.ones(window_size)/window_size, mode='valid')
        axes[1, 0].plot(range(window_size-1, len(r_rewards)), 
                       r_smooth, color='red', linewidth=2, label='REINFORCE')
    
    axes[1, 0].set_title("Smoothed Rewards (Moving Average)")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Reward")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Evaluation comparison (bar chart)
    methods = ['Q-learning', 'REINFORCE']
    mean_rewards = [qlearning_stats["eval_stats"]["mean_reward"],
                   reinforce_stats["eval_stats"]["mean_reward"]]
    success_rates = [qlearning_stats["eval_stats"]["success_rate"],
                    reinforce_stats["eval_stats"]["success_rate"]]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = axes[1, 1].bar(x - width/2, mean_rewards, width, label='Mean Reward', color=['blue', 'red'], alpha=0.7)
    ax2 = axes[1, 1].twinx()
    bars2 = ax2.bar(x + width/2, success_rates, width, label='Success Rate', color=['lightblue', 'lightcoral'], alpha=0.7)
    
    axes[1, 1].set_title("Final Performance Comparison")
    axes[1, 1].set_xlabel("Algorithm")
    axes[1, 1].set_ylabel("Mean Reward")
    ax2.set_ylabel("Success Rate")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(methods)
    axes[1, 1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plots saved to {save_path}")
    plt.show()


def print_comparison_summary(qlearning_results: Dict, reinforce_results: Dict):
    """
    Print a summary comparison of both algorithms.
    
    Args:
        qlearning_results: Q-learning results
        reinforce_results: REINFORCE results
    """
    print("\n" + "="*60)
    print("ALGORITHM COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\n{'Metric':<25} {'Q-learning':<15} {'REINFORCE':<15}")
    print("-" * 55)
    
    # Performance metrics
    q_eval = qlearning_results["eval_stats"]
    r_eval = reinforce_results["eval_stats"]
    
    print(f"{'Mean Reward':<25} {q_eval['mean_reward']:<15.2f} {r_eval['mean_reward']:<15.2f}")
    print(f"{'Success Rate':<25} {q_eval['success_rate']:<15.1%} {r_eval['success_rate']:<15.1%}")
    print(f"{'Mean Episode Length':<25} {q_eval['mean_length']:<15.1f} {r_eval['mean_length']:<15.1f}")
    print(f"{'Training Time (s)':<25} {qlearning_results['training_time']:<15.1f} {reinforce_results['training_time']:<15.1f}")
    
    print("\n" + "="*60)
    
    # Determine winner
    print("ANALYSIS:")
    if q_eval['success_rate'] > r_eval['success_rate']:
        print("🏆 Q-learning achieves higher success rate")
    elif r_eval['success_rate'] > q_eval['success_rate']:
        print("🏆 REINFORCE achieves higher success rate")
    else:
        print("🤝 Both algorithms achieve similar success rates")
    
    if q_eval['mean_reward'] > r_eval['mean_reward']:
        print("💰 Q-learning achieves higher mean reward")
    elif r_eval['mean_reward'] > q_eval['mean_reward']:
        print("💰 REINFORCE achieves higher mean reward")
    else:
        print("💰 Both algorithms achieve similar mean rewards")
    
    if qlearning_results['training_time'] < reinforce_results['training_time']:
        print("⚡ Q-learning trains faster")
    else:
        print("⚡ REINFORCE trains faster")
    
    print("\n" + "="*60)


def main():
    """Main comparison function."""
    print("Starting Q-learning vs REINFORCE Comparison")
    print("="*50)
    
    # Parameters
    grid_size = 5
    num_episodes = 2000
    observation_type = "coordinates"  # Can try "one_hot" or "grid" too
    
    # Train Q-learning baseline
    qlearning_agent, qlearning_results = train_qlearning_baseline(
        grid_size=grid_size, 
        num_episodes=num_episodes
    )
    
    print("\n" + "="*50)
    
    # Train REINFORCE agent
    reinforce_agent, reinforce_results = train_reinforce_agent(
        grid_size=grid_size,
        num_episodes=num_episodes,
        observation_type=observation_type
    )
    
    print("\n" + "="*50)
    
    # Compare and visualize results
    compare_learning_curves(
        qlearning_results, 
        reinforce_results, 
        save_path=f"comparison_plots_{observation_type}.png"
    )
    
    # Print summary
    print_comparison_summary(qlearning_results, reinforce_results)
    
    return qlearning_agent, reinforce_agent, qlearning_results, reinforce_results


if __name__ == "__main__":
    qlearning_agent, reinforce_agent, qlearning_results, reinforce_results = main()
