"""
Example: Q-Learning (SARSA-max) on 5x5 Stochastic Gridworld

This example demonstrates how to use Q-learning to solve the stochastic gridworld
problem through off-policy temporal difference learning. It includes:
- Environment setup
- Running Q-learning
- Visualizing learning progress
- Policy evaluation
- Comparison with SARSA and model-based methods
"""

import sys
import os

# Add src to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt

from gridworld.environment import StochasticGridworld, Action
from algorithms.qlearning import QLearning
from algorithms.sarsa import SARSA
from algorithms.value_iteration import ValueIteration
from algorithms.policy_iteration import PolicyIteration
from utils.visualization import plot_gridworld, plot_value_function, plot_policy


def plot_learning_curves(qlearning_results, title="Q-Learning Progress"):
    """Plot learning curves for Q-learning."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    episodes = range(len(qlearning_results['episode_returns']))
    
    # Plot 1: Episode returns
    ax1.plot(episodes, qlearning_results['episode_returns'], alpha=0.3, color='blue')
    # Add moving average
    window_size = min(100, len(qlearning_results['episode_returns']) // 10)
    if window_size > 1:
        moving_avg = np.convolve(qlearning_results['episode_returns'], 
                               np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(qlearning_results['episode_returns'])), 
                moving_avg, color='red', linewidth=2, label=f'{window_size}-episode moving average')
        ax1.legend()
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return')
    ax1.set_title('Episode Returns')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Episode lengths
    ax2.plot(episodes, qlearning_results['episode_lengths'], alpha=0.3, color='green')
    if window_size > 1:
        moving_avg_length = np.convolve(qlearning_results['episode_lengths'], 
                                      np.ones(window_size)/window_size, mode='valid')
        ax2.plot(range(window_size-1, len(qlearning_results['episode_lengths'])), 
                moving_avg_length, color='darkgreen', linewidth=2, label=f'{window_size}-episode moving average')
        ax2.legend()
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Episode Lengths')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Epsilon decay
    ax3.plot(episodes, qlearning_results['epsilon_history'], color='purple')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon (Exploration Rate)')
    ax3.set_title('Exploration Rate Decay')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Q-value evolution for key states
    if qlearning_results['q_value_history']:
        ax4.set_xlabel('Training Progress (x100 episodes)')
        ax4.set_ylabel('Max Q-Value')
        ax4.set_title('Q-Value Evolution')
        ax4.grid(True, alpha=0.3)
        
        # Plot Q-value evolution for different states
        colors = ['red', 'blue', 'green', 'orange']
        for i, (state, color) in enumerate(zip([(4, 0), (2, 2), (1, 3), (3, 1)], colors)):
            if state in qlearning_results['q_value_history'][0]:
                max_q_values = []
                for snapshot in qlearning_results['q_value_history']:
                    if state in snapshot:
                        max_q_values.append(max(snapshot[state].values()))
                    else:
                        max_q_values.append(0)
                
                ax4.plot(range(len(max_q_values)), max_q_values, 
                        color=color, label=f'State {state}', marker='o', markersize=3)
        
        ax4.legend()
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=14)
    return fig


def main():
    print("=" * 55)
    print("Q-LEARNING (SARSA-MAX) ON STOCHASTIC GRIDWORLD")
    print("=" * 55)
    
    # Create the environment
    env = StochasticGridworld(size=5, gamma=0.99, max_steps=200)
    
    print("Environment Details:")
    print(f"- Grid size: {env.size}x{env.size}")
    print(f"- Start state: {env.start_state}")
    print(f"- Goal state: {env.goal_state}")
    print(f"- Pit states: {env.pit_states}")
    print(f"- Discount factor: {env.gamma}")
    print(f"- Step reward: {env.step_reward}")
    print(f"- Goal reward: {env.goal_reward}")
    print(f"- Pit reward: {env.pit_reward}")
    print(f"- Transition probabilities: {env.intended_prob:.1f} intended, {env.slip_prob:.1f} each slip")
    print()
    
    print("Initial gridworld layout:")
    print(env.render())
    print()
    
    # Initialize Q-learning
    qlearning = QLearning(env, alpha=0.1, epsilon=0.3, epsilon_decay=0.995, min_epsilon=0.01)
    
    print("Q-Learning Parameters:")
    print(f"- Learning rate (α): {qlearning.alpha}")
    print(f"- Initial exploration (ε): {qlearning.initial_epsilon}")
    print(f"- Epsilon decay: {qlearning.epsilon_decay}")
    print(f"- Min epsilon: {qlearning.min_epsilon}")
    print()
    
    print("Running Q-Learning...")
    results = qlearning.run(num_episodes=2000, verbose=True, save_frequency=100)
    
    print("\nResults Summary:")
    print(f"- Episodes completed: {results['episodes']}")
    print(f"- Final epsilon: {qlearning.epsilon:.6f}")
    
    # Final policy evaluation
    evaluation = results['final_evaluation']
    print("\nFinal Policy Evaluation (Greedy Policy):")
    print(f"- Mean return: {evaluation['mean_return']:.6f} ± {evaluation['std_return']:.6f}")
    print(f"- Mean episode length: {evaluation['mean_length']:.2f} ± {evaluation['std_length']:.2f}")
    print(f"- Success rate: {evaluation['success_rate']:.3f}")
    print()
    
    # Show some example state values and policies
    print("Sample State Values and Q-Values:")
    sample_states = [(4, 0), (2, 2), (1, 3), (3, 1)]
    for state in sample_states:
        value = qlearning.get_state_value(state)
        action = qlearning.get_policy_action(state)
        q_values = qlearning.get_action_values(state)
        
        print(f"State {state}:")
        print(f"  Value: {value:.6f}")
        print(f"  Policy action: {action}")
        if q_values:
            print("  Q-values:")
            for act, val in q_values.items():
                print(f"    {act.name}: {val:.6f}")
        print()
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Plot 1: Environment layout
    fig1 = plot_gridworld(env, title="5x5 Stochastic Gridworld Environment")
    
    # Plot 2: Learning curves
    fig2 = plot_learning_curves(results, title="Q-Learning Progress")
    
    # Plot 3: Final value function (derived from Q-values)
    value_function = {}
    for state in env.get_all_states():
        value_function[state] = qlearning.get_state_value(state)
    
    fig3 = plot_value_function(env, value_function, 
                              title="Learned Value Function (Q-Learning)")
    
    # Plot 4: Final policy
    policy = {}
    for state in env.get_all_states():
        policy[state] = qlearning.get_policy_action(state)
    
    fig4 = plot_policy(env, policy, value_function,
                      title="Learned Policy (Q-Learning)")
    
    # Show all plots
    plt.show()
    
    # Optional: Save results
    print("\nTo save results, uncomment the following lines in the script:")
    print("# import pickle")
    print("# with open('qlearning_results.pkl', 'wb') as f:")
    print("#     pickle.dump(results, f)")
    print("# fig1.savefig('gridworld_environment.png', dpi=300, bbox_inches='tight')")
    print("# fig2.savefig('qlearning_learning_curves.png', dpi=300, bbox_inches='tight')")
    print("# fig3.savefig('value_function_qlearning.png', dpi=300, bbox_inches='tight')")
    print("# fig4.savefig('policy_qlearning.png', dpi=300, bbox_inches='tight')")
    
    return results, qlearning


def demonstrate_policy_execution(qlearning, num_episodes=5):
    """Demonstrate executing the learned policy in the environment."""
    print("\n" + "=" * 50)
    print("POLICY EXECUTION DEMONSTRATION")
    print("=" * 50)
    
    # Run a few episodes with greedy policy
    print(f"Running {num_episodes} episodes with the learned greedy policy:\n")
    
    old_epsilon = qlearning.epsilon
    qlearning.epsilon = 0.0  # Use greedy policy
    
    for episode in range(num_episodes):
        state = qlearning.env.reset()
        episode_return = 0
        episode_length = 0
        path = [state]
        
        print(f"Episode {episode + 1}:")
        print(f"  Start: {state}")
        
        while not qlearning.env.terminal and episode_length < qlearning.env.max_steps:
            action = qlearning.get_policy_action(state)
            if action is None:
                break
                
            next_state, reward, done, info = qlearning.env.step(action)
            episode_return += reward
            episode_length += 1
            path.append(next_state)
            
            q_values = qlearning.get_action_values(state)
            current_q = q_values.get(action, 0) if q_values else 0
            
            print(f"  Step {episode_length}: {state} --{action.name}--> {next_state} "
                  f"(reward: {reward:.3f}, Q: {current_q:.6f})")
            
            state = next_state
            if done:
                break
        
        outcome = "SUCCESS" if state == qlearning.env.goal_state else "FAILURE"
        print(f"  Outcome: {outcome}")
        print(f"  Total return: {episode_return:.6f}")
        print(f"  Episode length: {episode_length}")
        print(f"  Path length: {len(path)}")
        print()
    
    qlearning.epsilon = old_epsilon  # Restore epsilon


def compare_sarsa_vs_qlearning():
    """Compare SARSA (on-policy) vs Q-learning (off-policy)."""
    print("\n" + "=" * 70)
    print("ALGORITHM COMPARISON: SARSA (ON-POLICY) vs Q-LEARNING (OFF-POLICY)")
    print("=" * 70)
    
    # Create environment
    env = StochasticGridworld(size=5, gamma=0.99, max_steps=200)
    
    # Run Q-learning
    print("Running Q-Learning (off-policy)...")
    qlearning = QLearning(env, alpha=0.1, epsilon=0.3, epsilon_decay=0.995, min_epsilon=0.01)
    qlearning_results = qlearning.run(num_episodes=2000, verbose=False)
    qlearning_evaluation = qlearning_results['final_evaluation']
    
    # Run SARSA
    print("Running SARSA (on-policy)...")
    sarsa = SARSA(env, alpha=0.1, epsilon=0.3, epsilon_decay=0.995, min_epsilon=0.01)
    sarsa_results = sarsa.run(num_episodes=2000, verbose=False)
    sarsa_evaluation = sarsa_results['final_evaluation']
    
    # Run Value Iteration for comparison
    print("Running Value Iteration (optimal)...")
    vi = ValueIteration(env, theta=1e-6)
    vi_results = vi.run(max_iterations=1000, verbose=False)
    vi_evaluation = vi.evaluate_policy(num_episodes=1000)
    
    print("\nComparison Results:")
    print("=" * 60)
    print(f"{'Metric':<25} {'Q-Learning':<12} {'SARSA':<12} {'Value Iter':<12}")
    print("-" * 60)
    print(f"{'Learning Type':<25} {'Off-Policy':<12} {'On-Policy':<12} {'Model-Based':<12}")
    print(f"{'Episodes/Iterations':<25} {qlearning_results['episodes']:<12} {sarsa_results['episodes']:<12} {vi_results['iterations']:<12}")
    print(f"{'Mean Return':<25} {qlearning_evaluation['mean_return']:<12.6f} {sarsa_evaluation['mean_return']:<12.6f} {vi_evaluation['mean_return']:<12.6f}")
    print(f"{'Success Rate':<25} {qlearning_evaluation['success_rate']:<12.3f} {sarsa_evaluation['success_rate']:<12.3f} {vi_evaluation['success_rate']:<12.3f}")
    print(f"{'Mean Episode Length':<25} {qlearning_evaluation['mean_length']:<12.2f} {sarsa_evaluation['mean_length']:<12.2f} {vi_evaluation['mean_length']:<12.2f}")
    
    # Compare final policies
    policy_agreements = {}
    
    # Q-learning vs SARSA
    agreement_ql_sarsa = 0
    total_states = 0
    for state in env.get_all_states():
        if not env.is_terminal(state):
            ql_action = qlearning.get_policy_action(state)
            sarsa_action = sarsa.get_policy_action(state)
            if ql_action == sarsa_action:
                agreement_ql_sarsa += 1
            total_states += 1
    
    policy_agreements['QL-SARSA'] = agreement_ql_sarsa / total_states if total_states > 0 else 0
    
    # Q-learning vs VI
    agreement_ql_vi = 0
    for state in env.get_all_states():
        if not env.is_terminal(state):
            ql_action = qlearning.get_policy_action(state)
            vi_action = vi.get_policy_action(state)
            if ql_action == vi_action:
                agreement_ql_vi += 1
    
    policy_agreements['QL-VI'] = agreement_ql_vi / total_states if total_states > 0 else 0
    
    # SARSA vs VI
    agreement_sarsa_vi = 0
    for state in env.get_all_states():
        if not env.is_terminal(state):
            sarsa_action = sarsa.get_policy_action(state)
            vi_action = vi.get_policy_action(state)
            if sarsa_action == vi_action:
                agreement_sarsa_vi += 1
    
    policy_agreements['SARSA-VI'] = agreement_sarsa_vi / total_states if total_states > 0 else 0
    
    print(f"{'Policy Agreement':<25}")
    print(f"{'  Q-Learning-SARSA':<25} {policy_agreements['QL-SARSA']:<12.3f}")
    print(f"{'  Q-Learning-VI':<25} {policy_agreements['QL-VI']:<12.3f}")
    print(f"{'  SARSA-VI':<25} {policy_agreements['SARSA-VI']:<12.3f}")
    
    print("\nKey Insights:")
    print("- Q-learning (off-policy): Updates using max Q-value regardless of action taken")
    print("- SARSA (on-policy): Updates using the actual next action chosen")
    print("- Q-learning can be more sample efficient in some environments")
    print("- SARSA is often more stable and safer for online learning")
    
    if policy_agreements['QL-SARSA'] > 0.8:
        print("- Q-learning and SARSA learned very similar policies ✓")
    
    if policy_agreements['QL-VI'] > 0.8 and policy_agreements['SARSA-VI'] > 0.8:
        print("- Both TD methods learned policies close to optimal ✓")
    
    return qlearning_results, sarsa_results, vi_results


if __name__ == "__main__":
    # Run main Q-learning example
    results, qlearning = main()
    
    # Demonstrate policy execution
    demonstrate_policy_execution(qlearning, num_episodes=5)
    
    # Compare SARSA vs Q-learning
    comparison_results = compare_sarsa_vs_qlearning()
    
    print("\nExample completed successfully!")
    print("\nNext steps:")
    print("1. Compare learning curves between SARSA and Q-learning")
    print("2. Try different hyperparameters for both algorithms")
    print("3. Test on environments where off-policy vs on-policy matters more")
    print("4. Implement Double Q-learning or other advanced variants")
    print("5. Analyze convergence properties and sample efficiency")
