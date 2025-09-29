"""
Example: Value Iteration on 5x5 Stochastic Gridworld

This example demonstrates how to use Value Iteration to solve the stochastic gridworld
problem. It includes:
- Environment setup
- Running Value Iteration
- Visualizing results
- Policy evaluation
- Comparison with Policy Iteration
"""

import sys
import os

# Add src to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt

from gridworld.environment import StochasticGridworld, Action
from algorithms.value_iteration import ValueIteration
from algorithms.policy_iteration import PolicyIteration
from utils.visualization import plot_gridworld, plot_value_function, plot_policy, plot_learning_curve


def main():
    print("=" * 50)
    print("VALUE ITERATION ON STOCHASTIC GRIDWORLD")
    print("=" * 50)
    
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
    
    # Initialize Value Iteration
    vi = ValueIteration(env, theta=1e-6)
    
    print("Running Value Iteration...")
    results = vi.run(max_iterations=1000, verbose=True)
    
    print("\nResults Summary:")
    print(f"- Converged: {results['converged']}")
    print(f"- Iterations: {results['iterations']}")
    print(f"- Final delta: {results['final_delta']:.8f}")
    print()
    
    # Show some example state values and policies
    print("Sample State Values and Policies:")
    sample_states = [(4, 0), (2, 2), (1, 3), (0, 4)]
    for state in sample_states:
        value = vi.get_state_value(state)
        action = vi.get_policy_action(state)
        action_values = vi.get_action_values(state)
        
        print(f"State {state}:")
        print(f"  Value: {value:.6f}")
        print(f"  Policy action: {action}")
        if action_values:
            print("  Action values:")
            for act, val in action_values.items():
                print(f"    {act.name}: {val:.6f}")
        print()
    
    # Evaluate the learned policy
    print("Evaluating learned policy...")
    evaluation = vi.evaluate_policy(num_episodes=1000)
    
    print("Policy Evaluation Results:")
    print(f"- Mean return: {evaluation['mean_return']:.6f} ± {evaluation['std_return']:.6f}")
    print(f"- Mean episode length: {evaluation['mean_length']:.2f} ± {evaluation['std_length']:.2f}")
    print(f"- Success rate: {evaluation['success_rate']:.3f}")
    print()
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Plot 1: Environment layout
    fig1 = plot_gridworld(env, title="5x5 Stochastic Gridworld Environment")
    
    # Plot 2: Final value function
    fig2 = plot_value_function(env, results['final_value_function'], 
                              title="Optimal Value Function (Value Iteration)")
    
    # Plot 3: Final policy
    fig3 = plot_policy(env, results['final_policy'], results['final_value_function'],
                      title="Optimal Policy (Value Iteration)")
    
    # Plot 4: Learning curve
    if results['value_history']:
        fig4 = plot_learning_curve(results['value_history'], env,
                                  title="Value Iteration Convergence")
    
    # Show all plots
    plt.show()
    
    # Optional: Save results
    print("\nTo save results, uncomment the following lines in the script:")
    print("# import pickle")
    print("# with open('value_iteration_results.pkl', 'wb') as f:")
    print("#     pickle.dump(results, f)")
    print("# fig1.savefig('gridworld_environment.png', dpi=300, bbox_inches='tight')")
    print("# fig2.savefig('value_function_vi.png', dpi=300, bbox_inches='tight')")
    print("# fig3.savefig('optimal_policy_vi.png', dpi=300, bbox_inches='tight')")
    print("# fig4.savefig('convergence_curve_vi.png', dpi=300, bbox_inches='tight')")
    
    return results


def demonstrate_policy_execution():
    """Demonstrate executing the learned policy in the environment."""
    print("\n" + "=" * 50)
    print("POLICY EXECUTION DEMONSTRATION")
    print("=" * 50)
    
    # Create environment and solve with value iteration
    env = StochasticGridworld(size=5, gamma=0.99, max_steps=200)
    vi = ValueIteration(env, theta=1e-6)
    vi.run(max_iterations=1000, verbose=False)
    
    # Run a few episodes
    num_episodes = 5
    print(f"Running {num_episodes} episodes with the learned policy:\n")
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_return = 0
        episode_length = 0
        path = [state]
        
        print(f"Episode {episode + 1}:")
        print(f"  Start: {state}")
        
        while not env.terminal and episode_length < env.max_steps:
            action = vi.get_policy_action(state)
            if action is None:
                break
                
            next_state, reward, done, info = env.step(action)
            episode_return += reward
            episode_length += 1
            path.append(next_state)
            
            print(f"  Step {episode_length}: {state} --{action.name}--> {next_state} (reward: {reward:.3f})")
            
            state = next_state
            if done:
                break
        
        outcome = "SUCCESS" if state == env.goal_state else "FAILURE"
        print(f"  Outcome: {outcome}")
        print(f"  Total return: {episode_return:.6f}")
        print(f"  Episode length: {episode_length}")
        print(f"  Path length: {len(path)}")
        print()


def compare_algorithms():
    """Compare Value Iteration vs Policy Iteration."""
    print("\n" + "=" * 60)
    print("ALGORITHM COMPARISON: VALUE ITERATION vs POLICY ITERATION")
    print("=" * 60)
    
    # Create environment
    env = StochasticGridworld(size=5, gamma=0.99, max_steps=200)
    
    # Run Value Iteration
    print("Running Value Iteration...")
    vi = ValueIteration(env, theta=1e-6)
    vi_results = vi.run(max_iterations=1000, verbose=False)
    vi_evaluation = vi.evaluate_policy(num_episodes=1000)
    
    # Run Policy Iteration
    print("Running Policy Iteration...")
    pi = PolicyIteration(env, theta=1e-6)
    pi_results = pi.run(max_iterations=100, verbose=False)
    pi_evaluation = pi.evaluate_policy(num_episodes=1000)
    
    print("\nComparison Results:")
    print("=" * 40)
    print(f"{'Metric':<25} {'Value Iter':<12} {'Policy Iter':<12}")
    print("-" * 40)
    print(f"{'Converged':<25} {vi_results['converged']:<12} {pi_results['converged']:<12}")
    print(f"{'Iterations':<25} {vi_results['iterations']:<12} {pi_results['iterations']:<12}")
    print(f"{'Mean Return':<25} {vi_evaluation['mean_return']:<12.6f} {pi_evaluation['mean_return']:<12.6f}")
    print(f"{'Success Rate':<25} {vi_evaluation['success_rate']:<12.3f} {pi_evaluation['success_rate']:<12.3f}")
    print(f"{'Mean Episode Length':<25} {vi_evaluation['mean_length']:<12.2f} {pi_evaluation['mean_length']:<12.2f}")
    
    # Check if policies are identical
    policies_identical = True
    for state in env.get_all_states():
        if vi.get_policy_action(state) != pi.get_policy_action(state):
            policies_identical = False
            break
    
    print(f"{'Policies Identical':<25} {policies_identical}")
    
    # Check if value functions are close
    max_value_diff = 0
    for state in env.get_all_states():
        diff = abs(vi.get_state_value(state) - pi.get_state_value(state))
        max_value_diff = max(max_value_diff, diff)
    
    print(f"{'Max Value Difference':<25} {max_value_diff:<12.8f}")
    
    print("\nInsights:")
    if vi_results['iterations'] < pi_results['iterations']:
        print("- Value Iteration converged in fewer iterations")
    elif vi_results['iterations'] > pi_results['iterations']:
        print("- Policy Iteration converged in fewer iterations")
    else:
        print("- Both algorithms converged in the same number of iterations")
    
    if policies_identical:
        print("- Both algorithms found identical optimal policies ✓")
    else:
        print("- Algorithms found different policies (may still be optimal)")
    
    if max_value_diff < 1e-5:
        print("- Value functions are essentially identical ✓")
    else:
        print(f"- Value functions differ by up to {max_value_diff:.8f}")
    
    return vi_results, pi_results, vi_evaluation, pi_evaluation


if __name__ == "__main__":
    # Run main Value Iteration example
    results = main()
    
    # Demonstrate policy execution
    demonstrate_policy_execution()
    
    # Compare with Policy Iteration
    comparison_results = compare_algorithms()
    
    print("\nExample completed successfully!")
    print("\nNext steps:")
    print("1. Try different convergence thresholds (theta) for both algorithms")
    print("2. Experiment with different environment sizes")
    print("3. Implement and compare model-free algorithms (Q-learning, SARSA)")
    print("4. Analyze computational complexity and memory usage")
    print("5. Test on environments with different reward structures")
