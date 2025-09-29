"""
Example: Policy Iteration on 5x5 Stochastic Gridworld

This example demonstrates how to use Policy Iteration to solve the stochastic gridworld
problem. It includes:
- Environment setup
- Running Policy Iteration
- Visualizing results
- Policy evaluation
"""

import sys
import os

# Add src to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt

from gridworld.environment import StochasticGridworld, Action
from algorithms.policy_iteration import PolicyIteration
from utils.visualization import plot_gridworld, plot_value_function, plot_policy, plot_learning_curve


def main():
    print("=" * 50)
    print("POLICY ITERATION ON STOCHASTIC GRIDWORLD")
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
    
    # Initialize Policy Iteration
    pi = PolicyIteration(env, theta=1e-6)
    
    print("Running Policy Iteration...")
    results = pi.run(max_iterations=100, verbose=True)
    
    print("\nResults Summary:")
    print(f"- Converged: {results['converged']}")
    print(f"- Iterations: {results['iterations']}")
    print(f"- Total policy evaluation iterations: {results['total_eval_iterations']}")
    print(f"- Policy changes: {results['policy_changes']}")
    print()
    
    # Show some example state values and policies
    print("Sample State Values and Policies:")
    sample_states = [(4, 0), (2, 2), (1, 3), (0, 4)]
    for state in sample_states:
        value = pi.get_state_value(state)
        action = pi.get_policy_action(state)
        action_values = pi.get_action_values(state)
        
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
    evaluation = pi.evaluate_policy(num_episodes=1000)
    
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
                              title="Optimal Value Function (Policy Iteration)")
    
    # Plot 3: Final policy
    fig3 = plot_policy(env, results['final_policy'], results['final_value_function'],
                      title="Optimal Policy (Policy Iteration)")
    
    # Plot 4: Learning curve
    if results['value_history']:
        fig4 = plot_learning_curve(results['value_history'], env,
                                  title="Policy Iteration Convergence")
    
    # Show all plots
    plt.show()
    
    # Optional: Save results
    print("\nTo save results, uncomment the following lines in the script:")
    print("# import pickle")
    print("# with open('policy_iteration_results.pkl', 'wb') as f:")
    print("#     pickle.dump(results, f)")
    print("# fig1.savefig('gridworld_environment.png', dpi=300, bbox_inches='tight')")
    print("# fig2.savefig('value_function.png', dpi=300, bbox_inches='tight')")
    print("# fig3.savefig('optimal_policy.png', dpi=300, bbox_inches='tight')")
    print("# fig4.savefig('convergence_curve.png', dpi=300, bbox_inches='tight')")
    
    return results


def demonstrate_policy_execution():
    """Demonstrate executing the learned policy in the environment."""
    print("\n" + "=" * 50)
    print("POLICY EXECUTION DEMONSTRATION")
    print("=" * 50)
    
    # Create environment and solve with policy iteration
    env = StochasticGridworld(size=5, gamma=0.99, max_steps=200)
    pi = PolicyIteration(env, theta=1e-6)
    pi.run(max_iterations=100, verbose=False)
    
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
            action = pi.get_policy_action(state)
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


if __name__ == "__main__":
    # Run main example
    results = main()
    
    # Demonstrate policy execution
    demonstrate_policy_execution()
    
    print("\nExample completed successfully!")
    print("\nNext steps:")
    print("1. Try modifying environment parameters (size, rewards, probabilities)")
    print("2. Implement and compare other algorithms (Value Iteration, Q-learning, etc.)")
    print("3. Analyze sensitivity to hyperparameters")
    print("4. Experiment with different gridworld layouts")
