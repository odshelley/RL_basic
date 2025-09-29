"""
Example: CPT Value Iteration on 5x5 Stochastic Gridworld

This example demonstrates how CPT (Cumulative Prospect Theory) Value Iteration 
differs from standard Value Iteration by using probability weighting in decisions.

The agent will be more sensitive to unlikely events (like slipping into pits)
due to probability distortion.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict

# Add src to Python path for uv compatibility
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import from the local src directory
from gridworld.environment import StochasticGridworld, Action
from algorithms.value_iteration import ValueIteration
from algorithms.cpt_value_iteration import CPTValueIteration, prelec_weighting, tversky_kahneman_weighting
from utils.visualization import plot_gridworld, plot_value_function, plot_policy


def compare_policies(standard_policy: Dict, cpt_policy: Dict, env: StochasticGridworld):
    """Compare standard and CPT policies to highlight differences."""
    differences = []
    
    for state in env.get_all_states():
        if env.is_terminal(state):
            continue
            
        std_action = standard_policy.get(state)
        cpt_action = cpt_policy.get(state)
        
        if std_action != cpt_action:
            differences.append({
                'state': state,
                'standard': std_action.name if std_action else 'None',
                'cpt': cpt_action.name if cpt_action else 'None'
            })
    
    return differences


def main():
    print("=" * 60)
    print("CPT VALUE ITERATION VS STANDARD VALUE ITERATION")
    print("=" * 60)
    
    # Create environment
    env = StochasticGridworld(size=5, gamma=0.99)
    
    print("\nEnvironment Configuration:")
    print(f"- Grid size: {env.size}x{env.size}")
    print(f"- Discount factor: {env.gamma}")
    print(f"- Transition probabilities: {env.intended_prob:.1f} intended, {env.slip_prob:.1f} each slip")
    print()
    
    # Run standard Value Iteration
    print("Running Standard Value Iteration...")
    standard_vi = ValueIteration(env, theta=1e-6)
    standard_results = standard_vi.run(verbose=False)
    print(f"Converged in {standard_results['iterations']} iterations")
    
    # Run CPT Value Iteration with different weighting functions
    print("\nRunning CPT Value Iteration (Prelec weighting, γ=0.65)...")
    cpt_vi_prelec = CPTValueIteration(env, theta=1e-6, probability_weighting=prelec_weighting(0.65))
    cpt_results_prelec = cpt_vi_prelec.run(verbose=False)
    print(f"Converged in {cpt_results_prelec['iterations']} iterations")
    
    print("\nRunning CPT Value Iteration (Tversky-Kahneman weighting, γ=0.61)...")
    cpt_vi_tk = CPTValueIteration(env, theta=1e-6, probability_weighting=tversky_kahneman_weighting(0.61))
    cpt_results_tk = cpt_vi_tk.run(verbose=False)
    print(f"Converged in {cpt_results_tk['iterations']} iterations")
    
    # Compare policies
    print("\n" + "="*40)
    print("POLICY COMPARISON")
    print("="*40)
    
    # Standard vs Prelec
    prelec_diffs = compare_policies(standard_results['final_policy'], 
                                   cpt_results_prelec['final_policy'], env)
    if prelec_diffs:
        print(f"\nStandard vs Prelec CPT: {len(prelec_diffs)} policy differences")
        for diff in prelec_diffs[:5]:  # Show first 5 differences
            print(f"  State {diff['state']}: {diff['standard']} → {diff['cpt']}")
    else:
        print("\nStandard vs Prelec CPT: Identical policies")
    
    # Standard vs Tversky-Kahneman
    tk_diffs = compare_policies(standard_results['final_policy'], 
                               cpt_results_tk['final_policy'], env)
    if tk_diffs:
        print(f"\nStandard vs Tversky-Kahneman CPT: {len(tk_diffs)} policy differences")
        for diff in tk_diffs[:5]:  # Show first 5 differences
            print(f"  State {diff['state']}: {diff['standard']} → {diff['cpt']}")
    else:
        print("\nStandard vs Tversky-Kahneman CPT: Identical policies")
    
    # Evaluate policies
    print("\n" + "="*40)
    print("POLICY EVALUATION (1000 episodes)")
    print("="*40)
    
    # Standard policy
    standard_eval = standard_vi.evaluate_policy(1000)
    print(f"\nStandard Value Iteration:")
    print(f"  Mean return: {standard_eval['mean_return']:.3f} ± {standard_eval['std_return']:.3f}")
    print(f"  Success rate: {standard_eval['success_rate']*100:.1f}%")
    print(f"  Mean episode length: {standard_eval['mean_length']:.1f}")
    
    # CPT policies evaluation (using standard environment dynamics)
    # Note: This evaluates the CPT policy in the true environment
    cpt_vi_prelec.policy = cpt_results_prelec['final_policy']
    prelec_eval = evaluate_cpt_policy(env, cpt_vi_prelec.policy, 1000)
    print(f"\nCPT Prelec (γ=0.65):")
    print(f"  Mean return: {prelec_eval['mean_return']:.3f} ± {prelec_eval['std_return']:.3f}")
    print(f"  Success rate: {prelec_eval['success_rate']*100:.1f}%")
    print(f"  Mean episode length: {prelec_eval['mean_length']:.1f}")
    
    cpt_vi_tk.policy = cpt_results_tk['final_policy']
    tk_eval = evaluate_cpt_policy(env, cpt_vi_tk.policy, 1000)
    print(f"\nCPT Tversky-Kahneman (γ=0.61):")
    print(f"  Mean return: {tk_eval['mean_return']:.3f} ± {tk_eval['std_return']:.3f}")
    print(f"  Success rate: {tk_eval['success_rate']*100:.1f}%")
    print(f"  Mean episode length: {tk_eval['mean_length']:.1f}")
    
    # Visualize probability weighting functions
    visualize_weighting_functions()
    
    # Plot value functions
    plot_value_comparison(env, standard_vi, cpt_vi_prelec, cpt_vi_tk)
    
    return standard_vi, cpt_vi_prelec, cpt_vi_tk


def evaluate_cpt_policy(env, policy, num_episodes):
    """Evaluate a CPT policy in the environment."""
    returns = []
    lengths = []
    success_count = 0
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_return = 0
        episode_length = 0
        
        while not env.terminal and episode_length < env.max_steps:
            action = policy.get(state)
            if action is None:
                break
                
            next_state, reward, done, _ = env.step(action)
            episode_return += reward
            episode_length += 1
            state = next_state
            
            if done:
                if state == env.goal_state:
                    success_count += 1
                break
        
        returns.append(episode_return)
        lengths.append(episode_length)
    
    return {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'mean_length': np.mean(lengths),
        'success_rate': success_count / num_episodes
    }


def visualize_weighting_functions():
    """Plot different probability weighting functions."""
    p = np.linspace(0.001, 0.999, 1000)
    
    plt.figure(figsize=(10, 6))
    
    # Linear (risk-neutral)
    plt.plot(p, p, 'k--', label='Linear (Risk-neutral)', linewidth=2)
    
    # Prelec
    prelec = prelec_weighting(0.65)
    plt.plot(p, [prelec(pi) for pi in p], 'b-', label='Prelec (γ=0.65)', linewidth=2)
    
    # Tversky-Kahneman
    tk = tversky_kahneman_weighting(0.61)
    plt.plot(p, [tk(pi) for pi in p], 'r-', label='Tversky-Kahneman (γ=0.61)', linewidth=2)
    
    # More risk-averse
    prelec_averse = prelec_weighting(0.4)
    plt.plot(p, [prelec_averse(pi) for pi in p], 'g-', label='Prelec (γ=0.4, risk-averse)', linewidth=2)
    
    plt.xlabel('Objective Probability', fontsize=12)
    plt.ylabel('Decision Weight', fontsize=12)
    plt.title('Probability Weighting Functions in CPT', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('cpt_weighting_functions.png', dpi=150)
    plt.show()


def plot_value_comparison(env, standard_vi, cpt_prelec, cpt_tk):
    """Plot value functions side by side."""
    # Standard VI
    fig1 = plot_value_function(env, standard_vi.V, 'Standard Value Iteration', figsize=(5, 4))
    
    # CPT Prelec
    cpt_v_prelec = {s: cpt_prelec.get_state_value(s) for s in env.get_all_states()}
    fig2 = plot_value_function(env, cpt_v_prelec, 'CPT Prelec (γ=0.65)', figsize=(5, 4))
    
    # CPT Tversky-Kahneman
    cpt_v_tk = {s: cpt_tk.get_state_value(s) for s in env.get_all_states()}
    fig3 = plot_value_function(env, cpt_v_tk, 'CPT Tversky-Kahneman (γ=0.61)', figsize=(5, 4))
    
    # Save the figures
    fig1.savefig('cpt_standard_vi.png', dpi=150, bbox_inches='tight')
    fig2.savefig('cpt_prelec_vi.png', dpi=150, bbox_inches='tight')
    fig3.savefig('cpt_tk_vi.png', dpi=150, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    standard_vi, cpt_vi_prelec, cpt_vi_tk = main()
    
    print("\n" + "="*60)
    print("CPT INTERPRETATION")
    print("="*60)
    print("\nProbability weighting in CPT causes the agent to:")
    print("- Overweight small probabilities (like slipping into pits)")
    print("- Underweight moderate to high probabilities")
    print("- This can lead to more cautious policies near dangerous states")
    print("\nThe Choquet integral respects this probability distortion,")
    print("leading to risk-sensitive decision making that differs from")
    print("expected utility maximization.")