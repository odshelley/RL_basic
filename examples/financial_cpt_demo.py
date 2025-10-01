#!/usr/bin/env python3
"""
Demonstration of CPT on Financial Cliff Walking Environment

This shows how different CPT parameters affect decision-making in a financial
context where the agent can go bankrupt or achieve surplus.
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithms.value_iteration import ValueIteration
from algorithms.cpt_value_iteration import CPTValueIteration
from gridworld.financial_environment import FinancialCliffWalking
from utils.visualization import plot_value_function


def create_cpt_variants(env):
    """Create different CPT agents with varying risk preferences."""
    variants = {
        'Standard RL': ValueIteration(env),
        'Choquet Only': CPTValueIteration(env, 
                                        weighting_function='prelec',
                                        weighting_gamma=0.65,
                                        use_utility_function=False),
        'Full CPT (TK 1992)': CPTValueIteration(env,
                                              weighting_function='tversky_kahneman',
                                              weighting_gamma=0.61,
                                              weighting_delta=0.69,
                                              utility_alpha=0.88,
                                              utility_beta=0.88,
                                              loss_aversion_lambda=2.25,
                                              use_utility_function=True),
        'High Loss Aversion': CPTValueIteration(env,
                                              weighting_function='tversky_kahneman',
                                              weighting_gamma=0.61,
                                              weighting_delta=0.69,
                                              utility_alpha=0.88,
                                              utility_beta=0.88,
                                              loss_aversion_lambda=5.0,
                                              use_utility_function=True),
        'Risk Seeking': CPTValueIteration(env,
                                        weighting_function='prelec',
                                        weighting_gamma=1.2,  # Risk seeking weighting
                                        utility_alpha=1.2,
                                        utility_beta=1.2,
                                        loss_aversion_lambda=1.0,  # No loss aversion
                                        use_utility_function=True)
    }
    return variants


def analyze_financial_policies(env, agents):
    """Analyze how different agents handle financial risk."""
    print("=" * 70)
    print("FINANCIAL RISK ANALYSIS")
    print("=" * 70)
    
    # Key balances to analyze
    critical_balances = [0, 1, 2, 3, 5]  # Reference point and various surplus levels
    
    for name, agent in agents.items():
        print(f"\n{name}:")
        agent.value_iteration(theta=1e-6)
        
        print(f"  Converged in {agent.num_iterations} iterations")
        
        # Check policy at critical balances
        for balance in critical_balances:
            if env.is_valid_balance(balance):
                try:
                    action = agent.get_policy(balance)
                    action_name = env.get_all_actions()[action].name if isinstance(action, int) else action.name
                    print(f"  Balance ${balance}: {action_name}")
                except:
                    print(f"  Balance ${balance}: Unable to get policy")
    
    return agents


def evaluate_financial_performance(env, agents, num_episodes=100):
    """Evaluate financial performance of different agents."""
    print("\n" + "=" * 70)
    print("FINANCIAL PERFORMANCE EVALUATION")
    print("=" * 70)
    
    results = {}
    
    for name, agent in agents.items():
        returns = []
        bankruptcies = 0
        surplus_achieved = 0
        episode_lengths = []
        
        for _ in range(num_episodes):
            balance = env.start_balance  # Start at 0 balance
            total_return = 0
            steps = 0
            
            while not env.is_terminal(balance) and steps < env.max_steps:
                try:
                    action = agent.get_policy(balance)
                    if isinstance(action, int):
                        action_enum = env.get_all_actions()[action]
                    else:
                        action_enum = action
                    
                    # Get reward and next balance
                    reward = env.get_reward(balance, action_enum)
                    
                    # Sample next balance from transition probabilities
                    transitions = env.get_transition_probabilities(balance, action_enum)
                    next_balances = list(transitions.keys())
                    probabilities = list(transitions.values())
                    next_balance = np.random.choice(next_balances, p=probabilities)
                    
                    total_return += (env.gamma ** steps) * reward
                    balance = next_balance
                    steps += 1
                except:
                    break
            
            returns.append(total_return)
            episode_lengths.append(steps)
            
            # Check final outcome
            if balance < 0:  # Bankruptcy
                bankruptcies += 1
            elif balance > 5:  # Significant surplus 
                surplus_achieved += 1
        
        results[name] = {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'bankruptcy_rate': bankruptcies / num_episodes,
            'surplus_rate': surplus_achieved / num_episodes,
            'mean_length': np.mean(episode_lengths)
        }
        
        print(f"\n{name}:")
        print(f"  Mean return: {results[name]['mean_return']:.3f} ± {results[name]['std_return']:.3f}")
        print(f"  Bankruptcy rate: {results[name]['bankruptcy_rate']:.1%}")
        print(f"  Surplus rate: {results[name]['surplus_rate']:.1%}")
        print(f"  Mean episode length: {results[name]['mean_length']:.1f}")
    
    return results


def plot_financial_values(env, agents):
    """Plot value functions for financial environment."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    balances = env.get_all_states()
    
    for i, (name, agent) in enumerate(agents.items()):
        if i >= 6:  # Only plot first 6
            break
            
        # Get value function for all balances
        values = []
        for balance in balances:
            if not env.is_terminal(balance):
                try:
                    value = agent.get_state_value(balance)
                    values.append(value)
                except:
                    values.append(0.0)
            else:
                values.append(0.0)  # Terminal states have 0 value
        
        ax = axes[i]
        
        # Plot as line chart
        ax.plot(balances, values, 'b-o', linewidth=2, markersize=4)
        ax.set_title(f'{name}\nValue Function')
        ax.set_xlabel('Balance ($)')
        ax.set_ylabel('State Value')
        ax.grid(True, alpha=0.3)
        
        # Highlight reference point (balance = 0)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Reference Point')
        ax.legend()
        
        # Highlight bankruptcy region
        if env.min_balance < 0:
            ax.axvspan(env.min_balance, -0.5, alpha=0.2, color='red', label='Bankruptcy')
    
    # Remove unused subplots
    for i in range(len(agents), 6):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('financial_cpt_values.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Run the financial CPT demonstration."""
    print("=" * 70)
    print("CPT IN FINANCIAL CLIFF WALKING ENVIRONMENT")
    print("=" * 70)
    
    # Create financial environment
    env = FinancialCliffWalking(max_balance=10, 
                               max_steps=20,
                               transaction_cost=0.1,
                               bankruptcy_penalty=-10.0,
                               surplus_penalty_rate=0.1)
    
    print(f"Environment: Financial Cliff Walking")
    print(f"Max balance: ${env.max_balance}")
    print(f"Max steps: {env.max_steps}")
    print(f"Transaction cost: ${env.transaction_cost}")
    print(f"Bankruptcy penalty: ${env.bankruptcy_penalty}")
    print(f"Surplus penalty rate: {env.surplus_penalty_rate}")
    
    # Create different CPT variants
    agents = create_cpt_variants(env)
    
    # Analyze policies
    agents = analyze_financial_policies(env, agents)
    
    # Evaluate performance
    results = evaluate_financial_performance(env, agents)
    
    # Plot value functions
    plot_financial_values(env, agents)
    
    return env, agents, results


if __name__ == "__main__":
    env, agents, results = main()
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("\nCPT effects in financial decision-making:")
    print("1. Loss aversion makes agents avoid bankruptcy risk")
    print("2. Probability weighting overemphasizes rare events")
    print("3. Utility function curvature affects risk preferences")
    print("4. Reference point (balance=$0) creates asymmetric behavior")
    print("\nThese effects lead to more conservative financial policies")
    print("compared to standard expected utility maximization.")
