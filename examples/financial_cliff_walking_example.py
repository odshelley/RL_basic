"""
Example: Financial Cliff Walking Environment

This example demonstrates the Financial Cliff Walking environment which is ideal
for showing CPT (Cumulative Prospect Theory) effects:

- Clear reference point (zero balance)
- Asymmetric outcomes (bankruptcy vs surplus)  
- Loss aversion (bankruptcy much worse than surplus)
- Risk-sensitive decision making
- Natural transaction costs favoring conservative policies

The optimal policy should maintain balance around 1 to avoid bankruptcy risk
while minimizing transaction costs.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, List, Tuple

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gridworld.financial_environment import (
    FinancialCliffWalking, 
    FinancialAction,
    create_conservative_environment,
    create_risky_environment,
    create_standard_environment
)


def run_random_policy(env: FinancialCliffWalking, num_episodes: int = 1000) -> Dict:
    """Run random policy for baseline comparison."""
    results = []
    
    for episode in range(num_episodes):
        balance = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            valid_actions = env.get_valid_actions(balance)
            if not valid_actions:
                break
                
            action = np.random.choice(valid_actions)
            next_balance, reward, done, info = env.step(action)
            total_reward += reward
            balance = next_balance
        
        results.append((balance, total_reward, info.get('bankrupt', False)))
    
    return env.evaluate_policy_performance(results)


def run_conservative_policy(env: FinancialCliffWalking, num_episodes: int = 1000) -> Dict:
    """
    Run conservative policy: 
    - If balance = 0: deposit to get to 1
    - If balance = 1: do nothing (safe buffer)  
    - If balance > 1: withdraw to get back to 1
    """
    results = []
    
    for episode in range(num_episodes):
        balance = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Conservative policy logic
            if balance == 0:
                action = FinancialAction.DEPOSIT
            elif balance == 1:
                action = FinancialAction.NOTHING
            elif balance > 1:
                action = FinancialAction.WITHDRAW
            else:  # balance < 0 (shouldn't happen)
                break
            
            # Check if action is valid
            valid_actions = env.get_valid_actions(balance)
            if action not in valid_actions:
                if valid_actions:
                    action = valid_actions[0]  # Fallback
                else:
                    break
            
            next_balance, reward, done, info = env.step(action)
            total_reward += reward
            balance = next_balance
        
        results.append((balance, total_reward, info.get('bankrupt', False)))
    
    return env.evaluate_policy_performance(results)


def run_aggressive_policy(env: FinancialCliffWalking, num_episodes: int = 1000) -> Dict:
    """
    Run aggressive policy:
    - Stay at balance 0 (risky but no transaction costs)
    - Only deposit if forced by noise
    """
    results = []
    
    for episode in range(num_episodes):
        balance = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Aggressive policy: prefer to stay at 0
            if balance == 0:
                action = FinancialAction.NOTHING
            elif balance > 0:
                action = FinancialAction.WITHDRAW
            else:  # balance < 0 (bankruptcy)
                break
            
            # Check if action is valid
            valid_actions = env.get_valid_actions(balance)
            if action not in valid_actions:
                if valid_actions:
                    action = valid_actions[0]  # Fallback
                else:
                    break
            
            next_balance, reward, done, info = env.step(action)
            total_reward += reward
            balance = next_balance
        
        results.append((balance, total_reward, info.get('bankrupt', False)))
    
    return env.evaluate_policy_performance(results)


def simulate_single_episode(env: FinancialCliffWalking, policy_name: str) -> None:
    """Simulate and display a single episode with step-by-step details."""
    print(f"\n{'='*50}")
    print(f"SINGLE EPISODE SIMULATION: {policy_name.upper()} POLICY")
    print(f"{'='*50}")
    
    balance = env.reset()
    print(f"Initial state:")
    print(env.render())
    print()
    
    step = 0
    while not env.terminal and step < 15:  # Limit for display
        # Choose action based on policy
        if policy_name == "conservative":
            if balance == 0:
                action = FinancialAction.DEPOSIT
            elif balance == 1:
                action = FinancialAction.NOTHING
            else:
                action = FinancialAction.WITHDRAW
        elif policy_name == "aggressive":
            if balance == 0:
                action = FinancialAction.NOTHING
            else:
                action = FinancialAction.WITHDRAW
        else:  # random
            valid_actions = env.get_valid_actions(balance)
            action = np.random.choice(valid_actions) if valid_actions else FinancialAction.NOTHING
        
        # Take step
        next_balance, reward, done, info = env.step(action)
        
        print(f"Step {step + 1}:")
        print(f"  Action: {action.name}")
        print(f"  Balance: ${balance} → ${next_balance}")
        print(f"  Reward: {reward:.3f}")
        print(f"  Done: {done}")
        if info.get('message'):
            print(f"  Info: {info['message']}")
        print()
        
        balance = next_balance
        step += 1
    
    print("Final state:")
    print(env.render())


def compare_environments():
    """Compare different financial environment configurations."""
    print("\n" + "="*60)
    print("FINANCIAL ENVIRONMENT COMPARISON")
    print("="*60)
    
    environments = {
        "Conservative": create_conservative_environment(),
        "Standard": create_standard_environment(), 
        "Risky": create_risky_environment()
    }
    
    print(f"\n{'Environment':<15} {'Bankruptcy':<12} {'Success':<10} {'Mean Reward':<12}")
    print("-" * 50)
    
    for name, env in environments.items():
        # Test with conservative policy
        results = run_conservative_policy(env, num_episodes=1000)
        
        print(f"{name:<15} {results['bankruptcy_rate']:<12.1%} {results['success_rate']:<10.1%} {results['mean_reward']:<12.3f}")


def analyze_risk_sensitivity():
    """Analyze how different policies perform under risk."""
    print("\n" + "="*60)
    print("RISK SENSITIVITY ANALYSIS")
    print("="*60)
    print("This shows why CPT agents would behave differently:")
    print("- Conservative policy: Higher transaction costs but avoids bankruptcy")
    print("- Aggressive policy: Lower transaction costs but higher bankruptcy risk")
    print()
    
    env = create_standard_environment()
    
    policies = {
        "Random": run_random_policy,
        "Conservative": run_conservative_policy,
        "Aggressive": run_aggressive_policy
    }
    
    print(f"{'Policy':<12} {'Bankruptcy':<12} {'Success':<10} {'Surplus':<10} {'Mean Reward':<12}")
    print("-" * 60)
    
    policy_results = {}
    for name, policy_func in policies.items():
        results = policy_func(env, num_episodes=2000)
        policy_results[name] = results
        
        print(f"{name:<12} {results['bankruptcy_rate']:<12.1%} " + 
              f"{results['success_rate']:<10.1%} {results['surplus_rate']:<10.1%} " +
              f"{results['mean_reward']:<12.3f}")
    
    return policy_results


def plot_policy_comparison(policy_results: Dict):
    """Plot comparison of different policies."""
    policies = list(policy_results.keys())
    bankruptcy_rates = [policy_results[p]['bankruptcy_rate'] for p in policies]
    success_rates = [policy_results[p]['success_rate'] for p in policies]
    mean_rewards = [policy_results[p]['mean_reward'] for p in policies]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Bankruptcy rates
    axes[0].bar(policies, bankruptcy_rates, color=['red', 'orange', 'darkred'], alpha=0.7)
    axes[0].set_title('Bankruptcy Rate')
    axes[0].set_ylabel('Rate')
    axes[0].set_ylim(0, max(bankruptcy_rates) * 1.1)
    
    # Success rates  
    axes[1].bar(policies, success_rates, color=['green', 'lightgreen', 'darkgreen'], alpha=0.7)
    axes[1].set_title('Success Rate')
    axes[1].set_ylabel('Rate')
    axes[1].set_ylim(0, 1)
    
    # Mean rewards
    colors = ['blue' if r >= 0 else 'red' for r in mean_rewards]
    axes[2].bar(policies, mean_rewards, color=colors, alpha=0.7)
    axes[2].set_title('Mean Reward')
    axes[2].set_ylabel('Reward')
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('financial_policy_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_cpt_motivation():
    """Demonstrate why this environment is perfect for CPT."""
    print("\n" + "="*60)
    print("WHY THIS ENVIRONMENT IS PERFECT FOR CPT")
    print("="*60)
    
    print("\n1. REFERENCE POINT:")
    print("   - Natural reference: $0 balance")
    print("   - Gains: Positive balance (surplus)")
    print("   - Losses: Negative balance (bankruptcy)")
    
    print("\n2. LOSS AVERSION:")
    print("   - Bankruptcy penalty: -$10 (huge loss)")
    print("   - Surplus penalty: -$0.1 per unit (small loss)")
    print("   - CPT agent should be much more afraid of bankruptcy")
    
    print("\n3. PROBABILITY DISTORTION:")
    print("   - Small bankruptcy probability gets overweighted")
    print("   - Makes risky actions (staying at $0) seem worse")
    print("   - Favors conservative buffer strategy")
    
    print("\n4. DIMINISHING SENSITIVITY:")
    print("   - Difference between $0 and $1 surplus matters less")
    print("   - But difference between $0 and -$1 (bankruptcy) matters hugely")
    
    print("\n5. EXPECTED POLICY DIFFERENCES:")
    print("   - Standard RL: May take more risks to avoid transaction costs")
    print("   - CPT RL: Will maintain higher safety buffer due to loss aversion")
    print("           and probability overweighting of rare bankruptcy events")


def main():
    """Main demonstration of Financial Cliff Walking environment."""
    print("="*60)
    print("FINANCIAL CLIFF WALKING ENVIRONMENT")
    print("="*60)
    print("A financial adaptation of cliff walking ideal for CPT analysis")
    
    # Create standard environment
    env = create_standard_environment()
    
    print(f"\nEnvironment Configuration:")
    print(f"- Max balance: ${env.max_balance}")
    print(f"- Max steps: {env.max_steps}")
    print(f"- Transaction cost: ${env.transaction_cost}")
    print(f"- Bankruptcy penalty: ${env.bankruptcy_penalty}")
    print(f"- Surplus penalty rate: ${env.surplus_penalty_rate}/unit")
    print(f"- Action noise probability: {env.action_noise_prob}")
    
    # Simulate single episodes
    simulate_single_episode(env, "conservative")
    simulate_single_episode(env, "aggressive")
    
    # Compare environments
    compare_environments()
    
    # Analyze risk sensitivity
    policy_results = analyze_risk_sensitivity()
    
    # Plot comparison
    plot_policy_comparison(policy_results)
    
    # Explain CPT motivation
    demonstrate_cpt_motivation()
    
    return env, policy_results


if __name__ == "__main__":
    env, results = main()
    
    print(f"\n{'='*60}")
    print("NEXT STEPS FOR CPT ANALYSIS")
    print("="*60)
    print("1. Implement CPT Value Iteration for this environment")
    print("2. Compare CPT vs Standard RL policies")  
    print("3. Vary loss aversion (λ) and probability weighting (γ)")
    print("4. Analyze policy differences in different risk scenarios")
    print("\nThis environment should show much stronger CPT effects than")
    print("the gridworld due to asymmetric losses and clear reference point!")
