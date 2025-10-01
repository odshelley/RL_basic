#!/usr/bin/env python3
"""
Simple Financial CPT Demonstration

A simple demonstration of how CPT principles would affect financial decision making,
using a basic financial scenario without the full MDP framework.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gridworld.financial_environment import FinancialCliffWalking, FinancialAction


def prelec_weighting(p, gamma=0.65):
    """Prelec probability weighting function."""
    if p == 0:
        return 0
    if p == 1:
        return 1
    return np.exp(-(-np.log(p))**gamma)


def tversky_kahneman_weighting(p, gamma=0.61, delta=0.69):
    """Tversky & Kahneman probability weighting function."""
    if p == 0:
        return 0
    if p == 1:
        return 1
    # For gains (using gamma)
    return (p**gamma) / ((p**gamma + (1-p)**gamma)**(1/gamma))


def cpt_utility(x, alpha=0.88, beta=0.88, lam=2.25, reference=0.0):
    """CPT utility function with loss aversion."""
    relative_x = x - reference
    if relative_x >= 0:
        return relative_x ** alpha
    else:
        return -lam * ((-relative_x) ** beta)


def evaluate_financial_choice(balance, action, env, weighting_func=None, utility_func=None):
    """
    Evaluate a financial choice using CPT principles.
    
    Returns the CPT value of taking an action from a given balance.
    """
    if env.is_terminal(balance):
        return 0.0
    
    # Get transition probabilities
    transitions = env.get_transition_probabilities(balance, action)
    
    # Get outcomes (rewards + future value, simplified as just rewards here)
    outcomes = []
    probabilities = []
    
    for next_balance, prob in transitions.items():
        reward = env.get_reward(balance, action, next_balance)
        # Simplified: just use immediate reward as outcome
        outcomes.append(reward)
        probabilities.append(prob)
    
    # Convert to numpy arrays and sort by outcome value
    outcomes = np.array(outcomes)
    probabilities = np.array(probabilities)
    
    # Sort by outcome value (ascending)
    sorted_indices = np.argsort(outcomes)
    outcomes = outcomes[sorted_indices]
    probabilities = probabilities[sorted_indices]
    
    # Apply utility function if provided
    if utility_func is not None:
        utility_values = np.array([utility_func(outcome) for outcome in outcomes])
    else:
        utility_values = outcomes
    
    # Apply probability weighting if provided
    if weighting_func is not None:
        # Compute cumulative probabilities from the worst outcome
        cumulative_probs = np.cumsum(probabilities)
        
        # Apply probability weighting to cumulative probabilities
        weighted_cumprobs = np.array([weighting_func(p) for p in cumulative_probs])
        
        # Compute decision weights (differences between consecutive weighted cumulative probabilities)
        decision_weights = np.zeros_like(weighted_cumprobs)
        decision_weights[0] = weighted_cumprobs[0]
        for i in range(1, len(decision_weights)):
            decision_weights[i] = weighted_cumprobs[i] - weighted_cumprobs[i-1]
    else:
        decision_weights = probabilities
    
    # Compute CPT value as sum of utility × decision weight
    cpt_value = np.sum(utility_values * decision_weights)
    return cpt_value


def compare_decision_frameworks():
    """Compare standard expected utility vs CPT for financial decisions."""
    
    print("=" * 70)
    print("FINANCIAL DECISION MAKING: EXPECTED UTILITY vs CPT")
    print("=" * 70)
    
    # Create financial environment
    env = FinancialCliffWalking(max_balance=10, max_steps=20, 
                               transaction_cost=0.1, bankruptcy_penalty=-10.0)
    
    # Test different balances and actions
    test_scenarios = [
        (0, "Starting with $0 balance"),
        (1, "Small surplus: $1 balance"), 
        (2, "Moderate surplus: $2 balance"),
        (5, "Large surplus: $5 balance")
    ]
    
    actions_to_test = [
        (FinancialAction.DEPOSIT, "DEPOSIT (+$1)"),
        (FinancialAction.WITHDRAW, "WITHDRAW (-$1)"),
        (FinancialAction.NOTHING, "DO NOTHING")
    ]
    
    # Different decision frameworks
    frameworks = {
        "Expected Utility": {"weighting": None, "utility": None},
        "CPT (Prelec weighting only)": {"weighting": prelec_weighting, "utility": None},
        "CPT (Full TK 1992)": {"weighting": tversky_kahneman_weighting, "utility": cpt_utility},
        "CPT (High loss aversion)": {
            "weighting": tversky_kahneman_weighting, 
            "utility": lambda x: cpt_utility(x, lam=5.0)
        }
    }
    
    for balance, scenario_desc in test_scenarios:
        print(f"\n{scenario_desc}")
        print("-" * 50)
        
        # Evaluate each action under each framework
        results = {}
        for framework_name, framework_config in frameworks.items():
            results[framework_name] = {}
            
            for action, action_desc in actions_to_test:
                if action in env.get_valid_actions(balance):
                    value = evaluate_financial_choice(
                        balance, action, env,
                        weighting_func=framework_config["weighting"],
                        utility_func=framework_config["utility"]
                    )
                    results[framework_name][action_desc] = value
                else:
                    results[framework_name][action_desc] = "Invalid"
        
        # Print results in a table format
        print(f"{'Action':<15}", end="")
        for framework in frameworks.keys():
            print(f"{framework:<20}", end="")
        print()
        
        print("-" * (15 + 20 * len(frameworks)))
        
        for action, action_desc in actions_to_test:
            print(f"{action_desc:<15}", end="")
            for framework_name in frameworks.keys():
                value = results[framework_name][action_desc]
                if isinstance(value, str):
                    print(f"{value:<20}", end="")
                else:
                    print(f"{value:>8.3f}{'':>12}", end="")
            print()
        
        # Determine best action for each framework
        print(f"\nBest Action:")
        for framework_name in frameworks.keys():
            valid_actions = {k: v for k, v in results[framework_name].items() 
                           if isinstance(v, (int, float))}
            if valid_actions:
                best_action = max(valid_actions.keys(), key=lambda k: valid_actions[k])
                print(f"  {framework_name}: {best_action}")


def plot_utility_functions():
    """Plot different utility functions to show CPT effects."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Outcome range
    outcomes = np.linspace(-10, 10, 200)
    
    # Standard utility (identity)
    standard_utility = outcomes
    
    # CPT utility functions
    cpt_standard = [cpt_utility(x, alpha=0.88, beta=0.88, lam=2.25) for x in outcomes]
    cpt_high_la = [cpt_utility(x, alpha=0.88, beta=0.88, lam=5.0) for x in outcomes]
    
    # Plot utility functions
    ax1.plot(outcomes, standard_utility, 'b-', label='Standard Utility', linewidth=2)
    ax1.plot(outcomes, cpt_standard, 'r--', label='CPT (λ=2.25)', linewidth=2)
    ax1.plot(outcomes, cpt_high_la, 'g:', label='CPT High Loss Aversion (λ=5.0)', linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Outcome (relative to reference point)')
    ax1.set_ylabel('Utility')
    ax1.set_title('Utility Functions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot probability weighting functions
    probabilities = np.linspace(0.01, 0.99, 100)
    
    prelec_weights = [prelec_weighting(p, gamma=0.65) for p in probabilities]
    tk_weights = [tversky_kahneman_weighting(p, gamma=0.61) for p in probabilities]
    
    ax2.plot(probabilities, probabilities, 'b-', label='Standard (no weighting)', linewidth=2)
    ax2.plot(probabilities, prelec_weights, 'r--', label='Prelec (γ=0.65)', linewidth=2) 
    ax2.plot(probabilities, tk_weights, 'g:', label='Tversky-Kahneman (γ=0.61)', linewidth=2)
    ax2.set_xlabel('Objective Probability')
    ax2.set_ylabel('Decision Weight')
    ax2.set_title('Probability Weighting Functions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cpt_functions.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Run the simple CPT financial demonstration."""
    
    # Compare decision frameworks
    compare_decision_frameworks()
    
    # Plot CPT functions
    plot_utility_functions()
    
    print("\n" + "=" * 70)
    print("KEY CPT INSIGHTS FOR FINANCIAL DECISIONS")
    print("=" * 70)
    print("\n1. LOSS AVERSION:")
    print("   - Losses feel worse than equivalent gains feel good")
    print("   - Makes agents more conservative near reference point")
    
    print("\n2. PROBABILITY WEIGHTING:")
    print("   - Small probabilities are overweighted (fear of rare bankruptcies)")
    print("   - Large probabilities are underweighted")
    
    print("\n3. REFERENCE DEPENDENCE:")
    print("   - Outcomes evaluated relative to reference point (e.g., $0 balance)")
    print("   - Same outcome can be gain or loss depending on reference")
    
    print("\n4. IMPLICATIONS FOR FINANCIAL RL:")
    print("   - CPT agents should be more risk-averse near bankruptcy")
    print("   - May prefer 'buffer' strategies (maintain small positive balance)")
    print("   - Overweight low-probability high-impact events")


if __name__ == "__main__":
    main()
