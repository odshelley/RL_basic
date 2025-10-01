"""
CPT Value Iteration Algorithm for the Stochastic Gridworld

This implements Value Iteration with Cumulative Prospect Theory (CPT), where
the standard expectation is replaced by a Choquet integral with probability weighting.

For each (s, a):
Q_{t+1}(s, a) ← (1 - α_t)Q_t(s, a) + α_t[r(s, a) + γ ρ_g(max_{a'} Q_t(S', a') | s, a)]

where ρ_g is the Choquet integral with probability weighting function g.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Callable
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gridworld.environment import StochasticGridworld, Action


class CPTValueIteration:
    def __init__(self, 
                 env: StochasticGridworld, 
                 theta: float = 1e-6,
                 probability_weighting: Optional[Callable[[float], float]] = None,
                 alpha: float = 1.0,
                 # CPT Utility Function Parameters
                 utility_alpha: float = 0.88,
                 utility_beta: float = 0.88,
                 loss_aversion_lambda: float = 2.25,
                 reference_point: float = 0.0,
                 use_utility_function: bool = True):
        """
        Initialize CPT Value Iteration algorithm with full CPT framework.
        
        Args:
            env: The gridworld environment
            theta: Threshold for value function convergence
            probability_weighting: Function g for probability weighting. 
                                 If None, uses Prelec weighting function
            alpha: Learning rate (1.0 for standard VI, < 1.0 for incremental)
            utility_alpha: Utility curvature parameter for gains (α ≈ 0.88)
            utility_beta: Utility curvature parameter for losses (β ≈ 0.88) 
            loss_aversion_lambda: Loss aversion parameter (λ ≈ 2.25)
            reference_point: Reference point for utility function (usually 0)
            use_utility_function: If True, uses full CPT; if False, uses only Choquet
        """
        self.env = env
        self.theta = theta
        self.alpha = alpha
        
        # CPT Utility Function Parameters (Tversky & Kahneman 1992)
        self.utility_alpha = utility_alpha  # Gains curvature (α ≈ 0.88)
        self.utility_beta = utility_beta    # Losses curvature (β ≈ 0.88)
        self.loss_aversion_lambda = loss_aversion_lambda  # Loss aversion (λ ≈ 2.25)
        self.reference_point = reference_point  # Reference point (usually 0)
        self.use_utility_function = use_utility_function
        
        # Set probability weighting function
        if probability_weighting is None:
            # Default: Prelec (1998) weighting function with parameter 0.65
            self.g = lambda p: np.exp(-(-np.log(p))**0.65) if p > 0 else 0
        else:
            self.g = probability_weighting
        
        # Get all states and actions
        self.states = env.get_all_states()
        self.actions = env.get_all_actions()
        self.num_states = len(self.states)
        self.num_actions = len(self.actions)
        
        # Initialize Q-function instead of V-function for CPT
        self.Q = {}
        for state in self.states:
            self.Q[state] = {action: 0.0 for action in self.actions}
        
        # Policy will be extracted from Q-values
        self.policy = {}
        
        # Track convergence
        self.iteration_count = 0
        self.converged = False
        self.q_history = []
    
    def compute_choquet_integral(self, 
                                outcomes: List[float], 
                                probabilities: List[float]) -> float:
        """
        Compute the Choquet integral with probability weighting.
        
        For a discrete distribution {x_i, p_i}, sort values x_(1) ≤ ... ≤ x_(n),
        compute cumulative probabilities C_i = Σ_{j≤i} p_(j),
        and decision weights π_(i) = g(C_i) - g(C_{i-1}).
        Then ρ_g(X) = Σ_i π_(i) x_(i)
        
        Args:
            outcomes: List of possible outcomes (values)
            probabilities: List of corresponding probabilities
            
        Returns:
            Choquet integral value
        """
        if not outcomes:
            return 0.0
        
        # Create list of (outcome, probability) pairs
        pairs = list(zip(outcomes, probabilities))
        
        # Sort by outcome value (ascending)
        sorted_pairs = sorted(pairs, key=lambda x: x[0])
        
        # Extract sorted outcomes and probabilities
        sorted_outcomes = [p[0] for p in sorted_pairs]
        sorted_probs = [p[1] for p in sorted_pairs]
        
        # Compute cumulative probabilities
        cumulative_probs = np.cumsum([0] + sorted_probs)
        
        # Compute decision weights: π_i = g(C_i) - g(C_{i-1})
        decision_weights = []
        for i in range(len(sorted_outcomes)):
            weight = self.g(cumulative_probs[i+1]) - self.g(cumulative_probs[i])
            decision_weights.append(weight)
        
        # Compute Choquet integral: Σ_i π_i x_i
        choquet_value = sum(w * x for w, x in zip(decision_weights, sorted_outcomes))
        
        return choquet_value
    
    def cpt_utility_function(self, x: float) -> float:
        """
        CPT utility function from Tversky & Kahneman (1992).
        
        v(x) = { x^α,           x ≥ 0  (gains)
               { -λ(-x)^β,      x < 0  (losses)
        
        Args:
            x: Outcome value relative to reference point
            
        Returns:
            Utility value
        """
        if not self.use_utility_function:
            return x  # Return raw value if utility function disabled
        
        # Compute deviation from reference point
        deviation = x - self.reference_point
        
        if deviation >= 0:
            # Gains: concave utility (diminishing sensitivity)
            return deviation ** self.utility_alpha
        else:
            # Losses: convex utility + loss aversion (losses loom larger)
            return -self.loss_aversion_lambda * ((-deviation) ** self.utility_beta)
    
    def compute_cpt_action_value(self, state: Tuple[int, int], action: Action) -> float:
        """
        Compute CPT Q-value for a specific state-action pair using Choquet integral.
        
        Args:
            state: Current state
            action: Action to evaluate
            
        Returns:
            CPT value for taking action in state
        """
        if self.env.is_terminal(state):
            return 0.0
        
        # Get transition probabilities
        transitions = self.env.get_transition_probabilities(state, action)
        
        # Prepare outcomes and probabilities for Choquet integral
        raw_outcomes = []
        probabilities = []
        
        for next_state, prob in transitions.items():
            # Immediate reward
            reward = self.env.get_reward(state, action, next_state)
            
            # Future value (max over actions in next state)
            if self.env.is_terminal(next_state):
                future_value = 0.0
            else:
                future_value = max(self.Q[next_state].values())
            
            # Total raw outcome for this transition
            raw_outcome = reward + self.env.gamma * future_value
            
            raw_outcomes.append(raw_outcome)
            probabilities.append(prob)
        
        # Apply CPT utility function to outcomes
        utility_outcomes = [self.cpt_utility_function(outcome) for outcome in raw_outcomes]
        
        # Compute Choquet integral over utility-transformed outcomes
        cpt_value = self.compute_choquet_integral(utility_outcomes, probabilities)
        
        return cpt_value
    
    def cpt_value_iteration_step(self) -> float:
        """
        Perform one step of CPT value iteration.
        
        Returns:
            Maximum change in Q-function (delta)
        """
        delta = 0.0
        old_Q = {s: {a: self.Q[s][a] for a in self.actions} for s in self.states}
        
        for state in self.states:
            if self.env.is_terminal(state):
                continue
            
            for action in self.actions:
                # Compute CPT value using Choquet integral
                cpt_value = self.compute_cpt_action_value(state, action)
                
                # Update Q-value with learning rate
                old_value = old_Q[state][action]
                new_value = (1 - self.alpha) * old_value + self.alpha * cpt_value
                self.Q[state][action] = new_value
                
                # Track maximum change
                delta = max(delta, abs(old_value - new_value))
        
        return delta
    
    def extract_policy(self) -> Dict[Tuple[int, int], Optional[Action]]:
        """
        Extract the greedy policy from current Q-function.
        
        Returns:
            Policy mapping states to actions
        """
        policy = {}
        
        for state in self.states:
            if self.env.is_terminal(state):
                policy[state] = None
                continue
            
            # Find best action based on Q-values
            best_action = max(self.actions, key=lambda a: self.Q[state][a])
            policy[state] = best_action
        
        return policy
    
    def run(self, max_iterations: int = 1000, verbose: bool = True) -> Dict:
        """
        Run the complete CPT Value Iteration algorithm.
        
        Args:
            max_iterations: Maximum number of iterations
            verbose: Whether to print progress
            
        Returns:
            Dictionary with results
        """
        if verbose:
            print("Starting CPT Value Iteration...")
            print(f"Environment: {self.env.size}x{self.env.size} grid")
            print(f"States: {self.num_states}, Actions: {self.num_actions}")
            print(f"Discount factor: {self.env.gamma}")
            print(f"Learning rate: {self.alpha}")
            print(f"Probability weighting: {self.g.__name__ if hasattr(self.g, '__name__') else 'Custom'}")
            if self.use_utility_function:
                print(f"CPT Utility Function: α={self.utility_alpha}, β={self.utility_beta}, λ={self.loss_aversion_lambda}")
                print(f"Reference point: {self.reference_point}")
            else:
                print("CPT Utility Function: DISABLED (Choquet expectation only)")
            print()
        
        for iteration in range(max_iterations):
            # Store current state for history
            self.q_history.append({s: self.Q[s].copy() for s in self.states})
            
            # Perform one CPT value iteration step
            delta = self.cpt_value_iteration_step()
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration + 1}: max Q-value change = {delta:.8f}")
            
            # Check for convergence
            if delta < self.theta:
                self.converged = True
                self.iteration_count = iteration + 1
                
                if verbose:
                    print(f"\nCPT Value Iteration converged in {iteration + 1} iterations!")
                    print(f"Final max Q-value change: {delta:.8f}")
                
                break
        else:
            if verbose:
                print(f"CPT Value Iteration did not converge in {max_iterations} iterations")
        
        # Extract optimal policy
        self.policy = self.extract_policy()
        
        if verbose and self.converged:
            print("Risk-sensitive policy extracted successfully!")
        
        return {
            'converged': self.converged,
            'iterations': self.iteration_count,
            'final_delta': delta if 'delta' in locals() else float('inf'),
            'final_q_function': {s: self.Q[s].copy() for s in self.states},
            'final_policy': self.policy.copy(),
            'q_history': self.q_history
        }
    
    def get_state_value(self, state: Tuple[int, int]) -> float:
        """Get value for a state (max Q-value over actions)."""
        if self.env.is_terminal(state):
            return 0.0
        return max(self.Q[state].values())
    
    def get_action_values(self, state: Tuple[int, int]) -> Dict[Action, float]:
        """Get Q-values for all actions in a state."""
        if self.env.is_terminal(state):
            return {}
        return self.Q[state].copy()
    
    def get_policy_action(self, state: Tuple[int, int]) -> Optional[Action]:
        """Get the policy action for a specific state."""
        return self.policy.get(state)


# Common probability weighting functions for CPT
def prelec_weighting(gamma: float = 0.65):
    """Prelec (1998) probability weighting function."""
    return lambda p: np.exp(-(-np.log(p))**gamma) if p > 0 else 0

def tversky_kahneman_weighting(gamma: float = 0.61):
    """Tversky & Kahneman (1992) probability weighting function."""
    return lambda p: p**gamma / (p**gamma + (1-p)**gamma)**(1/gamma) if 0 < p < 1 else p

def linear_weighting():
    """Linear (risk-neutral) weighting - recovers standard expectation."""
    return lambda p: p


# CPT Utility Function Presets
def create_standard_cpt_agent(env, **kwargs):
    """Create CPT agent with standard Tversky & Kahneman (1992) parameters."""
    return CPTValueIteration(
        env=env,
        utility_alpha=0.88,
        utility_beta=0.88, 
        loss_aversion_lambda=2.25,
        reference_point=0.0,
        use_utility_function=True,
        **kwargs
    )

def create_choquet_only_agent(env, **kwargs):
    """Create agent with only Choquet expectation (no utility function)."""
    return CPTValueIteration(
        env=env,
        use_utility_function=False,
        **kwargs
    )

def create_high_loss_aversion_agent(env, **kwargs):
    """Create CPT agent with high loss aversion (λ=5.0)."""
    return CPTValueIteration(
        env=env,
        utility_alpha=0.88,
        utility_beta=0.88,
        loss_aversion_lambda=5.0,
        reference_point=0.0,
        use_utility_function=True,
        **kwargs
    )

def create_risk_seeking_agent(env, **kwargs):
    """Create agent with risk-seeking utility (α, β > 1)."""
    return CPTValueIteration(
        env=env,
        utility_alpha=1.2,
        utility_beta=1.2,
        loss_aversion_lambda=2.25,
        reference_point=0.0,
        use_utility_function=True,
        **kwargs
    )