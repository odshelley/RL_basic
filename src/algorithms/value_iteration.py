"""
Value Iteration Algorithm for the Stochastic Gridworld

Value Iteration directly computes the optimal value function V* by iteratively applying
the Bellman optimality equation:

V(s) = max_a Σ_s' P(s'|s,a) [R(s,a,s') + γV(s')]

Once V* is found, the optimal policy is extracted by choosing the greedy action:
π*(s) = argmax_a Σ_s' P(s'|s,a) [R(s,a,s') + γV(s')]

Value Iteration is often more efficient than Policy Iteration for many problems.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import sys
import os

# Add the src directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gridworld.environment import StochasticGridworld, Action


class ValueIteration:
    def __init__(self, env: StochasticGridworld, theta: float = 1e-6):
        """
        Initialize Value Iteration algorithm.
        
        Args:
            env: The gridworld environment
            theta: Threshold for value function convergence
        """
        self.env = env
        self.theta = theta
        
        # Get all states and actions
        self.states = env.get_all_states()
        self.actions = env.get_all_actions()
        self.num_states = len(self.states)
        self.num_actions = len(self.actions)
        
        # Initialize value function
        self.V = {state: 0.0 for state in self.states}
        
        # Policy will be extracted after value iteration converges
        self.policy = {}
        
        # Track convergence
        self.iteration_count = 0
        self.converged = False
        self.value_history = []
    
    def compute_action_value(self, state: Tuple[int, int], action: Action, 
                           value_function: Dict[Tuple[int, int], float]) -> float:
        """
        Compute Q-value for a specific state-action pair.
        
        Args:
            state: Current state
            action: Action to evaluate
            value_function: Current value function estimate
            
        Returns:
            Expected return for taking action in state
        """
        if self.env.is_terminal(state):
            return 0.0
        
        q_value = 0.0
        transitions = self.env.get_transition_probabilities(state, action)
        
        for next_state, prob in transitions.items():
            reward = self.env.get_reward(state, action, next_state)
            q_value += prob * (reward + self.env.gamma * value_function[next_state])
        
        return q_value
    
    def value_iteration_step(self) -> float:
        """
        Perform one step of value iteration.
        
        Returns:
            Maximum change in value function (delta)
        """
        delta = 0.0
        old_V = self.V.copy()
        
        for state in self.states:
            if self.env.is_terminal(state):
                continue
             
            # Compute action values for all actions
            action_values = []
            for action in self.actions:
                q_value = self.compute_action_value(state, action, old_V)
                action_values.append(q_value)
            
            # Update value function with maximum action value
            new_value = max(action_values)
            self.V[state] = new_value
            
            # Track maximum change
            delta = max(delta, abs(old_V[state] - new_value))
        
        return delta
    
    def extract_policy(self) -> Dict[Tuple[int, int], Optional[Action]]:
        """
        Extract the greedy policy from the current value function.
        
        Returns:
            Policy mapping states to actions
        """
        policy = {}
        
        for state in self.states:
            if self.env.is_terminal(state):
                policy[state] = None
                continue
            
            # Find best action for this state
            best_action = None
            best_value = float('-inf')
            
            for action in self.actions:
                q_value = self.compute_action_value(state, action, self.V)
                
                if q_value > best_value:
                    best_value = q_value
                    best_action = action
            
            policy[state] = best_action
        
        return policy
    
    def run(self, max_iterations: int = 1000, verbose: bool = True) -> Dict:
        """
        Run the complete Value Iteration algorithm.
        
        Args:
            max_iterations: Maximum number of iterations
            verbose: Whether to print progress
            
        Returns:
            Dictionary with results
        """
        if verbose:
            print("Starting Value Iteration...")
            print(f"Environment: {self.env.size}x{self.env.size} grid")
            print(f"States: {self.num_states}, Actions: {self.num_actions}")
            print(f"Discount factor: {self.env.gamma}")
            print()
        
        for iteration in range(max_iterations):
            # Store current state for history
            self.value_history.append(self.V.copy())
            
            # Perform one value iteration step
            delta = self.value_iteration_step()
            
            if verbose:
                print(f"Iteration {iteration + 1}: max value change = {delta:.8f}")
            
            # Check for convergence
            if delta < self.theta:
                self.converged = True
                self.iteration_count = iteration + 1
                
                if verbose:
                    print(f"\nValue Iteration converged in {iteration + 1} iterations!")
                    print(f"Final max value change: {delta:.8f}")
                
                break
        else:
            if verbose:
                print(f"Value Iteration did not converge in {max_iterations} iterations")
        
        # Extract optimal policy
        self.policy = self.extract_policy()
        
        if verbose and self.converged:
            print("Optimal policy extracted successfully!")
        
        return {
            'converged': self.converged,
            'iterations': self.iteration_count,
            'final_delta': delta if 'delta' in locals() else float('inf'),
            'final_value_function': self.V.copy(),
            'final_policy': self.policy.copy(),
            'value_history': self.value_history
        }
    
    def get_action_values(self, state: Tuple[int, int]) -> Dict[Action, float]:
        """
        Get action values (Q-values) for a given state under current value function.
        
        Args:
            state: State to compute action values for
            
        Returns:
            Dictionary mapping action to its value
        """
        if self.env.is_terminal(state):
            return {}
        
        action_values = {}
        for action in self.actions:
            action_values[action] = self.compute_action_value(state, action, self.V)
        
        return action_values
    
    def get_state_value(self, state: Tuple[int, int]) -> float:
        """Get value function for a specific state."""
        return self.V.get(state, 0.0)
    
    def get_policy_action(self, state: Tuple[int, int]) -> Optional[Action]:
        """Get the policy action for a specific state."""
        return self.policy.get(state)
    
    def evaluate_policy(self, num_episodes: int = 1000) -> Dict:
        """
        Evaluate the learned policy by running episodes.
        
        Args:
            num_episodes: Number of episodes to run
            
        Returns:
            Dictionary with evaluation statistics
        """
        if not self.policy:
            raise ValueError("Policy not extracted yet. Run value iteration first.")
        
        returns = []
        episode_lengths = []
        success_count = 0
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_return = 0
            episode_length = 0
            
            while not self.env.terminal and episode_length < self.env.max_steps:
                action = self.get_policy_action(state)
                if action is None:
                    break
                
                next_state, reward, done, _ = self.env.step(action)
                episode_return += reward
                episode_length += 1
                state = next_state
                
                if done:
                    if state == self.env.goal_state:
                        success_count += 1
                    break
            
            returns.append(episode_return)
            episode_lengths.append(episode_length)
        
        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': success_count / num_episodes,
            'returns': returns,
            'lengths': episode_lengths
        }
