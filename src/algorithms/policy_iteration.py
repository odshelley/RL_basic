"""
Policy Iteration Algorithm for the Stochastic Gridworld

Policy Iteration alternates between:
1. Policy Evaluation: Compute value function V^π for current policy π
2. Policy Improvement: Update policy π to be greedy with respect to V^π

The algorithm converges to the optimal policy π* and optimal value function V*.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import sys
import os

# Add the src directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gridworld.environment import StochasticGridworld, Action


class PolicyIteration:
    def __init__(self, env: StochasticGridworld, theta: float = 1e-6):
        """
        Initialize Policy Iteration algorithm.
        
        Args:
            env: The gridworld environment
            theta: Threshold for policy evaluation convergence
        """
        self.env = env
        self.theta = theta
        
        # Get all states and actions
        self.states = env.get_all_states()
        self.actions = env.get_all_actions()
        self.num_states = len(self.states)
        self.num_actions = len(self.actions)
        
        # Initialize value function and policy
        self.V = {state: 0.0 for state in self.states}
        
        # Initialize with random policy (uniform over actions)
        self.policy = {}
        for state in self.states:
            if env.is_terminal(state):
                self.policy[state] = None
            else:
                self.policy[state] = np.random.choice(self.actions)
        
        # Track convergence
        self.iteration_count = 0
        self.policy_stable = False
        self.value_history = []
        self.policy_history = []
    
    def policy_evaluation(self) -> int:
        """
        Evaluate the current policy using iterative policy evaluation.
        
        Returns:
            Number of iterations until convergence
        """
        iterations = 0
        
        while True:
            delta = 0
            old_V = self.V.copy()
            
            for state in self.states:
                if self.env.is_terminal(state):
                    continue
                
                action = self.policy[state]
                if action is None:
                    continue
                
                # Compute expected value under current policy
                v = 0
                transitions = self.env.get_transition_probabilities(state, action)
                
                for next_state, prob in transitions.items():
                    reward = self.env.get_reward(state, action, next_state)
                    v += prob * (reward + self.env.gamma * old_V[next_state])
                
                self.V[state] = v
                delta = max(delta, abs(old_V[state] - self.V[state]))
            
            iterations += 1
            
            if delta < self.theta:
                break
        
        return iterations
    
    def policy_improvement(self) -> bool:
        """
        Improve the policy by making it greedy with respect to the current value function.
        
        Returns:
            True if policy changed, False if policy is stable
        """
        policy_stable = True
        
        for state in self.states:
            if self.env.is_terminal(state):
                continue
            
            old_action = self.policy[state]
            
            # Find best action for this state
            action_values = {}
            for action in self.actions:
                value = 0
                transitions = self.env.get_transition_probabilities(state, action)
                
                for next_state, prob in transitions.items():
                    reward = self.env.get_reward(state, action, next_state)
                    value += prob * (reward + self.env.gamma * self.V[next_state])
                
                action_values[action] = value
            
            # Choose action with highest value (break ties randomly)
            max_value = max(action_values.values())
            best_actions = [action for action, value in action_values.items() 
                          if abs(value - max_value) < 1e-10]
            
            self.policy[state] = np.random.choice(best_actions)
            
            if self.policy[state] != old_action:
                policy_stable = False
        
        return policy_stable
    
    def run(self, max_iterations: int = 1000, verbose: bool = True) -> Dict:
        """
        Run the complete Policy Iteration algorithm.
        
        Args:
            max_iterations: Maximum number of policy improvement iterations
            verbose: Whether to print progress
            
        Returns:
            Dictionary with results
        """
        if verbose:
            print("Starting Policy Iteration...")
            print(f"Environment: {self.env.size}x{self.env.size} grid")
            print(f"States: {self.num_states}, Actions: {self.num_actions}")
            print(f"Discount factor: {self.env.gamma}")
            print()
        
        policy_changes = 0
        total_eval_iterations = 0
        
        for iteration in range(max_iterations):
            # Store current state for history
            self.value_history.append(self.V.copy())
            self.policy_history.append(self.policy.copy())
            
            # Policy Evaluation
            eval_iterations = self.policy_evaluation()
            total_eval_iterations += eval_iterations
            
            # Policy Improvement
            policy_stable = self.policy_improvement()
            
            if verbose:
                print(f"Iteration {iteration + 1}:")
                print(f"  Policy evaluation converged in {eval_iterations} iterations")
                print(f"  Policy stable: {policy_stable}")
                if not policy_stable:
                    policy_changes += 1
                print()
            
            if policy_stable:
                self.policy_stable = True
                self.iteration_count = iteration + 1
                
                if verbose:
                    print(f"Policy Iteration converged in {iteration + 1} iterations!")
                    print(f"Total policy evaluation iterations: {total_eval_iterations}")
                    print(f"Total policy changes: {policy_changes}")
                
                break
        else:
            if verbose:
                print(f"Policy Iteration did not converge in {max_iterations} iterations")
        
        # Final evaluation
        final_eval_iterations = self.policy_evaluation()
        total_eval_iterations += final_eval_iterations
        
        return {
            'converged': self.policy_stable,
            'iterations': self.iteration_count,
            'total_eval_iterations': total_eval_iterations,
            'policy_changes': policy_changes,
            'final_value_function': self.V.copy(),
            'final_policy': self.policy.copy(),
            'value_history': self.value_history,
            'policy_history': self.policy_history
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
            value = 0
            transitions = self.env.get_transition_probabilities(state, action)
            
            for next_state, prob in transitions.items():
                reward = self.env.get_reward(state, action, next_state)
                value += prob * (reward + self.env.gamma * self.V[next_state])
            
            action_values[action] = value
        
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
