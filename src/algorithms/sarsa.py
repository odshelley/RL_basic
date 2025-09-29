"""
SARSA (TD(0)) Algorithm for the Stochastic Gridworld

SARSA is a model-free, on-policy temporal difference learning algorithm.
It learns Q-values through experience using the update rule:

Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]

Where:
- α is the learning rate
- γ is the discount factor  
- The quintuple (s,a,r,s',a') gives the algorithm its name: State-Action-Reward-State-Action

SARSA is on-policy because it updates Q-values using the action that would actually
be taken in the next state according to the current policy (usually ε-greedy).
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import sys
import os
import random

# Add the src directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gridworld.environment import StochasticGridworld, Action


class SARSA:
    def __init__(self, env: StochasticGridworld, alpha: float = 0.1, epsilon: float = 0.1, 
                 epsilon_decay: float = 0.995, min_epsilon: float = 0.01):
        """
        Initialize SARSA algorithm.
        
        Args:
            env: The gridworld environment
            alpha: Learning rate (step size)
            epsilon: Initial exploration rate for ε-greedy policy
            epsilon_decay: Decay rate for epsilon
            min_epsilon: Minimum epsilon value
        """
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.initial_epsilon = epsilon
        
        # Get all states and actions
        self.states = env.get_all_states()
        self.actions = env.get_all_actions()
        self.num_states = len(self.states)
        self.num_actions = len(self.actions)
        
        # Initialize Q-table with small random values to break ties
        self.Q = {}
        for state in self.states:
            self.Q[state] = {}
            if self.env.is_terminal(state):
                # Terminal states have zero Q-values for all actions
                for action in self.actions:
                    self.Q[state][action] = 0.0
            else:
                # Non-terminal states get small random initialization
                for action in self.actions:
                    self.Q[state][action] = np.random.normal(0, 0.01)
        
        # Track learning progress
        self.episode_returns = []
        self.episode_lengths = []
        self.q_value_history = []
        self.epsilon_history = []
    
    def get_epsilon_greedy_action(self, state: Tuple[int, int]) -> Action:
        """
        Choose action using ε-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            Action chosen by ε-greedy policy
        """
        if self.env.is_terminal(state):
            return None
            
        if random.random() < self.epsilon:
            # Explore: choose random action
            return random.choice(self.actions)
        else:
            # Exploit: choose greedy action
            return self.get_greedy_action(state)
    
    def get_greedy_action(self, state: Tuple[int, int]) -> Action:
        """
        Choose the greedy action (highest Q-value) for a state.
        
        Args:
            state: Current state
            
        Returns:
            Action with highest Q-value
        """
        if self.env.is_terminal(state):
            return None
        
        # Ensure state exists in Q-table
        if state not in self.Q:
            # Initialize Q-values for this state if missing
            self.Q[state] = {action: np.random.normal(0, 0.01) for action in self.actions}
            
        # Find action with maximum Q-value
        max_q = max(self.Q[state].values())
        # Handle ties by randomly choosing among best actions
        best_actions = [action for action, q_val in self.Q[state].items() if q_val == max_q]
        return random.choice(best_actions)
    
    def update_q_value(self, state: Tuple[int, int], action: Action, reward: float,
                       next_state: Tuple[int, int], next_action: Action):
        """
        Update Q-value using SARSA update rule.
        
        Args:
            state: Current state
            action: Current action
            reward: Reward received
            next_state: Next state
            next_action: Next action (chosen by current policy)
        """
        # Ensure states exist in Q-table
        if state not in self.Q:
            self.Q[state] = {a: np.random.normal(0, 0.01) for a in self.actions}
        if next_state not in self.Q:
            if self.env.is_terminal(next_state):
                self.Q[next_state] = {a: 0.0 for a in self.actions}
            else:
                self.Q[next_state] = {a: np.random.normal(0, 0.01) for a in self.actions}
            
        current_q = self.Q[state][action]
        
        if self.env.is_terminal(next_state) or next_action is None:
            # Terminal state: next Q-value is 0
            target = reward
        else:
            # Non-terminal: use SARSA update with next Q-value
            next_q = self.Q[next_state][next_action]
            target = reward + self.env.gamma * next_q
        
        # SARSA update rule
        self.Q[state][action] = current_q + self.alpha * (target - current_q)
    
    def run_episode(self, verbose: bool = False) -> Tuple[float, int]:
        """
        Run a single episode using SARSA learning.
        
        Args:
            verbose: Whether to print episode details
            
        Returns:
            Tuple of (episode_return, episode_length)
        """
        state = self.env.reset()
        action = self.get_epsilon_greedy_action(state)
        
        episode_return = 0.0
        episode_length = 0
        
        if verbose:
            print(f"Episode start: {state}")
        
        while not self.env.terminal and episode_length < self.env.max_steps:
            if action is None:
                break
                
            # Take action and observe outcome
            next_state, reward, done, info = self.env.step(action)
            
            # Choose next action using current policy
            next_action = self.get_epsilon_greedy_action(next_state)
            
            # Update Q-value using SARSA rule
            self.update_q_value(state, action, reward, next_state, next_action)
            
            # Update episode statistics
            episode_return += reward
            episode_length += 1
            
            if verbose:
                print(f"  Step {episode_length}: {state} --{action.name}--> {next_state} "
                      f"(reward: {reward:.3f}, Q: {self.Q[state][action]:.6f})")
            
            # Move to next state-action pair
            state = next_state
            action = next_action
            
            if done:
                break
        
        if verbose:
            outcome = "SUCCESS" if state == self.env.goal_state else "FAILURE"
            print(f"  Episode end: {state} - {outcome}")
            print(f"  Return: {episode_return:.6f}, Length: {episode_length}")
        
        return episode_return, episode_length
    
    def run(self, num_episodes: int = 1000, verbose: bool = False, 
            save_frequency: int = 100) -> Dict:
        """
        Run SARSA learning for multiple episodes.
        
        Args:
            num_episodes: Number of episodes to run
            verbose: Whether to print detailed progress
            save_frequency: How often to save Q-value snapshots
            
        Returns:
            Dictionary with training results
        """
        if verbose:
            print("Starting SARSA Learning...")
            print(f"Environment: {self.env.size}x{self.env.size} grid")
            print(f"Episodes: {num_episodes}")
            print(f"Learning rate (α): {self.alpha}")
            print(f"Initial exploration (ε): {self.initial_epsilon}")
            print(f"Epsilon decay: {self.epsilon_decay}")
            print(f"Min epsilon: {self.min_epsilon}")
            print(f"Discount factor (γ): {self.env.gamma}")
            print()
        
        self.episode_returns = []
        self.episode_lengths = []
        self.q_value_history = []
        self.epsilon_history = []
        
        for episode in range(num_episodes):
            # Run episode
            episode_return, episode_length = self.run_episode(verbose=False)
            
            # Store results
            self.episode_returns.append(episode_return)
            self.episode_lengths.append(episode_length)
            self.epsilon_history.append(self.epsilon)
            
            # Save Q-value snapshot periodically
            if episode % save_frequency == 0:
                avg_return = np.mean(self.episode_returns[-save_frequency:]) if self.episode_returns else 0
                if verbose:
                    print(f"Episode {episode:4d}: Avg Return = {avg_return:7.4f}, "
                          f"Epsilon = {self.epsilon:.4f}")
                
                # Save Q-value snapshot for key states
                snapshot = {}
                key_states = [self.env.start_state, (2, 2), (1, 3), (3, 1)]
                for state in key_states:
                    if state in self.Q and not self.env.is_terminal(state):
                        snapshot[state] = dict(self.Q[state])
                self.q_value_history.append(snapshot)
            
            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Final evaluation
        final_policy_return = self.evaluate_policy(num_episodes=1000, use_greedy=True)
        
        if verbose:
            print(f"\nSARSA Learning completed!")
            print(f"Final epsilon: {self.epsilon:.6f}")
            print(f"Final policy evaluation: {final_policy_return['mean_return']:.6f} ± {final_policy_return['std_return']:.6f}")
            print(f"Success rate: {final_policy_return['success_rate']:.3f}")
        
        return {
            'episode_returns': self.episode_returns,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history,
            'q_value_history': self.q_value_history,
            'final_q_table': dict(self.Q),
            'final_evaluation': final_policy_return,
            'converged': True,  # SARSA doesn't have a clear convergence criterion
            'episodes': num_episodes
        }
    
    def get_policy_action(self, state: Tuple[int, int]) -> Optional[Action]:
        """
        Get the action prescribed by the current greedy policy.
        
        Args:
            state: State to get action for
            
        Returns:
            Best action according to learned Q-values, or None for terminal states
        """
        return self.get_greedy_action(state)
    
    def get_state_value(self, state: Tuple[int, int]) -> float:
        """
        Get the state value V(s) = max_a Q(s,a).
        
        Args:
            state: State to get value for
            
        Returns:
            Maximum Q-value for the state
        """
        if self.env.is_terminal(state):
            return 0.0
        if state not in self.Q:
            # Initialize Q-values for this state if missing
            self.Q[state] = {action: np.random.normal(0, 0.01) for action in self.actions}
        return max(self.Q[state].values())
    
    def get_action_values(self, state: Tuple[int, int]) -> Dict[Action, float]:
        """
        Get all Q-values Q(s,a) for a given state.
        
        Args:
            state: State to get action values for
            
        Returns:
            Dictionary mapping actions to Q-values
        """
        if state not in self.Q:
            if self.env.is_terminal(state):
                self.Q[state] = {action: 0.0 for action in self.actions}
            else:
                self.Q[state] = {action: np.random.normal(0, 0.01) for action in self.actions}
        return dict(self.Q[state])
    
    def evaluate_policy(self, num_episodes: int = 1000, 
                       use_greedy: bool = True) -> Dict[str, float]:
        """
        Evaluate the current policy by running episodes.
        
        Args:
            num_episodes: Number of episodes to run for evaluation
            use_greedy: If True, use greedy policy; if False, use ε-greedy
            
        Returns:
            Dictionary with evaluation metrics
        """
        returns = []
        lengths = []
        successes = 0
        
        # Temporarily store current epsilon if using greedy policy
        old_epsilon = self.epsilon
        if use_greedy:
            self.epsilon = 0.0  # Greedy policy
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_return = 0.0
            episode_length = 0
            
            while not self.env.terminal and episode_length < self.env.max_steps:
                action = self.get_epsilon_greedy_action(state)
                if action is None:
                    break
                    
                state, reward, done, info = self.env.step(action)
                episode_return += reward
                episode_length += 1
                
                if done:
                    break
            
            returns.append(episode_return)
            lengths.append(episode_length)
            if state == self.env.goal_state:
                successes += 1
        
        # Restore epsilon
        self.epsilon = old_epsilon
        
        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'success_rate': successes / num_episodes,
            'returns': returns,
            'lengths': lengths
        }
