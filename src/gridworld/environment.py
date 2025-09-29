"""
Stochastic Gridworld Environment

A 5x5 gridworld with:
- Start state: (4,0) (bottom-left)
- Goal: (0,4) (top-right), terminal reward +1.0
- Pit (cliff): top row except goal (0,0)...(0,3), terminal reward -1.0
- Actions: up, down, left, right
- Stochastic dynamics: 80% intended, 10% left, 10% right
- Step reward: -0.01 per move
- Discount: γ=0.99
- Episode end: hitting Goal/Pit or after 200 steps
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from enum import Enum


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class StochasticGridworld:
    def __init__(self, size: int = 5, gamma: float = 0.99, max_steps: int = 200):
        """
        Initialize the stochastic gridworld environment.
        
        Args:
            size: Grid size (default 5x5)
            gamma: Discount factor
            max_steps: Maximum steps per episode
        """
        self.size = size
        self.gamma = gamma
        self.max_steps = max_steps
        
        # State definitions
        self.start_state = (4, 0)  # Bottom-left
        self.goal_state = (0, 4)   # Top-right
        self.pit_states = [(0, j) for j in range(4)]  # Top row except goal
        
        # Rewards
        self.goal_reward = 1.0
        self.pit_reward = -1.0
        self.step_reward = -0.01
        
        # Transition probabilities
        self.intended_prob = 0.8
        self.slip_prob = 0.1  # Each of left/right slip
        
        # Action mappings
        self.action_deltas = {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1)
        }
        
        # Initialize state
        self.current_state = self.start_state
        self.step_count = 0
        self.terminal = False
        
    def get_all_states(self) -> List[Tuple[int, int]]:
        """Return all possible states in the grid."""
        return [(i, j) for i in range(self.size) for j in range(self.size)]
    
    def get_all_actions(self) -> List[Action]:
        """Return all possible actions."""
        return list(Action)
    
    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """Check if a state is terminal."""
        return state == self.goal_state or state in self.pit_states
    
    def is_valid_state(self, state: Tuple[int, int]) -> bool:
        """Check if a state is within the grid bounds."""
        i, j = state
        return 0 <= i < self.size and 0 <= j < self.size
    
    def get_reward(self, state: Tuple[int, int], action: Action, next_state: Tuple[int, int]) -> float:
        """Get reward for transitioning from state to next_state via action."""
        if next_state == self.goal_state:
            return self.goal_reward
        elif next_state in self.pit_states:
            return self.pit_reward
        else:
            return self.step_reward
    
    def get_next_state(self, state: Tuple[int, int], action: Action) -> Tuple[int, int]:
        """Get next state given current state and action (deterministic version)."""
        if self.is_terminal(state):
            return state
        
        delta = self.action_deltas[action]
        next_state = (state[0] + delta[0], state[1] + delta[1])
        
        # If next state is out of bounds, stay in current state
        if not self.is_valid_state(next_state):
            return state
        
        return next_state
    
    def get_slip_actions(self, action: Action) -> List[Action]:
        """Get the left and right slip actions for a given action."""
        action_list = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]
        action_idx = action_list.index(action)
        
        left_slip = action_list[(action_idx - 1) % 4]
        right_slip = action_list[(action_idx + 1) % 4]
        
        return [left_slip, right_slip]
    
    def get_transition_probabilities(self, state: Tuple[int, int], action: Action) -> Dict[Tuple[int, int], float]:
        """
        Get transition probabilities for all possible next states given current state and action.
        
        Returns:
            Dictionary mapping next_state -> probability
        """
        if self.is_terminal(state):
            return {state: 1.0}
        
        transitions = {}
        
        # Intended action
        intended_next = self.get_next_state(state, action)
        transitions[intended_next] = transitions.get(intended_next, 0) + self.intended_prob
        
        # Slip actions
        left_slip, right_slip = self.get_slip_actions(action)
        
        left_next = self.get_next_state(state, left_slip)
        transitions[left_next] = transitions.get(left_next, 0) + self.slip_prob
        
        right_next = self.get_next_state(state, right_slip)
        transitions[right_next] = transitions.get(right_next, 0) + self.slip_prob
        
        return transitions
    
    def reset(self) -> Tuple[int, int]:
        """Reset the environment to initial state."""
        self.current_state = self.start_state
        self.step_count = 0
        self.terminal = False
        return self.current_state
    
    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            (next_state, reward, done, info)
        """
        if self.terminal:
            return self.current_state, 0, True, {"message": "Episode already terminated"}
        
        # Sample next state based on transition probabilities
        transitions = self.get_transition_probabilities(self.current_state, action)
        states = list(transitions.keys())
        probs = list(transitions.values())
        
        next_state = states[np.random.choice(len(states), p=probs)]
        reward = self.get_reward(self.current_state, action, next_state)
        
        self.current_state = next_state
        self.step_count += 1
        
        # Check terminal conditions
        done = (self.is_terminal(next_state) or self.step_count >= self.max_steps)
        self.terminal = done
        
        info = {
            "step_count": self.step_count,
            "max_steps_reached": self.step_count >= self.max_steps
        }
        
        return next_state, reward, done, info
    
    def render(self) -> str:
        """Render the current state of the gridworld."""
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        
        # Mark special states
        grid[self.goal_state[0]][self.goal_state[1]] = 'G'
        for pit in self.pit_states:
            grid[pit[0]][pit[1]] = 'P'
        grid[self.start_state[0]][self.start_state[1]] = 'S'
        
        # Mark current position (if different from start)
        if self.current_state != self.start_state:
            if not self.is_terminal(self.current_state):
                grid[self.current_state[0]][self.current_state[1]] = 'X'
        
        # Create string representation
        result = []
        for row in grid:
            result.append(' '.join(row))
        
        return '\n'.join(result)
    
    def get_state_index(self, state: Tuple[int, int]) -> int:
        """Convert 2D state to 1D index."""
        return state[0] * self.size + state[1]
    
    def get_state_from_index(self, index: int) -> Tuple[int, int]:
        """Convert 1D index to 2D state."""
        return (index // self.size, index % self.size)
