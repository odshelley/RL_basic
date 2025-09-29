"""
Gymnasium wrapper for the Stochastic Gridworld environment.

This wrapper converts our custom StochasticGridworld to be compatible with
Gymnasium (gym) interface, enabling integration with modern RL libraries
like Stable-Baselines3 and standard RL tooling.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os
from typing import Tuple, Dict, Any, Optional

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gridworld.environment import StochasticGridworld, Action


class GymnasiumGridworldWrapper(gym.Env):
    """
    Gymnasium wrapper for StochasticGridworld environment.
    
    Converts our custom environment to standard Gymnasium interface:
    - Observation space: Box or Discrete
    - Action space: Discrete (4 actions)
    - Standard reset() and step() methods
    - Proper info dictionaries
    """
    
    def __init__(self, size: int = 5, gamma: float = 0.99, max_steps: int = 200,
                 observation_type: str = "coordinates"):
        """
        Initialize the Gymnasium wrapper.
        
        Args:
            size: Grid size (size x size)
            gamma: Discount factor
            max_steps: Maximum steps per episode
            observation_type: "coordinates", "one_hot", or "grid"
        """
        super().__init__()
        
        # Create the underlying gridworld
        self.env = StochasticGridworld(size=size, gamma=gamma, max_steps=max_steps)
        self.observation_type = observation_type
        
        # Define action space: 4 discrete actions
        self.action_space = spaces.Discrete(4)
        
        # Define observation space based on observation_type
        if observation_type == "coordinates":
            # Simple (x, y) coordinates
            self.observation_space = spaces.Box(
                low=0, high=size-1, shape=(2,), dtype=np.float32
            )
        elif observation_type == "one_hot":
            # One-hot encoding of grid position
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(size * size,), dtype=np.float32
            )
        elif observation_type == "grid":
            # Full grid representation with agent position, goal, pits
            self.observation_space = spaces.Box(
                low=-1, high=2, shape=(size, size), dtype=np.float32
            )
        else:
            raise ValueError(f"Unknown observation_type: {observation_type}")
        
        # Action mapping: gym action (int) -> our Action enum
        self.action_mapping = {
            0: Action.UP,
            1: Action.DOWN, 
            2: Action.LEFT,
            3: Action.RIGHT
        }
        
        # Reverse mapping: our Action enum -> gym action (int)
        self.reverse_action_mapping = {v: k for k, v in self.action_mapping.items()}
        
        # Episode tracking
        self.episode_steps = 0
        self.episode_return = 0.0
    
    def _get_observation(self, state: Tuple[int, int]) -> np.ndarray:
        """
        Convert state to observation based on observation_type.
        
        Args:
            state: (row, col) position in grid
            
        Returns:
            Observation as numpy array
        """
        row, col = state
        
        if self.observation_type == "coordinates":
            # Simple (x, y) coordinates
            return np.array([row, col], dtype=np.float32)
        
        elif self.observation_type == "one_hot":
            # One-hot encoding of position
            obs = np.zeros(self.env.size * self.env.size, dtype=np.float32)
            idx = row * self.env.size + col
            obs[idx] = 1.0
            return obs
        
        elif self.observation_type == "grid":
            # Full grid representation
            obs = np.zeros((self.env.size, self.env.size), dtype=np.float32)
            
            # Mark pits as -1
            for pit_row, pit_col in self.env.pit_states:
                obs[pit_row, pit_col] = -1.0
            
            # Mark goal as 2
            goal_row, goal_col = self.env.goal_state
            obs[goal_row, goal_col] = 2.0
            
            # Mark agent position as 1
            obs[row, col] = 1.0
            
            return obs
        
        else:
            raise ValueError(f"Unknown observation_type: {self.observation_type}")
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary for current state."""
        current_state = self.env.current_state
        
        return {
            'state': current_state,
            'is_goal': current_state == self.env.goal_state,
            'is_pit': current_state in self.env.pit_states,
            'episode_steps': self.episode_steps,
            'episode_return': self.episode_return,
            'max_steps_reached': self.episode_steps >= self.env.max_steps
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Reset underlying environment
        state = self.env.reset()
        
        # Reset episode tracking
        self.episode_steps = 0
        self.episode_return = 0.0
        
        # Get observation and info
        observation = self._get_observation(state)
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (0-3)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert gym action to our Action enum
        if action not in self.action_mapping:
            raise ValueError(f"Invalid action: {action}. Must be 0-3.")
        
        our_action = self.action_mapping[action]
        
        # Take step in underlying environment
        next_state, reward, done, env_info = self.env.step(our_action)
        
        # Update episode tracking
        self.episode_steps += 1
        self.episode_return += reward
        
        # Get observation
        observation = self._get_observation(next_state)
        
        # Determine termination conditions
        terminated = self.env.is_terminal(next_state)  # Goal or pit reached
        truncated = self.episode_steps >= self.env.max_steps  # Max steps reached
        
        # Create info dictionary
        info = self._get_info()
        info.update(env_info)  # Add any info from underlying environment
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode: str = "human") -> Optional[str]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
            
        Returns:
            String representation if mode is "ansi"
        """
        if mode == "human":
            print(self.env.render())
            return None
        elif mode == "ansi":
            return self.env.render()
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def close(self):
        """Close the environment."""
        pass  # Nothing to clean up for our simple environment
    
    @property
    def unwrapped(self):
        """Access to the underlying environment."""
        return self.env


# Convenience function to create the wrapped environment
def make_gridworld_env(size: int = 5, gamma: float = 0.99, max_steps: int = 200,
                      observation_type: str = "coordinates") -> GymnasiumGridworldWrapper:
    """
    Create a Gymnasium-wrapped gridworld environment.
    
    Args:
        size: Grid size
        gamma: Discount factor
        max_steps: Maximum steps per episode
        observation_type: Type of observation ("coordinates", "one_hot", "grid")
        
    Returns:
        Wrapped environment
    """
    return GymnasiumGridworldWrapper(
        size=size, gamma=gamma, max_steps=max_steps, observation_type=observation_type
    )
