"""
REINFORCE (Monte Carlo Policy Gradient) algorithm implementation.

This module implements the REINFORCE algorithm using function approximation
with neural networks, integrating with the Gymnasium-wrapped gridworld environment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.policy_networks import PolicyNetwork
from algorithms.gymnasium_wrapper import make_gridworld_env


@dataclass
class REINFORCEConfig:
    """Configuration for REINFORCE algorithm."""
    # Network architecture
    hidden_dims: Tuple[int, ...] = (64, 32)
    activation: str = "relu"
    
    # Training parameters
    learning_rate: float = 0.001
    gamma: float = 0.99
    
    # Environment parameters
    grid_size: int = 5
    max_steps_per_episode: int = 200
    observation_type: str = "coordinates"  # "coordinates", "one_hot", "grid"
    
    # Training control
    num_episodes: int = 2000
    log_interval: int = 100
    save_interval: int = 500
    
    # Optimization
    optimizer: str = "adam"  # "adam", "sgd", "rmsprop"
    weight_decay: float = 0.0
    gradient_clip: Optional[float] = None
    
    # Logging
    log_dir: str = "logs/reinforce"
    save_model: bool = True


class REINFORCEAgent:
    """
    REINFORCE (Monte Carlo Policy Gradient) agent with function approximation.
    
    Uses a policy network to learn action probabilities directly through
    policy gradient methods. Updates are performed using complete episode
    trajectories (Monte Carlo).
    """
    
    def __init__(self, config: REINFORCEConfig):
        """
        Initialize the REINFORCE agent.
        
        Args:
            config: Configuration for the agent
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create environment
        self.env = make_gridworld_env(
            size=config.grid_size,
            gamma=config.gamma,
            max_steps=config.max_steps_per_episode,
            observation_type=config.observation_type
        )
        
        # Determine input dimension based on observation type
        if config.observation_type == "coordinates":
            input_dim = 2
        elif config.observation_type == "one_hot":
            input_dim = config.grid_size * config.grid_size
        elif config.observation_type == "grid":
            input_dim = config.grid_size * config.grid_size
        else:
            raise ValueError(f"Unknown observation_type: {config.observation_type}")
        
        # Create policy network
        self.policy_net = PolicyNetwork(
            input_dim=input_dim,
            hidden_dims=config.hidden_dims,
            num_actions=self.env.action_space.n,
            activation=config.activation
        ).to(self.device)
        
        # Create optimizer
        if config.optimizer == "adam":
            self.optimizer = optim.Adam(
                self.policy_net.parameters(), 
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.policy_net.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(
                self.policy_net.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")
        
        # Logging setup
        self.writer = None
        if config.log_dir:
            os.makedirs(config.log_dir, exist_ok=True)
            self.writer = SummaryWriter(config.log_dir)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        
        print(f"REINFORCE agent initialized:")
        print(f"  Policy network: {sum(p.numel() for p in self.policy_net.parameters())} parameters")
        print(f"  Input dimension: {input_dim}")
        print(f"  Observation type: {config.observation_type}")
    
    def select_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Select action using the policy network.
        
        Args:
            state: Current state observation
            
        Returns:
            Tuple of (action, log_probability)
        """
        # Don't use torch.no_grad() here - we need gradients for policy updates!
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Move to device
        state = state.to(self.device)
        
        # Get action probabilities
        action_probs = self.policy_net.get_action_probabilities(state)
        
        # Sample action
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.squeeze()
    
    def collect_episode(self) -> Tuple[List[torch.Tensor], List[int], List[float], List[torch.Tensor]]:
        """
        Collect a complete episode trajectory.
        
        Returns:
            Tuple of (states, actions, rewards, log_probs)
        """
        states, actions, rewards, log_probs = [], [], [], []
        
        # Reset environment
        obs, info = self.env.reset()
        state = torch.FloatTensor(obs.flatten() if obs.ndim > 1 else obs)
        
        done = False
        while not done:
            # Select action
            action, log_prob = self.select_action(state)
            
            # Store current transition
            states.append(state.clone())
            actions.append(action)
            log_probs.append(log_prob)
            
            # Take action in environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            rewards.append(reward)
            
            # Update state
            state = torch.FloatTensor(next_obs.flatten() if next_obs.ndim > 1 else next_obs)
            done = terminated or truncated
        
        return states, actions, rewards, log_probs
    
    def compute_returns(self, rewards: List[float]) -> List[float]:
        """
        Compute discounted returns for each timestep.
        
        Args:
            rewards: List of rewards from episode
            
        Returns:
            List of discounted returns
        """
        returns = []
        running_return = 0.0
        
        # Compute returns in reverse order (from end to beginning)
        for reward in reversed(rewards):
            running_return = reward + self.config.gamma * running_return
            returns.insert(0, running_return)
        
        return returns
    
    def update_policy(self, states: List[torch.Tensor], actions: List[int], 
                     returns: List[float], log_probs: List[torch.Tensor]) -> float:
        """
        Update the policy network using REINFORCE.
        
        Args:
            states: Episode states
            actions: Episode actions
            returns: Episode returns
            log_probs: Episode log probabilities
            
        Returns:
            Policy loss value
        """
        # Convert to tensors
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns for better numerical stability
        if len(returns) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        
        # Compute policy loss
        policy_loss = []
        for log_prob, return_val in zip(log_probs, returns_tensor):
            # REINFORCE loss: -log_prob * return
            policy_loss.append(-log_prob * return_val)
        
        policy_loss = torch.stack(policy_loss).mean()  # Use mean instead of sum
        
        # Optimization step
        self.optimizer.zero_grad()
        policy_loss.backward()
        
        # Gradient clipping if specified
        if self.config.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.gradient_clip)
        
        self.optimizer.step()
        
        return policy_loss.item()
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the REINFORCE agent.
        
        Returns:
            Dictionary of training statistics
        """
        print(f"Starting REINFORCE training for {self.config.num_episodes} episodes...")
        
        for episode in range(self.config.num_episodes):
            # Collect episode
            states, actions, rewards, log_probs = self.collect_episode()
            
            # Compute returns
            returns = self.compute_returns(rewards)
            
            # Update policy
            policy_loss = self.update_policy(states, actions, returns, log_probs)
            
            # Store statistics
            episode_reward = sum(rewards)
            episode_length = len(rewards)
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.policy_losses.append(policy_loss)
            
            # Logging
            if (episode + 1) % self.config.log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-self.config.log_interval:])
                avg_length = np.mean(self.episode_lengths[-self.config.log_interval:])
                avg_loss = np.mean(self.policy_losses[-self.config.log_interval:])
                
                print(f"Episode {episode + 1:4d} | "
                      f"Avg Reward: {avg_reward:7.2f} | "
                      f"Avg Length: {avg_length:6.1f} | "
                      f"Policy Loss: {avg_loss:8.3f}")
                
                # TensorBoard logging
                if self.writer:
                    self.writer.add_scalar("Training/Average_Reward", avg_reward, episode + 1)
                    self.writer.add_scalar("Training/Average_Length", avg_length, episode + 1)
                    self.writer.add_scalar("Training/Policy_Loss", avg_loss, episode + 1)
                    self.writer.add_scalar("Training/Episode_Reward", episode_reward, episode + 1)
            
            # Save model
            if self.config.save_model and (episode + 1) % self.config.save_interval == 0:
                self.save_model(f"reinforce_episode_{episode + 1}.pt")
        
        print("Training completed!")
        
        # Final save
        if self.config.save_model:
            self.save_model("reinforce_final.pt")
        
        # Close writer
        if self.writer:
            self.writer.close()
        
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "policy_losses": self.policy_losses
        }
    
    def evaluate(self, num_episodes: int = 100, render: bool = False) -> Dict[str, float]:
        """
        Evaluate the trained agent.
        
        Args:
            num_episodes: Number of evaluation episodes
            render: Whether to render episodes
            
        Returns:
            Evaluation statistics
        """
        print(f"Evaluating agent for {num_episodes} episodes...")
        
        self.policy_net.eval()
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        with torch.no_grad():
            for episode in range(num_episodes):
                obs, info = self.env.reset()
                state = torch.FloatTensor(obs.flatten() if obs.ndim > 1 else obs)
                
                episode_reward = 0.0
                episode_length = 0
                done = False
                
                while not done:
                    # Select best action (greedy)
                    state_batch = state.unsqueeze(0).to(self.device)
                    action_probs = self.policy_net.get_action_probabilities(state_batch)
                    action = torch.argmax(action_probs, dim=1).item()
                    
                    # Take action
                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    if render and episode < 5:  # Render first 5 episodes
                        print(f"Episode {episode + 1}, Step {episode_length}: "
                              f"Action {action}, Reward {reward}, State {info.get('state', 'N/A')}")
                        if terminated or truncated:
                            self.env.render()
                    
                    # Check for success (reached goal)
                    if info.get('is_goal', False):
                        success_count += 1
                    
                    # Update state
                    state = torch.FloatTensor(next_obs.flatten() if next_obs.ndim > 1 else next_obs)
                    done = terminated or truncated
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
        
        self.policy_net.train()
        
        # Compute statistics
        stats = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
            "success_rate": success_count / num_episodes,
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards)
        }
        
        print("Evaluation Results:")
        print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"  Mean Length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
        print(f"  Success Rate: {stats['success_rate']:.1%}")
        print(f"  Reward Range: [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
        
        return stats
    
    def save_model(self, filename: str):
        """Save the model checkpoint."""
        if self.config.log_dir:
            filepath = os.path.join(self.config.log_dir, filename)
        else:
            filepath = filename
        
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_losses': self.policy_losses,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.policy_losses = checkpoint.get('policy_losses', [])
        print(f"Model loaded from {filepath}")
    
    def plot_training_stats(self, save_path: Optional[str] = None, window_size: int = 100):
        """
        Plot training statistics.
        
        Args:
            save_path: Path to save the plot
            window_size: Window size for moving average
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("REINFORCE Training Statistics", fontsize=16)
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.3, color='blue')
        if len(self.episode_rewards) >= window_size:
            moving_avg = np.convolve(self.episode_rewards, np.ones(window_size)/window_size, mode='valid')
            axes[0, 0].plot(range(window_size-1, len(self.episode_rewards)), moving_avg, color='red', linewidth=2)
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Episode lengths
        axes[0, 1].plot(self.episode_lengths, alpha=0.3, color='green')
        if len(self.episode_lengths) >= window_size:
            moving_avg = np.convolve(self.episode_lengths, np.ones(window_size)/window_size, mode='valid')
            axes[0, 1].plot(range(window_size-1, len(self.episode_lengths)), moving_avg, color='red', linewidth=2)
        axes[0, 1].set_title("Episode Lengths")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Length")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Policy losses
        axes[1, 0].plot(self.policy_losses, alpha=0.7, color='purple')
        axes[1, 0].set_title("Policy Loss")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Success rate (estimated from rewards)
        if len(self.episode_rewards) >= window_size:
            # Assume positive rewards indicate success
            successes = [1 if r > 0 else 0 for r in self.episode_rewards]
            success_rate = np.convolve(successes, np.ones(window_size)/window_size, mode='valid')
            axes[1, 1].plot(range(window_size-1, len(successes)), success_rate, color='orange', linewidth=2)
            axes[1, 1].set_title("Success Rate (Moving Average)")
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Success Rate")
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plots saved to {save_path}")
        
        plt.show()


def main():
    """Main function to run REINFORCE training."""
    # Configuration
    config = REINFORCEConfig(
        hidden_dims=(64, 32),
        learning_rate=0.001,
        gamma=0.99,
        grid_size=5,
        observation_type="coordinates",
        num_episodes=2000,
        log_interval=100,
        save_interval=500,
        log_dir="logs/reinforce"
    )
    
    # Create and train agent
    agent = REINFORCEAgent(config)
    training_stats = agent.train()
    
    # Evaluate agent
    eval_stats = agent.evaluate(num_episodes=100, render=False)
    
    # Plot results
    agent.plot_training_stats(save_path=os.path.join(config.log_dir, "training_plots.png"))
    
    return agent, training_stats, eval_stats


if __name__ == "__main__":
    agent, training_stats, eval_stats = main()
