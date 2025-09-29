"""
Neural network architectures for policy and value functions.

This module contains PyTorch neural networks for function approximation
in reinforcement learning, specifically designed for the gridworld environment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class PolicyNetwork(nn.Module):
    """
    Simple MLP policy network for discrete action spaces.
    
    Outputs action probabilities using softmax activation.
    Suitable for REINFORCE and other policy gradient methods.
    """
    
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...] = (64, 32), 
                 num_actions: int = 4, activation: str = "relu"):
        """
        Initialize the policy network.
        
        Args:
            input_dim: Dimension of input observations
            hidden_dims: Tuple of hidden layer sizes
            num_actions: Number of discrete actions
            activation: Activation function ("relu", "tanh", "elu")
        """
        super(PolicyNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_actions = num_actions
        
        # Choose activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim
        
        # Output layer (no activation, we'll apply softmax)
        layers.append(nn.Linear(prev_dim, num_actions))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Action logits of shape (batch_size, num_actions)
        """
        return self.network(x)
    
    def get_action_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get action probabilities using softmax.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Action probabilities of shape (batch_size, num_actions)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
    
    def sample_action(self, x: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Sample an action from the policy distribution.
        
        Args:
            x: Input tensor of shape (input_dim,) - single observation
            
        Returns:
            Tuple of (sampled_action, log_probability)
        """
        # Ensure input is 2D: (1, input_dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Get action probabilities
        probs = self.get_action_probabilities(x)
        
        # Create categorical distribution and sample
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.squeeze()
    
    def get_log_prob(self, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Get log probabilities for given state-action pairs.
        
        Args:
            x: State tensor of shape (batch_size, input_dim)
            actions: Action tensor of shape (batch_size,)
            
        Returns:
            Log probabilities of shape (batch_size,)
        """
        probs = self.get_action_probabilities(x)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(actions)


class ValueNetwork(nn.Module):
    """
    Simple MLP value network for state value estimation.
    
    Outputs scalar state values for use in actor-critic methods
    or as baselines for variance reduction.
    """
    
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...] = (64, 32),
                 activation: str = "relu"):
        """
        Initialize the value network.
        
        Args:
            input_dim: Dimension of input observations
            hidden_dims: Tuple of hidden layer sizes
            activation: Activation function ("relu", "tanh", "elu")
        """
        super(ValueNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Choose activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim
        
        # Output layer (single scalar value)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            State values of shape (batch_size, 1)
        """
        return self.network(x)
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get state value (convenience method that squeezes output).
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,)
            
        Returns:
            State values of shape (batch_size,) or scalar
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        values = self.forward(x)
        return values.squeeze(-1)


class ActorCriticNetwork(nn.Module):
    """
    Combined actor-critic network with shared features.
    
    Uses a shared feature extractor followed by separate heads
    for policy (actor) and value (critic) outputs.
    """
    
    def __init__(self, input_dim: int, shared_dims: Tuple[int, ...] = (64,),
                 policy_dims: Tuple[int, ...] = (32,), value_dims: Tuple[int, ...] = (32,),
                 num_actions: int = 4, activation: str = "relu"):
        """
        Initialize the actor-critic network.
        
        Args:
            input_dim: Dimension of input observations
            shared_dims: Tuple of shared feature layer sizes
            policy_dims: Tuple of policy head layer sizes
            value_dims: Tuple of value head layer sizes
            num_actions: Number of discrete actions
            activation: Activation function
        """
        super(ActorCriticNetwork, self).__init__()
        
        # Choose activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Shared feature extractor
        shared_layers = []
        prev_dim = input_dim
        for shared_dim in shared_dims:
            shared_layers.append(nn.Linear(prev_dim, shared_dim))
            shared_layers.append(self.activation)
            prev_dim = shared_dim
        
        self.shared_features = nn.Sequential(*shared_layers)
        shared_output_dim = prev_dim
        
        # Policy head
        policy_layers = []
        prev_dim = shared_output_dim
        for policy_dim in policy_dims:
            policy_layers.append(nn.Linear(prev_dim, policy_dim))
            policy_layers.append(self.activation)
            prev_dim = policy_dim
        policy_layers.append(nn.Linear(prev_dim, num_actions))
        
        self.policy_head = nn.Sequential(*policy_layers)
        
        # Value head
        value_layers = []
        prev_dim = shared_output_dim
        for value_dim in value_dims:
            value_layers.append(nn.Linear(prev_dim, value_dim))
            value_layers.append(self.activation)
            prev_dim = value_dim
        value_layers.append(nn.Linear(prev_dim, 1))
        
        self.value_head = nn.Sequential(*value_layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (action_logits, state_values)
        """
        shared_features = self.shared_features(x)
        action_logits = self.policy_head(shared_features)
        state_values = self.value_head(shared_features)
        
        return action_logits, state_values.squeeze(-1)
    
    def get_action_and_value(self, x: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample action and get value for a single observation.
        
        Args:
            x: Input tensor of shape (input_dim,)
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        action_logits, values = self.forward(x)
        probs = F.softmax(action_logits, dim=-1)
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.squeeze(), values.squeeze()
