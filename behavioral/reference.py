"""
Reference point providers for CPT evaluation.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class ReferenceProvider:
    """Base class for reference point providers"""
    
    def get_reference(self, state: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get reference point for given state"""
        raise NotImplementedError
    
    def update(self, **kwargs):
        """Update reference provider (e.g., with new returns)"""
        pass


class ConstantReference(ReferenceProvider):
    """Constant reference point"""
    
    def __init__(self, value: float = 0.0):
        self.value = value
    
    def get_reference(self, state: torch.Tensor, **kwargs) -> torch.Tensor:
        """Returns constant reference for all states"""
        batch_size = state.shape[0]
        return torch.full((batch_size,), self.value, device=state.device, dtype=state.dtype)


class EMAReturnReference(ReferenceProvider):
    """Exponential moving average of episode returns"""
    
    def __init__(self, tau: float = 0.01, initial_value: float = 0.0):
        self.tau = tau
        self.ema_return = initial_value
    
    def get_reference(self, state: torch.Tensor, **kwargs) -> torch.Tensor:
        """Returns current EMA return for all states"""
        batch_size = state.shape[0]
        return torch.full((batch_size,), self.ema_return, device=state.device, dtype=state.dtype)
    
    def update(self, episode_return: float):
        """Update EMA with new episode return"""
        self.ema_return = (1 - self.tau) * self.ema_return + self.tau * episode_return


class StateValueReference(ReferenceProvider):
    """State value baseline (stop-grad estimate of expected return)"""
    
    def __init__(self, value_network: Optional[nn.Module] = None):
        self.value_network = value_network
    
    def get_reference(self, state: torch.Tensor, **kwargs) -> torch.Tensor:
        """Returns state value estimate (detached from gradient)"""
        if self.value_network is None:
            # Fallback to zero reference
            batch_size = state.shape[0]
            return torch.zeros(batch_size, device=state.device, dtype=state.dtype)
        
        with torch.no_grad():
            return self.value_network(state).squeeze(-1)
    
    def set_value_network(self, value_network: nn.Module):
        """Set the value network for state value estimation"""
        self.value_network = value_network


def create_reference_provider(reference_type: str, params: Dict[str, Any]) -> ReferenceProvider:
    """Factory function to create reference providers"""
    if reference_type == "constant":
        return ConstantReference(value=params.get("constant", 0.0))
    elif reference_type == "ema_return":
        return EMAReturnReference(
            tau=params.get("ema_tau", 0.01),
            initial_value=params.get("initial_value", 0.0)
        )
    elif reference_type == "state_value":
        return StateValueReference()
    else:
        raise ValueError(f"Unknown reference type: {reference_type}")