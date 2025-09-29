"""
Discounting utilities for behavioral SAC.
Supports standard, beta-delta, and hyperbolic mixture discounting.
"""

import torch
from typing import List, Dict, Any


class DiscountingMode:
    """Base class for discounting modes"""
    
    def get_discount_factor(self, step: int = 0) -> float:
        """Get discount factor (may depend on step for hyperbolic)"""
        raise NotImplementedError
    
    def get_effective_gamma(self) -> float:
        """Get effective gamma for one-step backups"""
        raise NotImplementedError


class StandardDiscounting(DiscountingMode):
    """Standard exponential discounting"""
    
    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
    
    def get_discount_factor(self, step: int = 0) -> float:
        return self.gamma
    
    def get_effective_gamma(self) -> float:
        return self.gamma


class BetaDeltaDiscounting(DiscountingMode):
    """Beta-delta (quasi-hyperbolic) discounting"""
    
    def __init__(self, beta: float = 1.0, delta: float = 0.99):
        self.beta = beta
        self.delta = delta
        self.gamma_beta = beta * delta  # Effective discount for one-step backups
    
    def get_discount_factor(self, step: int = 0) -> float:
        if step == 0:
            return self.gamma_beta
        else:
            return self.delta
    
    def get_effective_gamma(self) -> float:
        return self.gamma_beta


class HyperbolicMixtureDiscounting(DiscountingMode):
    """Hyperbolic discounting via finite mixture of exponentials"""
    
    def __init__(self, gammas: List[float], probs: List[float]):
        assert len(gammas) == len(probs), "Gammas and probs must have same length"
        assert abs(sum(probs) - 1.0) < 1e-6, "Probabilities must sum to 1"
        
        self.gammas = gammas
        self.probs = probs
        
        # Effective gamma for one-step backups
        self.effective_gamma = sum(p * g for p, g in zip(probs, gammas))
    
    def get_discount_factor(self, step: int = 0) -> float:
        # For mixture, we typically use the effective gamma
        return self.effective_gamma
    
    def get_effective_gamma(self) -> float:
        return self.effective_gamma
    
    def apply_mixture_discounting(self, values: torch.Tensor) -> torch.Tensor:
        """
        Apply mixture discounting to a tensor of values.
        
        Args:
            values: tensor of shape (...) representing future values
            
        Returns:
            Discounted values using mixture weights
        """
        result = torch.zeros_like(values)
        for prob, gamma in zip(self.probs, self.gammas):
            result += prob * gamma * values
        return result


def create_discounting_mode(discounting_type: str, params: Dict[str, Any]) -> DiscountingMode:
    """Factory function to create discounting modes"""
    if discounting_type == "standard":
        return StandardDiscounting(gamma=params.get("gamma", 0.99))
    elif discounting_type == "beta_delta":
        return BetaDeltaDiscounting(
            beta=params.get("beta", 1.0),
            delta=params.get("delta", 0.99)
        )
    elif discounting_type == "hyperbolic_mixture":
        mixture_params = params.get("mixture", {})
        return HyperbolicMixtureDiscounting(
            gammas=mixture_params.get("gammas", [0.99, 0.95, 0.90]),
            probs=mixture_params.get("probs", [0.6, 0.3, 0.1])
        )
    else:
        raise ValueError(f"Unknown discounting type: {discounting_type}")