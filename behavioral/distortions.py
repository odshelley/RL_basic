"""
Distortion functions and Choquet expectation utilities for Behavioral SAC.
Implements probability weighting functions and Choquet integral estimators.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any


class DistortionFunction:
    """Base class for probability distortion functions g: [0,1] -> [0,1]"""
    
    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        """Apply distortion g(p)"""
        raise NotImplementedError
    
    def inv(self, u: torch.Tensor) -> torch.Tensor:
        """Generalized inverse g^{-1}(u)"""
        raise NotImplementedError


class IdentityDistortion(DistortionFunction):
    """Identity distortion g(p) = p"""
    
    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        return p
    
    def inv(self, u: torch.Tensor) -> torch.Tensor:
        return u


class PrelecDistortion(DistortionFunction):
    """Prelec distortion: g(p) = exp(-eta * (-ln(p))^alpha)"""
    
    def __init__(self, alpha: float = 0.65, eta: float = 1.0, eps: float = 1e-6):
        self.alpha = alpha
        self.eta = eta
        self.eps = eps
    
    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        # Clip to avoid numerical issues
        p_clipped = torch.clamp(p, self.eps, 1.0 - self.eps)
        log_p = torch.log(p_clipped)
        return torch.exp(-self.eta * torch.pow(-log_p, self.alpha))
    
    def inv(self, u: torch.Tensor) -> torch.Tensor:
        # Numerical inverse (approximate)
        u_clipped = torch.clamp(u, self.eps, 1.0 - self.eps)
        log_u = torch.log(u_clipped)
        inner = torch.pow(-log_u / self.eta, 1.0 / self.alpha)
        return torch.exp(-inner)


class WangDistortion(DistortionFunction):
    """Wang distortion using normal CDF with shift parameter"""
    
    def __init__(self, shift: float = 0.0):
        try:
            import scipy.stats
            self.scipy_available = True
        except ImportError:
            self.scipy_available = False
            print("Warning: scipy not available, Wang distortion will use approximation")
        self.shift = shift
    
    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        if self.scipy_available:
            from scipy.stats import norm
            # Convert to numpy for scipy, then back to torch
            p_np = p.detach().cpu().numpy()
            quantiles = norm.ppf(p_np)
            shifted_quantiles = quantiles + self.shift
            result_np = norm.cdf(shifted_quantiles)
            return torch.from_numpy(result_np).to(p.device)
        else:
            # Simple approximation when scipy not available
            return torch.clamp(p + self.shift * 0.1, 0.0, 1.0)
    
    def inv(self, u: torch.Tensor) -> torch.Tensor:
        if self.scipy_available:
            from scipy.stats import norm
            u_np = u.detach().cpu().numpy()
            quantiles = norm.ppf(u_np)
            shifted_quantiles = quantiles - self.shift
            result_np = norm.cdf(shifted_quantiles)
            return torch.from_numpy(result_np).to(u.device)
        else:
            # Simple approximation when scipy not available
            return torch.clamp(u - self.shift * 0.1, 0.0, 1.0)


def choquet_expectation(values: torch.Tensor, distortion: DistortionFunction) -> torch.Tensor:
    """
    Compute Choquet expectation using discrete estimator.
    
    Args:
        values: tensor of shape (..., K) with K samples
        distortion: distortion function g
        
    Returns:
        Choquet expectation of shape (...)
    """
    # Sort values in ascending order
    sorted_values, _ = torch.sort(values, dim=-1)
    K = values.shape[-1]
    
    # Compute cumulative probabilities C_i = i/K
    device = values.device
    i_range = torch.arange(1, K + 1, dtype=torch.float32, device=device)
    cumulative_probs = i_range / K
    
    # Compute distorted probabilities
    g_cumulative = distortion(cumulative_probs)
    
    # Compute weights: π^(i) = g(C_i) - g(C_{i-1})
    g_prev = torch.cat([torch.zeros(1, device=device), g_cumulative[:-1]])
    weights = g_cumulative - g_prev
    
    # Weighted sum
    return torch.sum(sorted_values * weights.unsqueeze(0), dim=-1)


class PowerUtility(nn.Module):
    """Power utility function u(x) = (x + eps)^alpha - eps^alpha for Lipschitz property"""
    
    def __init__(self, alpha: float = 0.88, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.eps_alpha = eps ** alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shifted = torch.clamp(x + self.eps, min=self.eps)
        return torch.pow(x_shifted, self.alpha) - self.eps_alpha


def cpt_functional(
    values: torch.Tensor,
    reference: torch.Tensor,
    u_plus: nn.Module,
    u_minus: nn.Module,
    g_plus: DistortionFunction,
    g_minus: DistortionFunction,
    lambda_loss_aversion: float = 2.0
) -> torch.Tensor:
    """
    Compute CPT functional for given values and reference point.
    
    Args:
        values: tensor of shape (..., K) with K value samples
        reference: tensor of shape (...,) with reference points
        u_plus, u_minus: utility functions for gains and losses
        g_plus, g_minus: distortion functions for gains and losses
        lambda_loss_aversion: loss aversion parameter
        
    Returns:
        CPT value of shape (...)
    """
    # Expand reference to match values shape
    ref_expanded = reference.unsqueeze(-1).expand_as(values)
    
    # Compute gains and losses
    gains = torch.clamp(values - ref_expanded, min=0.0)
    losses = torch.clamp(ref_expanded - values, min=0.0)
    
    # Apply utility functions
    u_gains = u_plus(gains)
    u_losses = u_minus(losses)
    
    # Compute Choquet expectations
    choquet_gains = choquet_expectation(u_gains, g_plus)
    choquet_losses = choquet_expectation(u_losses, g_minus)
    
    # CPT value: gains - λ * losses
    return choquet_gains - lambda_loss_aversion * choquet_losses


def create_distortion(distortion_type: str, params: Dict[str, Any]) -> DistortionFunction:
    """Factory function to create distortion functions"""
    if distortion_type == "identity":
        return IdentityDistortion()
    elif distortion_type == "prelec":
        return PrelecDistortion(
            alpha=params.get("alpha", 0.65),
            eta=params.get("eta", 1.0),
            eps=params.get("eps", 1e-6)
        )
    elif distortion_type == "wang":
        return WangDistortion(shift=params.get("shift", 0.0))
    else:
        raise ValueError(f"Unknown distortion type: {distortion_type}")


def create_utility(utility_type: str, params: Dict[str, Any]) -> nn.Module:
    """Factory function to create utility functions"""
    if utility_type == "power":
        return PowerUtility(
            alpha=params.get("alpha", 0.88),
            eps=params.get("eps", 1e-6)
        )
    else:
        raise ValueError(f"Unknown utility type: {utility_type}")