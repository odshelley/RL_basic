"""
Behavioral utilities for CPT-SAC and related algorithms.
"""

from .distortions import (
    DistortionFunction,
    IdentityDistortion,
    PrelecDistortion,
    WangDistortion,
    PowerUtility,
    choquet_expectation,
    cpt_functional,
    create_distortion,
    create_utility
)

from .reference import (
    ReferenceProvider,
    ConstantReference,
    EMAReturnReference,
    StateValueReference,
    create_reference_provider
)

from .discounting import (
    DiscountingMode,
    StandardDiscounting,
    BetaDeltaDiscounting,
    HyperbolicMixtureDiscounting,
    create_discounting_mode
)

__all__ = [
    # Distortions
    "DistortionFunction",
    "IdentityDistortion",
    "PrelecDistortion", 
    "WangDistortion",
    "PowerUtility",
    "choquet_expectation",
    "cpt_functional",
    "create_distortion",
    "create_utility",
    
    # Reference providers
    "ReferenceProvider",
    "ConstantReference",
    "EMAReturnReference",
    "StateValueReference",
    "create_reference_provider",
    
    # Discounting
    "DiscountingMode",
    "StandardDiscounting",
    "BetaDeltaDiscounting",
    "HyperbolicMixtureDiscounting",
    "create_discounting_mode"
]