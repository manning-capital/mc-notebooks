"""
Backward compatibility wrapper for the refactored stochastic module.

This module imports everything from the new stochastic package structure
and provides aliases for old class names to maintain backward compatibility.
"""

# Import everything from the new structure
# Note: We import from the package (stochastic/) not from this file

# Re-export everything from the package
from .stochastic import (
    # Constants
    DELTA_T,
    # Base classes
    StochasticModelResult,
    StochasticModel,
    # Non-rolling result classes
    GeometricBrownianMotionResult,
    OrnsteinUhlenbeckResult,
    # Non-rolling model classes
    GeometricBrownianMotion,
    OrnsteinUhlenbeck,
    # Rolling result classes
    RollingCointegrationResults,
    RollingOrnsteinUhlenbeckResults,
    # Rolling model classes
    RollingCointegration,
    RollingOrnsteinUhlenbeck,
)

# Backward compatibility aliases for old class names
StochasticModelParams = StochasticModelResult
GBMParams = GeometricBrownianMotionResult
OUParams = OrnsteinUhlenbeckResult

# Note: RollingOUParams is not provided as an alias because
# RollingOrnsteinUhlenbeckResults is a superset that includes all its fields
# plus additional fields (entry_level, exit_level, loss_level).

__all__ = [
    # Constants
    "DELTA_T",
    # Base classes (new names)
    "StochasticModelResult",
    "StochasticModel",
    # Base classes (old names for backward compatibility)
    "StochasticModelParams",
    # Non-rolling result classes (new names)
    "GeometricBrownianMotionResult",
    "OrnsteinUhlenbeckResult",
    # Non-rolling result classes (old names for backward compatibility)
    "GBMParams",
    "OUParams",
    # Non-rolling model classes
    "GeometricBrownianMotion",
    "OrnsteinUhlenbeck",
    # Rolling result classes
    "RollingCointegrationResults",
    "RollingOrnsteinUhlenbeckResults",
    # Rolling model classes
    "RollingCointegration",
    "RollingOrnsteinUhlenbeck",
]
