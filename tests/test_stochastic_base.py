"""
Tests for stochastic base classes and result dataclasses.
"""

import pytest

from src.utils.stochastic import (
    DELTA_T,
    StochasticModelResult,
    GeometricBrownianMotionResult,
    OrnsteinUhlenbeckResult,
    StochasticModel,
)


class TestDELTAT:
    """Test DELTA_T constant."""

    def test_delta_t_value(self):
        """Test that DELTA_T has the correct value."""
        assert DELTA_T == 1


class TestGeometricBrownianMotionResult:
    """Test GeometricBrownianMotionResult dataclass."""

    def test_init(self):
        """Test initialization of GBM result."""
        result = GeometricBrownianMotionResult(mu=0.1, sigma=0.2)
        assert result.mu == 0.1
        assert result.sigma == 0.2
        assert result.params == {"mu": 0.1, "sigma": 0.2}

    def test_to_dict(self):
        """Test to_dict method."""
        result = GeometricBrownianMotionResult(mu=0.1, sigma=0.2)
        assert result.to_dict() == {"mu": 0.1, "sigma": 0.2}

    def test_inherits_from_stochastic_model_result(self):
        """Test that GBM result inherits from StochasticModelResult."""
        result = GeometricBrownianMotionResult(mu=0.1, sigma=0.2)
        assert isinstance(result, StochasticModelResult)


class TestOrnsteinUhlenbeckResult:
    """Test OrnsteinUhlenbeckResult dataclass."""

    def test_init(self):
        """Test initialization of OU result."""
        result = OrnsteinUhlenbeckResult(mu=0.5, theta=0.0, sigma=0.1)
        assert result.mu == 0.5
        assert result.theta == 0.0
        assert result.sigma == 0.1
        assert result.params == {"mu": 0.5, "theta": 0.0, "sigma": 0.1}

    def test_to_dict(self):
        """Test to_dict method."""
        result = OrnsteinUhlenbeckResult(mu=0.5, theta=0.0, sigma=0.1)
        assert result.to_dict() == {"mu": 0.5, "theta": 0.0, "sigma": 0.1}

    def test_inherits_from_stochastic_model_result(self):
        """Test that OU result inherits from StochasticModelResult."""
        result = OrnsteinUhlenbeckResult(mu=0.5, theta=0.0, sigma=0.1)
        assert isinstance(result, StochasticModelResult)


class TestStochasticModel:
    """Test StochasticModel abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that StochasticModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            StochasticModel()

    def test_params_property_getter(self):
        """Test that params property raises error when not set."""
        # We need to create a concrete implementation to test
        from src.utils.stochastic import GeometricBrownianMotion

        gbm = GeometricBrownianMotion()
        with pytest.raises(ValueError, match="Parameters are not set"):
            _ = gbm.params

    def test_params_property_setter(self):
        """Test that params property can be set."""
        from src.utils.stochastic import (
            GeometricBrownianMotion,
            GeometricBrownianMotionResult,
        )

        gbm = GeometricBrownianMotion()
        result = GeometricBrownianMotionResult(mu=0.1, sigma=0.2)
        gbm.params = result
        assert gbm.params == result
