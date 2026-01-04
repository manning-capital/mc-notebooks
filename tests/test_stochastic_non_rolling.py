"""
Tests for non-rolling stochastic models: GeometricBrownianMotion and OrnsteinUhlenbeck.
"""

import pytest
import numpy as np
from numpy.typing import NDArray

from src.utils.stochastic import (
    GeometricBrownianMotion,
    OrnsteinUhlenbeck,
    GeometricBrownianMotionResult,
    OrnsteinUhlenbeckResult,
    DELTA_T,
)


class TestGeometricBrownianMotion:
    """Test GeometricBrownianMotion class."""

    def test_init_without_params(self):
        """Test initialization without parameters."""
        gbm = GeometricBrownianMotion()
        # Accessing params when not set raises ValueError
        with pytest.raises(ValueError, match="Parameters are not set"):
            _ = gbm.params

    def test_init_with_params(self):
        """Test initialization with parameters."""
        params = GeometricBrownianMotionResult(mu=0.1, sigma=0.2)
        gbm = GeometricBrownianMotion(params=params)
        assert gbm.params == params

    def test_fit(self):
        """Test fitting GBM to synthetic data."""
        # Generate synthetic GBM data
        np.random.seed(42)
        n = 100
        dt = 0.01
        mu_true = 0.1
        sigma_true = 0.2
        prices = np.exp(
            np.cumsum(
                np.random.normal(mu_true * dt, sigma_true * np.sqrt(dt), n)
            )
        )

        gbm = GeometricBrownianMotion()
        result = gbm.fit(prices)

        assert isinstance(result, GeometricBrownianMotionResult)
        assert hasattr(result, "mu")
        assert hasattr(result, "sigma")
        assert np.isfinite(result.mu)
        assert np.isfinite(result.sigma)
        assert result.sigma > 0

    def test_log_likelihood(self):
        """Test log likelihood computation."""
        params = GeometricBrownianMotionResult(mu=0.1, sigma=0.2)
        gbm = GeometricBrownianMotion(params=params)

        # Generate some price data
        np.random.seed(42)
        prices = np.exp(np.cumsum(np.random.randn(50) * 0.02))

        ll = gbm.log_likelihood(prices)
        assert np.isfinite(ll)
        assert isinstance(ll, (float, np.floating))

    def test_simulate(self):
        """Test GBM simulation."""
        params = GeometricBrownianMotionResult(mu=0.1, sigma=0.2)
        gbm = GeometricBrownianMotion(params=params)

        N = 100
        N_simulated = 10
        X_0 = 100.0

        simulated = gbm.simulate(N, N_simulated, X_0)

        assert simulated.shape == (N_simulated, N)
        assert np.all(simulated[:, 0] == X_0)
        assert np.all(simulated > 0)  # Prices should be positive


class TestOrnsteinUhlenbeck:
    """Test OrnsteinUhlenbeck class."""

    def test_init_without_params(self):
        """Test initialization without parameters."""
        ou = OrnsteinUhlenbeck()
        # Accessing params when not set raises ValueError
        with pytest.raises(ValueError, match="Parameters are not set"):
            _ = ou.params

    def test_init_with_params(self):
        """Test initialization with parameters."""
        params = OrnsteinUhlenbeckResult(mu=0.5, theta=0.0, sigma=0.1)
        ou = OrnsteinUhlenbeck(params=params)
        assert ou.params == params

    def test_fit(self):
        """Test fitting OU to synthetic data."""
        # Generate synthetic OU data
        np.random.seed(42)
        n = 200
        mu_true = 0.5
        theta_true = 0.0
        sigma_true = 0.1
        dt = 1.0

        # Simulate OU process
        X = np.zeros(n)
        for i in range(1, n):
            X[i] = (
                X[i - 1]
                + mu_true * (theta_true - X[i - 1]) * dt
                + sigma_true * np.sqrt(dt) * np.random.randn()
            )

        ou = OrnsteinUhlenbeck()
        result = ou.fit(X)

        assert isinstance(result, OrnsteinUhlenbeckResult)
        assert hasattr(result, "mu")
        assert hasattr(result, "theta")
        assert hasattr(result, "sigma")
        assert np.isfinite(result.mu)
        assert np.isfinite(result.theta)
        assert np.isfinite(result.sigma)
        assert result.mu > 0
        assert result.sigma > 0

    def test_fit_invalid_data(self):
        """Test that fit raises error for invalid data."""
        ou = OrnsteinUhlenbeck()

        # Data that doesn't follow OU process (random walk)
        np.random.seed(42)
        X = np.cumsum(np.random.randn(50))

        # This should raise ValueError if coefficient is not in (0, 1)
        with pytest.raises(ValueError, match="Invalid OLS coefficient"):
            ou.fit(X)

    def test_log_likelihood(self):
        """Test log likelihood computation."""
        params = OrnsteinUhlenbeckResult(mu=0.5, theta=0.0, sigma=0.1)
        ou = OrnsteinUhlenbeck(params=params)

        # Generate some OU-like data
        np.random.seed(42)
        X = np.cumsum(np.random.randn(50) * 0.1)

        ll = ou.log_likelihood(X)
        assert np.isfinite(ll)
        assert isinstance(ll, (float, np.floating))

    def test_simulate(self):
        """Test OU simulation."""
        params = OrnsteinUhlenbeckResult(mu=0.5, theta=0.0, sigma=0.1)
        ou = OrnsteinUhlenbeck(params=params)

        N = 100
        N_simulated = 10
        X_0 = 0.0

        simulated = ou.simulate(N, N_simulated, X_0)

        assert simulated.shape == (N_simulated, N)
        assert np.all(simulated[:, 0] == X_0)
        assert np.all(np.isfinite(simulated))

    def test_static_methods_F_and_G(self):
        """Test static methods F and G."""
        mu = 0.5
        sigma = 0.1
        theta = 0.0
        x = 0.1
        r = 0.01

        # Test F function
        F_val = OrnsteinUhlenbeck.F(x, mu, sigma, theta, r, use_analytical=False)
        assert np.isfinite(F_val)
        assert F_val > 0

        # Test G function
        G_val = OrnsteinUhlenbeck.G(x, mu, sigma, theta, r, use_analytical=False)
        assert np.isfinite(G_val)
        assert G_val > 0

    def test_static_methods_F_and_G_analytical(self):
        """Test static methods F and G with analytical approximation."""
        mu = 0.5
        sigma = 0.1
        theta = 0.0
        x = 0.1
        r = 0.01

        # Test F function with analytical
        F_val = OrnsteinUhlenbeck.F(x, mu, sigma, theta, r, use_analytical=True)
        assert np.isfinite(F_val)
        assert F_val > 0

        # Test G function with analytical
        G_val = OrnsteinUhlenbeck.G(x, mu, sigma, theta, r, use_analytical=True)
        assert np.isfinite(G_val)
        assert G_val > 0

    def test_static_methods_array_inputs(self):
        """Test static methods with array inputs."""
        mu = np.array([0.5, 0.6])
        sigma = np.array([0.1, 0.15])
        theta = np.array([0.0, 0.1])
        x = np.array([0.1, 0.2])
        r = 0.01

        # Test F function with arrays
        F_vals = OrnsteinUhlenbeck.F(x, mu, sigma, theta, r, use_analytical=True)
        assert F_vals.shape == (2,)
        assert np.all(np.isfinite(F_vals))
        assert np.all(F_vals > 0)

        # Test G function with arrays
        G_vals = OrnsteinUhlenbeck.G(x, mu, sigma, theta, r, use_analytical=True)
        assert G_vals.shape == (2,)
        assert np.all(np.isfinite(G_vals))
        assert np.all(G_vals > 0)

    def test_get_optimal_exit_level(self):
        """Test get_optimal_exit_level static method."""
        mu = 0.5
        sigma = 0.1
        theta = 0.0
        r = 0.01
        c = 0.001

        exit_level = OrnsteinUhlenbeck.get_optimal_exit_level(
            mu=mu, sigma=sigma, theta=theta, r=r, c=c, use_analytical=True
        )

        assert np.isfinite(exit_level)
        assert exit_level > theta  # Exit level should be above mean

    def test_get_optimal_entry_level(self):
        """Test get_optimal_entry_level static method."""
        mu = 0.5
        sigma = 0.1
        theta = 0.0
        r = 0.01
        c = 0.001

        entry_level = OrnsteinUhlenbeck.get_optimal_entry_level(
            mu=mu, sigma=sigma, theta=theta, r=r, c=c, use_analytical=True
        )

        assert np.isfinite(entry_level)
        assert entry_level < theta  # Entry level should be below mean

    def test_get_optimal_levels_with_arrays(self):
        """Test optimal level methods with array inputs."""
        mu = np.array([0.5, 0.6])
        sigma = np.array([0.1, 0.15])
        theta = np.array([0.0, 0.1])
        r = 0.01
        c = 0.001

        exit_levels = OrnsteinUhlenbeck.get_optimal_exit_level(
            mu=mu, sigma=sigma, theta=theta, r=r, c=c, use_analytical=True
        )

        assert exit_levels.shape == (2,)
        assert np.all(np.isfinite(exit_levels))
        assert np.all(exit_levels > theta)

        entry_levels = OrnsteinUhlenbeck.get_optimal_entry_level(
            mu=mu, sigma=sigma, theta=theta, r=r, c=c, use_analytical=True
        )

        assert entry_levels.shape == (2,)
        assert np.all(np.isfinite(entry_levels))
        assert np.all(entry_levels < theta)

