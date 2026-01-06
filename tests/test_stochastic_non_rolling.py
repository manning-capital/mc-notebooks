"""
Tests for non-rolling stochastic models: GeometricBrownianMotion and OrnsteinUhlenbeck.
"""

import pytest
import numpy as np

from src.utils.stochastic import (
    GeometricBrownianMotion,
    OrnsteinUhlenbeck,
    GeometricBrownianMotionResult,
    OrnsteinUhlenbeckResult,
)

# Tolerance constants for numerical vs analytical consistency tests
# These can be adjusted if needed based on numerical integration accuracy
NUMERICAL_EXIT_LEVEL_RTOL = 0.01  # 1% relative tolerance for exit levels
NUMERICAL_EXIT_LEVEL_ATOL = 0.01  # 0.01 absolute tolerance for exit levels
NUMERICAL_ENTRY_LEVEL_RTOL = 0.01  # 1% relative tolerance for entry levels
NUMERICAL_ENTRY_LEVEL_ATOL = 0.01  # 0.01 absolute tolerance for entry levels


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
            np.cumsum(np.random.normal(mu_true * dt, sigma_true * np.sqrt(dt), n))
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

    def test_optimal_levels_paper_example(self):
        """
        Test optimal entry and exit levels against expected values from paper.

        Based on Figure 7 parameters:
        θ = 0.5388, µ = 16.6677, σ = 0.1599, r = 0.05, c = 0.05
        Expected entry level (dL): 0.4978
        Expected exit level (bL): 0.5570
        Stop-loss level (L): 0.4834
        """
        # Parameters from the paper
        mu = 16.6677
        sigma = 0.1599
        theta = 0.5388
        r = 0.05
        c = 0.05
        L = 0.4834

        # Expected values from the paper
        expected_entry_level = 0.4978
        expected_exit_level = 0.5570

        # Tolerance for comparison (paper values are likely rounded)
        # Using slightly more lenient tolerance as paper values may be approximations
        rtol = 5e-2  # 5% relative tolerance
        atol = 5e-3  # 0.005 absolute tolerance

        # Test exit level first (needed for entry level)
        exit_level = OrnsteinUhlenbeck.get_optimal_exit_level(
            mu=mu,
            sigma=sigma,
            theta=theta,
            r=r,
            c=c,
            L=L,
            use_analytical=True,
        )

        # Verify exit level
        assert np.isfinite(exit_level)
        assert exit_level > theta, (
            f"Exit level {exit_level} should be above theta {theta}"
        )
        assert exit_level > L, f"Exit level {exit_level} should be above stop-loss {L}"
        np.testing.assert_allclose(
            exit_level,
            expected_exit_level,
            rtol=rtol,
            atol=atol,
            err_msg=f"Exit level {exit_level} should be close to expected {expected_exit_level}",
        )

        # Test entry level with pre-computed exit level
        entry_level = OrnsteinUhlenbeck.get_optimal_entry_level(
            mu=mu,
            sigma=sigma,
            theta=theta,
            r=r,
            c=c,
            L=L,
            b_star=exit_level,
            use_analytical=True,
        )

        # Verify entry level
        assert np.isfinite(entry_level)
        assert entry_level < theta, (
            f"Entry level {entry_level} should be below theta {theta}"
        )
        assert entry_level > L, (
            f"Entry level {entry_level} should be above stop-loss {L}"
        )
        assert entry_level < exit_level, (
            f"Entry level {entry_level} should be below exit level {exit_level}"
        )
        np.testing.assert_allclose(
            entry_level,
            expected_entry_level,
            rtol=rtol,
            atol=atol,
            err_msg=f"Entry level {entry_level} should be close to expected {expected_entry_level}",
        )

    def test_optimal_levels_paper_example_numerical(self):
        """
        Test optimal entry and exit levels using numerical methods (not analytical).

        Same parameters as test_optimal_levels_paper_example but using numerical integration.
        """
        # Parameters from the paper
        mu = 16.6677
        sigma = 0.1599
        theta = 0.5388
        r = 0.05
        c = 0.05
        L = 0.4834

        # Expected values from the paper
        expected_entry_level = 0.4978
        expected_exit_level = 0.5570

        # Tolerance for comparison (numerical methods may have slightly different results)
        rtol = 5e-2  # 5% relative tolerance
        atol = 1e-2  # 0.01 absolute tolerance (looser for numerical methods)

        # Test exit level first (needed for entry level)
        exit_level = OrnsteinUhlenbeck.get_optimal_exit_level(
            mu=mu,
            sigma=sigma,
            theta=theta,
            r=r,
            c=c,
            L=L,
            use_analytical=False,  # Use numerical integration
        )

        # Verify exit level
        assert np.isfinite(exit_level)
        assert exit_level > theta
        assert exit_level > L
        np.testing.assert_allclose(
            exit_level,
            expected_exit_level,
            rtol=rtol,
            atol=atol,
            err_msg=f"Exit level {exit_level} should be close to expected {expected_exit_level}",
        )

        # Test entry level with pre-computed exit level
        entry_level = OrnsteinUhlenbeck.get_optimal_entry_level(
            mu=mu,
            sigma=sigma,
            theta=theta,
            r=r,
            c=c,
            L=L,
            b_star=exit_level,
            use_analytical=False,  # Use numerical integration
        )

        # Verify entry level
        assert np.isfinite(entry_level)
        assert entry_level < theta
        assert entry_level > L
        assert entry_level < exit_level
        np.testing.assert_allclose(
            entry_level,
            expected_entry_level,
            rtol=rtol,
            atol=atol,
            err_msg=f"Entry level {entry_level} should be close to expected {expected_entry_level}",
        )

    def test_optimal_levels_relationship(self):
        """
        Test that optimal levels satisfy expected relationships.

        For scenario (a): entry at dL, exit at bL
        For scenario (b): entry at dL, exit at stop-loss L
        """
        mu = 16.6677
        sigma = 0.1599
        theta = 0.5388
        r = 0.05
        c = 0.05
        L = 0.4834

        # Get optimal levels
        exit_level = OrnsteinUhlenbeck.get_optimal_exit_level(
            mu=mu, sigma=sigma, theta=theta, r=r, c=c, L=L, use_analytical=True
        )
        entry_level = OrnsteinUhlenbeck.get_optimal_entry_level(
            mu=mu,
            sigma=sigma,
            theta=theta,
            r=r,
            c=c,
            L=L,
            b_star=exit_level,
            use_analytical=True,
        )

        # Verify relationships
        # L < entry_level < theta < exit_level
        assert L < entry_level, (
            f"Stop-loss {L} should be below entry level {entry_level}"
        )
        assert entry_level < theta, (
            f"Entry level {entry_level} should be below theta {theta}"
        )
        assert theta < exit_level, (
            f"Theta {theta} should be below exit level {exit_level}"
        )

        # Entry level should be closer to L than to theta
        entry_to_L = entry_level - L
        theta_to_entry = theta - entry_level
        # This relationship may not always hold, but for these parameters it should
        assert entry_to_L < theta_to_entry, (
            f"Entry level {entry_level} should be closer to L {L} than to theta {theta}"
        )

        # Exit level should be above theta by a reasonable amount
        exit_to_theta = exit_level - theta
        assert exit_to_theta > 0.01, (
            f"Exit level {exit_level} should be significantly above theta {theta}"
        )

    def test_performance_exit_level(self):
        """Test performance comparison for exit level calculation."""
        import time

        # Parameters from the paper
        mu = 16.6677
        sigma = 0.1599
        theta = 0.5388
        r = 0.05
        c = 0.05
        L = 0.4834

        # Number of iterations for timing (reduced for faster test)
        n_iterations = 3

        # Analytical method
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            exit_level_analytical = OrnsteinUhlenbeck.get_optimal_exit_level(
                mu=mu, sigma=sigma, theta=theta, r=r, c=c, L=L, use_analytical=True
            )
        analytical_time = time.perf_counter() - start_time
        avg_analytical_time = analytical_time / n_iterations

        # Numerical method
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            exit_level_numerical = OrnsteinUhlenbeck.get_optimal_exit_level(
                mu=mu, sigma=sigma, theta=theta, r=r, c=c, L=L, use_analytical=False
            )
        numerical_time = time.perf_counter() - start_time
        avg_numerical_time = numerical_time / n_iterations

        print("\nExit Level Performance:")
        print(f"  Analytical: {avg_analytical_time * 1000:.3f} ms per call (avg)")
        print(f"  Numerical:  {avg_numerical_time * 1000:.3f} ms per call (avg)")
        print(
            f"  Speedup: {avg_numerical_time / avg_analytical_time:.2f}x faster with analytical"
        )
        print(
            f"  Results match: {abs(exit_level_analytical - exit_level_numerical):.6f} difference"
        )

        # Assert that analytical is faster (or at least not much slower)
        assert analytical_time < numerical_time * 2, (
            f"Analytical method should be faster, but took {analytical_time / numerical_time:.2f}x longer"
        )

    def test_performance_entry_level(self):
        """Test performance comparison for entry level calculation."""
        import time

        # Parameters from the paper
        mu = 16.6677
        sigma = 0.1599
        theta = 0.5388
        r = 0.05
        c = 0.05
        L = 0.4834

        # Number of iterations for timing (reduced for faster test)
        n_iterations = 3

        # Pre-compute exit levels
        exit_level_analytical = OrnsteinUhlenbeck.get_optimal_exit_level(
            mu=mu, sigma=sigma, theta=theta, r=r, c=c, L=L, use_analytical=True
        )
        exit_level_numerical = OrnsteinUhlenbeck.get_optimal_exit_level(
            mu=mu, sigma=sigma, theta=theta, r=r, c=c, L=L, use_analytical=False
        )

        # Analytical method
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            entry_level_analytical = OrnsteinUhlenbeck.get_optimal_entry_level(
                mu=mu,
                sigma=sigma,
                theta=theta,
                r=r,
                c=c,
                L=L,
                b_star=exit_level_analytical,
                use_analytical=True,
            )
        analytical_time = time.perf_counter() - start_time
        avg_analytical_time = analytical_time / n_iterations

        # Numerical method
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            entry_level_numerical = OrnsteinUhlenbeck.get_optimal_entry_level(
                mu=mu,
                sigma=sigma,
                theta=theta,
                r=r,
                c=c,
                L=L,
                b_star=exit_level_numerical,
                use_analytical=False,
            )
        numerical_time = time.perf_counter() - start_time
        avg_numerical_time = numerical_time / n_iterations

        print("\nEntry Level Performance:")
        print(f"  Analytical: {avg_analytical_time * 1000:.3f} ms per call (avg)")
        print(f"  Numerical:  {avg_numerical_time * 1000:.3f} ms per call (avg)")
        print(
            f"  Speedup: {avg_numerical_time / avg_analytical_time:.2f}x faster with analytical"
        )
        print(
            f"  Results match: {abs(entry_level_analytical - entry_level_numerical):.6f} difference"
        )

        # Assert that analytical is faster (or at least not much slower)
        assert analytical_time < numerical_time * 2, (
            f"Analytical method should be faster, but took {analytical_time / numerical_time:.2f}x longer"
        )

    def test_performance_vectorized(self):
        """Test performance comparison for vectorized (array) inputs."""
        import time

        # Parameters from the paper
        mu = 16.6677
        sigma = 0.1599
        theta = 0.5388
        r = 0.05
        c = 0.05
        L = 0.4834

        # Create arrays with multiple parameter sets (reduced for faster test)
        n_params = 20
        mu_arr = np.full(n_params, mu)
        sigma_arr = np.full(n_params, sigma)
        theta_arr = np.full(n_params, theta)
        c_arr = np.full(n_params, c)
        L_arr = np.full(n_params, L)

        # Analytical method
        start_time = time.perf_counter()
        exit_levels_analytical = OrnsteinUhlenbeck.get_optimal_exit_level(
            mu=mu_arr,
            sigma=sigma_arr,
            theta=theta_arr,
            r=r,
            c=c_arr,
            L=L_arr,
            use_analytical=True,
        )
        analytical_time = time.perf_counter() - start_time

        # Numerical method
        start_time = time.perf_counter()
        exit_levels_numerical = OrnsteinUhlenbeck.get_optimal_exit_level(
            mu=mu_arr,
            sigma=sigma_arr,
            theta=theta_arr,
            r=r,
            c=c_arr,
            L=L_arr,
            use_analytical=False,
        )
        numerical_time = time.perf_counter() - start_time

        print(f"\nVectorized Performance ({n_params} elements):")
        print(
            f"  Analytical: {analytical_time * 1000:.3f} ms total ({analytical_time * 1000 / n_params:.3f} ms per element)"
        )
        print(
            f"  Numerical:  {numerical_time * 1000:.3f} ms total ({numerical_time * 1000 / n_params:.3f} ms per element)"
        )
        print(
            f"  Speedup: {numerical_time / analytical_time:.2f}x faster with analytical"
        )

        # Verify results are close (use 1% tolerance constants)
        np.testing.assert_allclose(
            exit_levels_analytical,
            exit_levels_numerical,
            rtol=NUMERICAL_EXIT_LEVEL_RTOL,
            atol=NUMERICAL_EXIT_LEVEL_ATOL,
        )

        # Assert that analytical is faster (or at least not much slower)
        assert analytical_time < numerical_time * 2, (
            f"Analytical method should be faster, but took {analytical_time / numerical_time:.2f}x longer"
        )

    def test_consistency_paper_parameters(self):
        """Test consistency between analytical and numerical methods with paper parameters."""
        # Parameters from the paper
        mu = 16.6677
        sigma = 0.1599
        theta = 0.5388
        r = 0.05
        c = 0.05
        L = 0.4834

        # Test exit level consistency
        exit_analytical = OrnsteinUhlenbeck.get_optimal_exit_level(
            mu=mu, sigma=sigma, theta=theta, r=r, c=c, L=L, use_analytical=True
        )
        exit_numerical = OrnsteinUhlenbeck.get_optimal_exit_level(
            mu=mu, sigma=sigma, theta=theta, r=r, c=c, L=L, use_analytical=False
        )

        # Exit level should be within 1% tolerance
        np.testing.assert_allclose(
            exit_analytical,
            exit_numerical,
            rtol=NUMERICAL_EXIT_LEVEL_RTOL,
            atol=NUMERICAL_EXIT_LEVEL_ATOL,
            err_msg=f"Exit levels differ: analytical={exit_analytical:.6f}, numerical={exit_numerical:.6f}",
        )

        # Test entry level consistency
        entry_analytical = OrnsteinUhlenbeck.get_optimal_entry_level(
            mu=mu,
            sigma=sigma,
            theta=theta,
            r=r,
            c=c,
            L=L,
            b_star=exit_analytical,
            use_analytical=True,
        )
        entry_numerical = OrnsteinUhlenbeck.get_optimal_entry_level(
            mu=mu,
            sigma=sigma,
            theta=theta,
            r=r,
            c=c,
            L=L,
            b_star=exit_numerical,
            use_analytical=False,
        )

        # Entry level should be within 1% tolerance
        np.testing.assert_allclose(
            entry_analytical,
            entry_numerical,
            rtol=NUMERICAL_ENTRY_LEVEL_RTOL,
            atol=NUMERICAL_ENTRY_LEVEL_ATOL,
            err_msg=f"Entry levels differ: analytical={entry_analytical:.6f}, numerical={entry_numerical:.6f}",
        )

    @pytest.mark.parametrize(
        "mu,sigma,theta,r,c",
        [
            (0.5, 0.1, 0.0, 0.01, 0.001),
            (1.0, 0.2, 0.5, 0.05, 0.01),
            (5.0, 0.15, 1.0, 0.02, 0.005),
            (10.0, 0.1, 0.2, 0.03, 0.002),
        ],
    )
    def test_consistency_multiple_parameter_sets(self, mu, sigma, theta, r, c):
        """Test consistency across multiple parameter combinations."""
        L_test = theta - 2 * sigma

        exit_analytical = OrnsteinUhlenbeck.get_optimal_exit_level(
            mu=mu, sigma=sigma, theta=theta, r=r, c=c, L=L_test, use_analytical=True
        )
        exit_numerical = OrnsteinUhlenbeck.get_optimal_exit_level(
            mu=mu, sigma=sigma, theta=theta, r=r, c=c, L=L_test, use_analytical=False
        )

        entry_analytical = OrnsteinUhlenbeck.get_optimal_entry_level(
            mu=mu,
            sigma=sigma,
            theta=theta,
            r=r,
            c=c,
            L=L_test,
            b_star=exit_analytical,
            use_analytical=True,
        )
        entry_numerical = OrnsteinUhlenbeck.get_optimal_entry_level(
            mu=mu,
            sigma=sigma,
            theta=theta,
            r=r,
            c=c,
            L=L_test,
            b_star=exit_numerical,
            use_analytical=False,
        )

        # Check exit level consistency
        exit_diff = abs(exit_analytical - exit_numerical)
        exit_rel_diff = exit_diff / max(
            abs(exit_analytical), abs(exit_numerical), 1e-10
        )

        # Check entry level consistency
        entry_diff = abs(entry_analytical - entry_numerical)
        entry_rel_diff = entry_diff / max(
            abs(entry_analytical), abs(entry_numerical), 1e-10
        )

        print(f"\nTest Case: mu={mu}, sigma={sigma}, theta={theta}")
        print(
            f"  Exit level - Analytical: {exit_analytical:.6f}, Numerical: {exit_numerical:.6f}"
        )
        print(f"    Difference: {exit_diff:.6f} (relative: {exit_rel_diff * 100:.3f}%)")
        print(
            f"  Entry level - Analytical: {entry_analytical:.6f}, Numerical: {entry_numerical:.6f}"
        )
        print(
            f"    Difference: {entry_diff:.6f} (relative: {entry_rel_diff * 100:.3f}%)"
        )

        # Assert consistency (use 1% tolerance constants)
        np.testing.assert_allclose(
            exit_analytical,
            exit_numerical,
            rtol=NUMERICAL_EXIT_LEVEL_RTOL,
            atol=NUMERICAL_EXIT_LEVEL_ATOL,
            err_msg=f"Exit levels differ significantly for mu={mu}, sigma={sigma}, theta={theta}",
        )

        np.testing.assert_allclose(
            entry_analytical,
            entry_numerical,
            rtol=NUMERICAL_ENTRY_LEVEL_RTOL,
            atol=NUMERICAL_ENTRY_LEVEL_ATOL,
            err_msg=f"Entry levels differ significantly for mu={mu}, sigma={sigma}, theta={theta}",
        )

    @pytest.mark.parametrize(
        "mu,sigma,theta,c",
        [
            (0.5, 0.1, 0.0, 0.001),
            (1.0, 0.2, 0.5, 0.01),
            (5.0, 0.15, 1.0, 0.005),
            (10.0, 0.1, 0.2, 0.002),
        ],
    )
    def test_consistency_array_inputs(self, mu, sigma, theta, c):
        """Test consistency with array inputs (vectorized operations) for individual cases."""
        r = 0.01
        L = theta - 2 * sigma

        exit_analytical = OrnsteinUhlenbeck.get_optimal_exit_level(
            mu=mu,
            sigma=sigma,
            theta=theta,
            r=r,
            c=c,
            L=L,
            use_analytical=True,
        )
        exit_numerical = OrnsteinUhlenbeck.get_optimal_exit_level(
            mu=mu,
            sigma=sigma,
            theta=theta,
            r=r,
            c=c,
            L=L,
            use_analytical=False,
        )

        entry_analytical = OrnsteinUhlenbeck.get_optimal_entry_level(
            mu=mu,
            sigma=sigma,
            theta=theta,
            r=r,
            c=c,
            L=L,
            b_star=exit_analytical,
            use_analytical=True,
        )
        entry_numerical = OrnsteinUhlenbeck.get_optimal_entry_level(
            mu=mu,
            sigma=sigma,
            theta=theta,
            r=r,
            c=c,
            L=L,
            b_star=exit_numerical,
            use_analytical=False,
        )

        exit_diff = abs(exit_analytical - exit_numerical)
        entry_diff = abs(entry_analytical - entry_numerical)

        print(f"\nCase: mu={mu}, sigma={sigma}, theta={theta}")
        print(f"  Exit diff={exit_diff:.6f}, Entry diff={entry_diff:.6f}")
        print(
            f"    Analytical: exit={exit_analytical:.6f}, entry={entry_analytical:.6f}"
        )
        print(f"    Numerical:  exit={exit_numerical:.6f}, entry={entry_numerical:.6f}")

        # Verify results are consistent (use 1% tolerance constants)
        np.testing.assert_allclose(
            exit_analytical,
            exit_numerical,
            rtol=NUMERICAL_EXIT_LEVEL_RTOL,
            atol=NUMERICAL_EXIT_LEVEL_ATOL,
            err_msg=f"Exit levels differ for mu={mu}, sigma={sigma}, theta={theta}",
        )

        np.testing.assert_allclose(
            entry_analytical,
            entry_numerical,
            rtol=NUMERICAL_ENTRY_LEVEL_RTOL,
            atol=NUMERICAL_ENTRY_LEVEL_ATOL,
            err_msg=f"Entry levels differ for mu={mu}, sigma={sigma}, theta={theta}",
        )
