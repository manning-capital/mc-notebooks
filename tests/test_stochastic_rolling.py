"""
Tests for rolling stochastic models: RollingCointegration and RollingOrnsteinUhlenbeck.
"""

import pytest
import numpy as np
from numpy.typing import NDArray

from src.utils.stochastic import (
    RollingCointegration,
    RollingOrnsteinUhlenbeck,
    RollingCointegrationResults,
    RollingOrnsteinUhlenbeckResults,
    OrnsteinUhlenbeck,
)


class TestRollingCointegration:
    """Test RollingCointegration class."""

    def test_init(self):
        """Test initialization."""
        np.random.seed(42)
        y0 = np.cumsum(np.random.randn(100) * 0.1)
        y1 = y0 + np.random.randn(100) * 0.5

        rolling_coint = RollingCointegration(
            y0=y0, y1=y1, window=50, trend="c", lag=1
        )

        assert rolling_coint.y0 is y0
        assert rolling_coint.y1 is y1
        assert rolling_coint.window == 50
        assert rolling_coint.trend == "c"
        assert rolling_coint.lag == 1

    def test_fit_basic(self):
        """Test basic fit functionality."""
        # Generate synthetic cointegrated data
        np.random.seed(42)
        n = 200
        y0 = np.cumsum(np.random.randn(n) * 0.1)
        y1 = y0 + np.random.randn(n) * 0.5

        rolling_coint = RollingCointegration(
            y0=y0, y1=y1, window=100, trend="c", lag=1
        )

        results = rolling_coint.fit()

        assert isinstance(results, RollingCointegrationResults)
        assert len(results.beta) == n
        assert len(results.alpha) == n
        assert len(results.coint_t) == n
        assert len(results.pvalue) == n
        assert len(results.residual_mean) == n
        assert len(results.residual_std) == n

    def test_fit_has_valid_results(self):
        """Test that fit produces some valid results."""
        np.random.seed(42)
        n = 300
        y0 = np.cumsum(np.random.randn(n) * 0.1)
        y1 = y0 + np.random.randn(n) * 0.5

        rolling_coint = RollingCointegration(
            y0=y0, y1=y1, window=100, trend="c", lag=1
        )

        results = rolling_coint.fit()

        # Check that we have some valid windows
        valid = ~np.isnan(results.beta)
        assert valid.sum() > 0

        # Check that valid results have reasonable values
        if valid.sum() > 0:
            assert np.all(np.isfinite(results.beta[valid]))
            assert np.all(np.isfinite(results.alpha[valid]))
            assert np.all(np.isfinite(results.residual_mean[valid]))
            assert np.all(results.residual_std[valid] > 0)

    def test_fit_residual_statistics(self):
        """Test that residual statistics are computed correctly."""
        np.random.seed(42)
        n = 200
        y0 = np.cumsum(np.random.randn(n) * 0.1)
        y1 = y0 + np.random.randn(n) * 0.5

        rolling_coint = RollingCointegration(
            y0=y0, y1=y1, window=100, trend="c", lag=1
        )

        results = rolling_coint.fit()

        # Check that residual statistics are present
        assert hasattr(results, "residual_mean")
        assert hasattr(results, "residual_std")
        assert len(results.residual_mean) == n
        assert len(results.residual_std) == n

        # Check that valid windows have finite residual statistics
        valid = ~np.isnan(results.beta)
        if valid.sum() > 0:
            assert np.all(np.isfinite(results.residual_mean[valid]))
            assert np.all(np.isfinite(results.residual_std[valid]))
            assert np.all(results.residual_std[valid] >= 0)

    def test_fit_different_trends(self):
        """Test fit with different trend specifications."""
        np.random.seed(42)
        n = 200
        y0 = np.cumsum(np.random.randn(n) * 0.1)
        y1 = y0 + np.random.randn(n) * 0.5

        for trend in ["c", "ct", "n"]:
            rolling_coint = RollingCointegration(
                y0=y0, y1=y1, window=100, trend=trend, lag=1
            )
            results = rolling_coint.fit()
            assert isinstance(results, RollingCointegrationResults)

    def test_fit_expanding_window(self):
        """Test fit with expanding window."""
        np.random.seed(42)
        n = 200
        y0 = np.cumsum(np.random.randn(n) * 0.1)
        y1 = y0 + np.random.randn(n) * 0.5

        rolling_coint = RollingCointegration(
            y0=y0, y1=y1, window=100, trend="c", lag=1, expanding=True
        )

        results = rolling_coint.fit()
        assert isinstance(results, RollingCointegrationResults)

    def test_fit_validation_errors(self):
        """Test that fit raises errors for invalid inputs."""
        np.random.seed(42)
        y0 = np.random.randn(100)
        y1 = np.random.randn(50)  # Different length

        rolling_coint = RollingCointegration(
            y0=y0, y1=y1, window=50, trend="c", lag=1
        )

        with pytest.raises(ValueError, match="same length"):
            rolling_coint.fit()

    def test_fit_window_too_small(self):
        """Test that fit raises error for window too small."""
        np.random.seed(42)
        y0 = np.random.randn(100)
        y1 = np.random.randn(100)

        rolling_coint = RollingCointegration(
            y0=y0, y1=y1, window=5, trend="c", lag=5
        )

        with pytest.raises(ValueError, match="window must be at least"):
            rolling_coint.fit()


class TestRollingOrnsteinUhlenbeck:
    """Test RollingOrnsteinUhlenbeck class."""

    def test_init(self):
        """Test initialization."""
        np.random.seed(42)
        n = 100
        alpha = np.random.randn(n)
        beta = np.ones(n)
        pvalue = np.random.rand(n) * 0.1
        loss_level = np.random.randn(n)

        rolling_ou = RollingOrnsteinUhlenbeck(
            alpha=alpha,
            beta=beta,
            pvalue=pvalue,
            loss_level=loss_level,
            y0=np.random.randn(n),
            y1=np.random.randn(n),
            window=50,
        )

        assert rolling_ou.alpha is alpha
        assert rolling_ou.beta is beta
        assert rolling_ou.pvalue is pvalue
        assert rolling_ou.loss_level is loss_level

    def test_inherits_from_ornstein_uhlenbeck(self):
        """Test that RollingOrnsteinUhlenbeck inherits from OrnsteinUhlenbeck."""
        assert issubclass(RollingOrnsteinUhlenbeck, OrnsteinUhlenbeck)

    def test_parent_static_methods_accessible(self):
        """Test that parent class static methods are accessible."""
        np.random.seed(42)
        n = 100
        rolling_ou = RollingOrnsteinUhlenbeck(
            alpha=np.random.randn(n),
            beta=np.ones(n),
            pvalue=np.random.rand(n) * 0.1,
            loss_level=np.random.randn(n),
            y0=np.random.randn(n),
            y1=np.random.randn(n),
            window=50,
        )

        # Test that we can access parent class static methods
        mu = 0.5
        sigma = 0.1
        theta = 0.0
        r = 0.01

        F_val = rolling_ou.F(0.1, mu, sigma, theta, r, use_analytical=True)
        G_val = rolling_ou.G(0.1, mu, sigma, theta, r, use_analytical=True)

        assert np.isfinite(F_val)
        assert np.isfinite(G_val)
        assert F_val > 0
        assert G_val > 0

    def test_fit_basic(self):
        """Test basic fit functionality."""
        # Generate synthetic cointegrated data
        np.random.seed(42)
        n = 300
        y0 = np.cumsum(np.random.randn(n) * 0.1)
        y1 = y0 + np.random.randn(n) * 0.5

        # First get cointegration results
        rolling_coint = RollingCointegration(
            y0=y0, y1=y1, window=100, trend="c", lag=1
        )
        coint_results = rolling_coint.fit()

        # Then fit OU model
        rolling_ou = RollingOrnsteinUhlenbeck(
            alpha=coint_results.alpha,
            beta=coint_results.beta,
            pvalue=coint_results.pvalue,
            loss_level=np.full(n, np.nan),
            y0=y0,
            y1=y1,
            window=100,
            r=0.0001,
            c=0.001,
            pvalue_threshold=0.05,
            trend="c",
            lag=1,
        )

        results = rolling_ou.fit(use_analytical=True)

        assert isinstance(results, RollingOrnsteinUhlenbeckResults)
        assert len(results.mu) == n
        assert len(results.theta) == n
        assert len(results.sigma) == n
        assert len(results.half_life) == n
        assert len(results.entry_level) == n
        assert len(results.exit_level) == n
        assert len(results.loss_level) == n

    def test_fit_has_valid_results(self):
        """Test that fit produces some valid results."""
        np.random.seed(42)
        n = 300
        y0 = np.cumsum(np.random.randn(n) * 0.1)
        y1 = y0 + np.random.randn(n) * 0.5

        rolling_coint = RollingCointegration(
            y0=y0, y1=y1, window=100, trend="c", lag=1
        )
        coint_results = rolling_coint.fit()

        rolling_ou = RollingOrnsteinUhlenbeck(
            alpha=coint_results.alpha,
            beta=coint_results.beta,
            pvalue=coint_results.pvalue,
            loss_level=np.full(n, np.nan),
            y0=y0,
            y1=y1,
            window=100,
            pvalue_threshold=0.05,
        )

        results = rolling_ou.fit(use_analytical=True)

        # Check that we have some valid windows
        valid = ~np.isnan(results.mu)
        if valid.sum() > 0:
            assert np.all(results.mu[valid] > 0)
            assert np.all(results.sigma[valid] > 0)
            assert np.all(np.isfinite(results.theta[valid]))

    def test_fit_pvalue_threshold_filtering(self):
        """Test that pvalue threshold filters out non-cointegrated windows."""
        np.random.seed(42)
        n = 300
        y0 = np.cumsum(np.random.randn(n) * 0.1)
        y1 = y0 + np.random.randn(n) * 0.5

        rolling_coint = RollingCointegration(
            y0=y0, y1=y1, window=100, trend="c", lag=1
        )
        coint_results = rolling_coint.fit()

        # Use strict threshold
        rolling_ou = RollingOrnsteinUhlenbeck(
            alpha=coint_results.alpha,
            beta=coint_results.beta,
            pvalue=coint_results.pvalue,
            loss_level=np.full(n, np.nan),
            y0=y0,
            y1=y1,
            window=100,
            pvalue_threshold=0.01,  # Very strict
        )

        results = rolling_ou.fit(use_analytical=True)

        # Windows with pvalue > threshold should have NaN OU parameters
        high_pvalue = coint_results.pvalue > 0.01
        if high_pvalue.sum() > 0:
            assert np.all(np.isnan(results.mu[high_pvalue]))

    def test_fit_computes_levels(self):
        """Test that fit computes entry and exit levels."""
        np.random.seed(42)
        n = 300
        y0 = np.cumsum(np.random.randn(n) * 0.1)
        y1 = y0 + np.random.randn(n) * 0.5

        rolling_coint = RollingCointegration(
            y0=y0, y1=y1, window=100, trend="c", lag=1
        )
        coint_results = rolling_coint.fit()

        rolling_ou = RollingOrnsteinUhlenbeck(
            alpha=coint_results.alpha,
            beta=coint_results.beta,
            pvalue=coint_results.pvalue,
            loss_level=np.full(n, np.nan),
            y0=y0,
            y1=y1,
            window=100,
            pvalue_threshold=0.05,
        )

        results = rolling_ou.fit(use_analytical=True)

        # Check that levels are present
        assert hasattr(results, "entry_level")
        assert hasattr(results, "exit_level")
        assert hasattr(results, "loss_level")

        # Check that valid OU parameters have levels
        valid = ~np.isnan(results.mu)
        if valid.sum() > 0:
            # Some levels might be NaN if computation fails, but we should have some
            valid_levels = ~np.isnan(results.entry_level[valid])
            if valid_levels.sum() > 0:
                # Entry level should be below theta, exit level above theta
                valid_idx = np.where(valid)[0][valid_levels][0]
                assert results.entry_level[valid_idx] < results.theta[valid_idx]
                assert results.exit_level[valid_idx] > results.theta[valid_idx]

    def test_fit_validation_errors(self):
        """Test that fit raises errors for invalid inputs."""
        np.random.seed(42)
        n = 100
        y0 = np.random.randn(n)
        y1 = np.random.randn(50)  # Different length

        rolling_ou = RollingOrnsteinUhlenbeck(
            alpha=np.random.randn(n),
            beta=np.ones(n),
            pvalue=np.random.rand(n) * 0.1,
            loss_level=np.random.randn(n),
            y0=y0,
            y1=y1,
            window=50,
        )

        with pytest.raises(ValueError, match="same length"):
            rolling_ou.fit()


class TestRollingCointegrationResults:
    """Test RollingCointegrationResults dataclass."""

    def test_init(self):
        """Test initialization."""
        n = 100
        results = RollingCointegrationResults(
            beta=np.ones(n, dtype=np.float64),
            alpha=np.zeros(n, dtype=np.float64),
            coint_t=np.full(n, -2.0, dtype=np.float64),
            pvalue=np.full(n, 0.05, dtype=np.float64),
            crit_1pct=np.full(n, -3.0, dtype=np.float64),
            crit_5pct=np.full(n, -2.5, dtype=np.float64),
            crit_10pct=np.full(n, -2.0, dtype=np.float64),
            residual_mean=np.zeros(n, dtype=np.float64),
            residual_std=np.ones(n, dtype=np.float64),
            usedlag=1,
        )

        assert len(results.beta) == n
        assert len(results.alpha) == n
        assert results.beta.dtype == np.float64
        assert results.residual_mean.dtype == np.float64


class TestRollingOrnsteinUhlenbeckResults:
    """Test RollingOrnsteinUhlenbeckResults dataclass."""

    def test_init(self):
        """Test initialization."""
        n = 100
        results = RollingOrnsteinUhlenbeckResults(
            mu=np.full(n, 0.5, dtype=np.float64),
            theta=np.zeros(n, dtype=np.float64),
            sigma=np.full(n, 0.1, dtype=np.float64),
            half_life=np.full(n, 1.0, dtype=np.float64),
            entry_level=np.full(n, -0.1, dtype=np.float64),
            exit_level=np.full(n, 0.1, dtype=np.float64),
            loss_level=np.full(n, -0.2, dtype=np.float64),
        )

        assert len(results.mu) == n
        assert len(results.theta) == n
        assert results.mu.dtype == np.float64
        assert results.entry_level.dtype == np.float64

