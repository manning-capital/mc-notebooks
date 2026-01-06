import numpy as np
from numpy.typing import NDArray
from typing import Literal, Optional
from dataclasses import dataclass
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import mackinnonp, mackinnoncrit
from statsmodels.tsa.tsatools import lagmat
import statsmodels.api as sm

from . import OrnsteinUhlenbeck
from .base import DELTA_T


@dataclass
class RollingCointegrationResults:
    """
    Results from rolling cointegration test.
    """

    beta: NDArray[np.float64]  # 1D array of float64
    alpha: NDArray[np.float64]  # 1D array of float64
    coint_t: NDArray[np.float64]  # 1D array of float64
    pvalue: NDArray[np.float64]  # 1D array of float64
    crit_1pct: NDArray[np.float64]  # 1D array of float64
    crit_5pct: NDArray[np.float64]  # 1D array of float64
    crit_10pct: NDArray[np.float64]  # 1D array of float64
    residual_mean: NDArray[
        np.float64
    ]  # 1D array of float64 - mean of residuals (spread) for each window
    residual_std: NDArray[
        np.float64
    ]  # 1D array of float64 - standard deviation of residuals (spread) for each window
    usedlag: int  # Number of lags used in ADF test


@dataclass
class RollingOrnsteinUhlenbeckResults:
    """
    Results from rolling Ornstein-Uhlenbeck parameter estimation.
    """

    mu: NDArray[np.float64]  # 1D array of float64
    theta: NDArray[np.float64]  # 1D array of float64
    sigma: NDArray[np.float64]  # 1D array of float64
    half_life: NDArray[np.float64]  # 1D array of float64


def _build_adf_matrices(
    residuals: NDArray[np.float64], lag: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Build the matrices needed for ADF regression on residuals.

    ADF regression (no trend): Δresid = ρ * resid_{t-1} + Σ γ_i * Δresid_{t-i} + error

    Parameters
    ----------
    residuals : ndarray
        Residuals from cointegrating regression
    lag : int
        Number of lagged differences to include

    Returns
    -------
    y : ndarray
        Dependent variable (differenced residuals)
    X : ndarray
        Regressors: [lagged_level, lagged_differences]
    """
    # Compute differences
    diff_resid = np.diff(residuals)

    # Build lag matrix of differences (trim='both' keeps aligned observations)
    # This creates [diff_t, diff_{t-1}, ..., diff_{t-lag}]
    xdall = lagmat(diff_resid[:, None], lag, trim="both", original="in")

    # Number of usable observations
    nobs = xdall.shape[0]

    # Replace first column (current diff) with lagged level
    # We want resid[lag:nobs+lag] as the lagged level
    xdall[:, 0] = residuals[lag : nobs + lag]

    # Dependent variable: current differenced residual
    y = diff_resid[lag:]

    # X matrix: [lagged_level, lagged_diffs]
    # Column 0 is lagged level, columns 1:lag+1 are lagged differences
    X = xdall[:, : lag + 1]

    return y, X


class RollingCointegration:
    """
    Rolling cointegration test.
    """

    def __init__(
        self,
        y0: NDArray[np.float64],
        y1: NDArray[np.float64],
        window: int,
        trend: Literal["c", "ct", "ctt", "n"] = "c",
        lag: int = 1,
        min_nobs: Optional[int] = None,
        expanding: bool = False,
    ):
        self.y0 = y0
        self.y1 = y1
        self.window = window
        self.trend = trend
        self.lag = lag
        self.min_nobs = min_nobs
        self.expanding = expanding

    def fit(
        self, method: Literal["inv", "lstsq", "pinv"] = "inv"
    ) -> RollingCointegrationResults:
        """
        Fit the rolling cointegration model.

        Performs rolling cointegrating regression and ADF test on residuals.
        """
        # Input validation
        y0 = np.asarray(self.y0, dtype=np.float64).squeeze()
        y1 = np.asarray(self.y1, dtype=np.float64).squeeze()

        if y0.ndim != 1 or y1.ndim != 1:
            raise ValueError("y0 and y1 must be 1-dimensional")
        if len(y0) != len(y1):
            raise ValueError("y0 and y1 must have the same length")

        nobs = len(y0)
        k_vars = 2  # Two variables for cointegration

        if self.window < self.lag + 5:
            raise ValueError(f"window must be at least lag + 5 = {self.lag + 5}")

        # =========================================================================
        # STAGE 1: Rolling cointegrating regression
        # y0 = alpha + beta*y1 + residuals (spread)
        # =========================================================================

        # Prepare exogenous: [y1, const] or just [y1] if no trend
        if self.trend == "n":
            exog = y1.reshape(-1, 1)
        else:
            exog = sm.add_constant(y1)  # [const, y1] -> need to reorder
            # add_constant puts const first, we want [y1, const] for consistency
            exog = exog[:, ::-1]  # Now [y1, const]

        # Run RollingOLS for cointegrating regression
        rolling_coint_model = RollingOLS(
            endog=y0,
            exog=exog,
            window=self.window,
            min_nobs=self.min_nobs,
            expanding=self.expanding,
        )
        rolling_coint_results = rolling_coint_model.fit(
            method=method, params_only=False
        )

        # Extract parameters
        params = rolling_coint_results.params
        beta = params[:, 0].copy()  # Coefficient on y1

        if self.trend == "n":
            alpha = np.zeros(nobs)  # No intercept
        else:
            alpha = params[:, 1].copy()  # Constant term

        # =========================================================================
        # STAGE 2: Rolling ADF test on spread (cointegration test)
        # =========================================================================

        # Initialize output arrays
        coint_t = np.full(nobs, np.nan)
        pvalue = np.full(nobs, np.nan)
        crit_1pct = np.full(nobs, np.nan)
        crit_5pct = np.full(nobs, np.nan)
        crit_10pct = np.full(nobs, np.nan)
        residual_mean = np.full(nobs, np.nan)
        residual_std = np.full(nobs, np.nan)

        # Determine starting index
        if self.expanding:
            first_idx = max(
                self.min_nobs if self.min_nobs is not None else 2, self.lag + 5
            )
        else:
            first_idx = self.window

        # Precompute critical values for cointegration test
        if self.trend != "n":
            adf_nobs_approx = self.window - self.lag - 1
            crit_vals = mackinnoncrit(
                N=k_vars, regression=self.trend, nobs=adf_nobs_approx - 1
            )
            crit_1pct_val = crit_vals[0]
            crit_5pct_val = crit_vals[1]
            crit_10pct_val = crit_vals[2]
        else:
            crit_1pct_val = crit_5pct_val = crit_10pct_val = np.nan

        # Main loop: For each window, compute spread and ADF test
        for t in range(first_idx - 1, nobs):
            # Skip if parameters not available
            if np.any(np.isnan(params[t])):
                continue

            # Get window bounds
            if self.expanding:
                w_start = 0
            else:
                w_start = t - self.window + 1
            w_end = t + 1

            # Compute spread for this window using THIS window's params
            y0_window = y0[w_start:w_end]
            y1_window = y1[w_start:w_end]
            spread_window = y0_window - beta[t] * y1_window - alpha[t]

            # Compute residual statistics
            residual_mean[t] = np.mean(spread_window)
            residual_std[t] = np.std(spread_window, ddof=1)

            # Check for degenerate cases
            if residual_std[t] < 1e-10:
                continue

            # -----------------------------------------------------------------
            # ADF Test on spread (Cointegration Test)
            # -----------------------------------------------------------------
            try:
                adf_y, adf_X = _build_adf_matrices(spread_window, self.lag)
                adf_nobs = len(adf_y)

                if adf_nobs >= self.lag + 2:
                    # Run ADF OLS
                    XtX = adf_X.T @ adf_X
                    Xty = adf_X.T @ adf_y
                    XtX_inv = np.linalg.inv(XtX)
                    adf_params = XtX_inv @ Xty

                    # Compute t-statistic
                    adf_resid = adf_y - adf_X @ adf_params
                    ssr = np.sum(adf_resid**2)
                    df = adf_nobs - adf_X.shape[1]
                    mse = ssr / df
                    se = np.sqrt(np.diag(XtX_inv) * mse)
                    t_stat = adf_params[0] / se[0]

                    coint_t[t] = t_stat
                    pvalue[t] = mackinnonp(t_stat, regression=self.trend, N=k_vars)

                    # Store critical values
                    if self.trend != "n":
                        crit_1pct[t] = crit_1pct_val
                        crit_5pct[t] = crit_5pct_val
                        crit_10pct[t] = crit_10pct_val

            except (np.linalg.LinAlgError, Exception):
                pass

        return RollingCointegrationResults(
            beta=beta,
            alpha=alpha,
            coint_t=coint_t,
            pvalue=pvalue,
            crit_1pct=crit_1pct,
            crit_5pct=crit_5pct,
            crit_10pct=crit_10pct,
            residual_mean=residual_mean,
            residual_std=residual_std,
            usedlag=self.lag,
        )


class RollingOrnsteinUhlenbeck(OrnsteinUhlenbeck):
    """
    Rolling Ornstein-Uhlenbeck parameter estimation.
    """

    def __init__(
        self,
        alpha: NDArray[np.float64],
        beta: NDArray[np.float64],
        pvalue: NDArray[np.float64],
        y0: NDArray[np.float64],
        y1: NDArray[np.float64],
        window: int,
        pvalue_threshold: float = 0.05,
        lag: int = 1,
        min_nobs: Optional[int] = None,
        expanding: bool = False,
    ):
        # Don't call super().__init__() since we're not using params-based initialization
        self.alpha = alpha
        self.beta = beta
        self.pvalue = pvalue
        self.y0 = y0
        self.y1 = y1
        self.window = window
        self.pvalue_threshold = pvalue_threshold
        self.lag = lag
        self.min_nobs = min_nobs
        self.expanding = expanding

    def fit(self) -> RollingOrnsteinUhlenbeckResults:
        """
        Fit the rolling Ornstein-Uhlenbeck model.

        Performs rolling OU parameter estimation on spread.
        """
        # Input validation
        y0 = np.asarray(self.y0, dtype=np.float64).squeeze()
        y1 = np.asarray(self.y1, dtype=np.float64).squeeze()
        alpha = np.asarray(self.alpha, dtype=np.float64)
        beta = np.asarray(self.beta, dtype=np.float64)
        pvalue = np.asarray(self.pvalue, dtype=np.float64)

        if y0.ndim != 1 or y1.ndim != 1:
            raise ValueError("y0 and y1 must be 1-dimensional")
        if len(y0) != len(y1):
            raise ValueError("y0 and y1 must have the same length")

        nobs = len(y0)

        # Initialize output arrays
        mu = np.full(nobs, np.nan)
        theta = np.full(nobs, np.nan)
        sigma = np.full(nobs, np.nan)
        half_life = np.full(nobs, np.nan)

        # Determine starting index
        if self.expanding:
            first_idx = max(
                self.min_nobs if self.min_nobs is not None else 2, self.lag + 5
            )
        else:
            first_idx = self.window

        # =========================================================================
        # STAGE 3: Rolling OU parameter estimation on spread
        # =========================================================================
        for t in range(first_idx - 1, nobs):
            # Skip if pvalue indicates non-cointegration
            if pvalue[t] > self.pvalue_threshold:
                continue

            # Get window bounds
            if self.expanding:
                w_start = 0
            else:
                w_start = t - self.window + 1
            w_end = t + 1

            # Compute spread for this window using pre-computed alpha and beta
            y0_window = y0[w_start:w_end]
            y1_window = y1[w_start:w_end]
            spread_window = y0_window - beta[t] * y1_window - alpha[t]

            # Check for degenerate cases
            if len(spread_window) < 3 or np.std(spread_window, ddof=1) < 1e-10:
                continue

            # -----------------------------------------------------------------
            # OU Parameter Estimation on spread
            # Regression: spread_{t+1} = intercept + coef * spread_t + noise
            # -----------------------------------------------------------------
            try:
                # Set up OU regression: spread_{t+1} on spread_t
                spread_next = spread_window[1:]
                spread_lag = spread_window[:-1]

                # Add constant: [spread_lag, const]
                X_ou = np.column_stack([spread_lag, np.ones(len(spread_lag))])

                # OLS regression
                XtX_ou = X_ou.T @ X_ou
                Xty_ou = X_ou.T @ spread_next
                XtX_inv_ou = np.linalg.inv(XtX_ou)
                ou_params = XtX_inv_ou @ Xty_ou

                coef = ou_params[0]  # AR(1) coefficient
                intercept = ou_params[1]  # Intercept

                # For valid mean-reverting OU: coef must be in (0, 1)
                # coef = exp(-mu * DELTA_T)
                if 0 < coef < 1:
                    # Extract OU parameters
                    mu_val = -np.log(coef) / DELTA_T
                    theta_val = intercept / (1 - coef)

                    # Compute sigma from residuals
                    ou_resid = spread_next - X_ou @ ou_params
                    n_resid = len(ou_resid)
                    residual_var = (
                        np.sum(ou_resid**2) / (n_resid - 2) if n_resid > 2 else np.nan
                    )

                    if residual_var > 0 and not np.isnan(residual_var):
                        ou_residual_std = np.sqrt(residual_var)
                        # ou_residual_std^2 = sigma^2 * (1 - exp(-2*mu*dt)) / (2*mu)
                        factor = (1 - np.exp(-2 * mu_val * DELTA_T)) / (2 * mu_val)
                        if factor > 0:
                            sigma_val = ou_residual_std / np.sqrt(factor)

                            mu[t] = mu_val
                            theta[t] = theta_val
                            sigma[t] = sigma_val
                            half_life[t] = np.log(2) / mu_val

            except (np.linalg.LinAlgError, Exception):
                pass

        return RollingOrnsteinUhlenbeckResults(
            mu=mu,
            theta=theta,
            sigma=sigma,
            half_life=half_life,
        )
