"""
Rolling Pairs Trading Statistics

Implements rolling window cointegration tests and Ornstein-Uhlenbeck parameter
estimation for pairs trading analysis. Combines:
1. Engle-Granger cointegration test (alpha, beta, p-value)
2. OU process parameters on the spread (mu, theta, sigma)

Uses statsmodels' RollingOLS for efficient computation.
"""

import numpy as np
from typing import Literal, Optional
from dataclasses import dataclass
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import mackinnonp, mackinnoncrit
from statsmodels.tsa.tsatools import lagmat
import statsmodels.api as sm
import sys
from pathlib import Path

# Add src to path for importing stochastic
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))
import src.utils.stochastic as stochastic

# Time step for OU parameter estimation (1 = daily data with dt=1)
DELTA_T = 1


@dataclass
class RollingPairsTradingResults:
    """
    Combined results from rolling cointegration test and OU parameter estimation.

    Cointegrating regression: y0 = alpha + beta*y1 + residuals (spread)
    OU process on spread: dS = mu*(theta - S)*dt + sigma*dW

    Attributes
    ----------
    # Cointegration test results
    coint_t : ndarray
        Rolling t-statistics from ADF test on residuals
    p_value : ndarray
        Rolling p-values for cointegration test

    # Cointegrating regression parameters
    alpha : ndarray
        Rolling intercept from cointegrating regression.
    beta : ndarray
        Rolling slope (hedge ratio) on y1.

    # OU process parameters (estimated on the spread)
    ou_mu : ndarray
        Rolling mean reversion rate. Higher = faster reversion.
    ou_theta : ndarray
        Rolling long-term mean of the spread.
    ou_sigma : ndarray
        Rolling volatility of the spread.
    half_life : ndarray
        Rolling half-life of mean reversion: ln(2) / ou_mu

    # Optimal trading levels
    entry_level : ndarray
        Optimal entry level (d_star) for entering trades.
    exit_level : ndarray
        Optimal exit level (b_star) for exiting trades.
    loss_level : ndarray
        Loss level (L) for stopping out of trades. Computed as ou_theta - stop_loss_factor * ou_sigma.

    # Residual statistics
    residual_mean : ndarray
        Rolling mean of residuals (spread) from cointegrating regression.
    residual_std : ndarray
        Rolling standard deviation of residuals (spread).

    # Metadata
    usedlag : int
        Number of lags used in ADF test
    """

    # Cointegration results
    coint_t: np.ndarray
    p_value: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray

    # OU parameters
    ou_mu: np.ndarray
    ou_theta: np.ndarray
    ou_sigma: np.ndarray
    half_life: np.ndarray

    # Optimal trading levels
    entry_level: np.ndarray
    exit_level: np.ndarray
    loss_level: np.ndarray

    # Residual statistics
    residual_mean: np.ndarray
    residual_std: np.ndarray

    # Metadata
    usedlag: int

    def __repr__(self):
        valid = ~np.isnan(self.coint_t)
        return (
            f"RollingPairsTradingResults(\n"
            f"  n_valid_windows={valid.sum()},\n"
            f"  --- Cointegration ---\n"
            f"  mean_p_value={np.nanmean(self.p_value):.4f},\n"
            f"  mean_alpha={np.nanmean(self.alpha):.4f},\n"
            f"  mean_beta={np.nanmean(self.beta):.4f},\n"
            f"  --- OU Parameters ---\n"
            f"  mean_mu={np.nanmean(self.ou_mu):.4f},\n"
            f"  mean_theta={np.nanmean(self.ou_theta):.4f},\n"
            f"  mean_sigma={np.nanmean(self.ou_sigma):.4f},\n"
            f"  mean_half_life={np.nanmean(self.half_life):.2f},\n"
            f"  --- Trading Levels ---\n"
            f"  mean_entry_level={np.nanmean(self.entry_level):.4f},\n"
            f"  mean_exit_level={np.nanmean(self.exit_level):.4f},\n"
            f"  mean_loss_level={np.nanmean(self.loss_level):.4f},\n"
            f"  --- Residual Statistics ---\n"
            f"  mean_residual_mean={np.nanmean(self.residual_mean):.4f},\n"
            f"  mean_residual_std={np.nanmean(self.residual_std):.4f}\n"
            f")"
        )


def _build_adf_matrices(
    residuals: np.ndarray, lag: int
) -> tuple[np.ndarray, np.ndarray]:
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


def rolling_pairs_trading(
    y0: np.ndarray,
    y1: np.ndarray,
    window: int,
    p_value_threshold: float = 0.05,
    trend: Literal["c", "ct", "ctt", "n"] = "c",
    method: Literal["inv", "lstsq", "pinv"] = "inv",
    lag: int = 1,
    min_nobs: Optional[int] = None,
    expanding: bool = False,
    r: float = 0.0001,
    c: float = 0.001,
    L: Optional[np.ndarray] = None,
    stop_loss_factor: float = 2.25,
    compute_levels: bool = True,
    use_analytical: bool = True,
) -> RollingPairsTradingResults:
    """
    Perform rolling pairs trading analysis: cointegration test + OU parameter estimation.

    This function efficiently computes over rolling windows:
    1. Cointegrating regression: y0 = alpha + beta*y1 + spread
    2. ADF test on the spread (cointegration test)
    3. OU process parameters on the spread: dS = mu*(theta - S)*dt + sigma*dW

    Parameters
    ----------
    y0 : array_like, 1d
        First price series (dependent variable in cointegrating regression).
    y1 : array_like, 1d
        Second price series (independent variable / hedge instrument).
    window : int
        Size of the rolling window.
    p_value_threshold: float = 0.05,
        Threshold for the p-value of the cointegration test. If the p-value is greater than this threshold, the ou parameters will be set to NaN.
    trend : {"c", "ct", "ctt", "n"}, default "c"
        Trend term in cointegrating equation:
        * "c" : constant only (recommended for pairs trading)
        * "ct" : constant and linear trend
        * "n" : no constant, no trend
    method : {"inv", "lstsq", "pinv"}, default "inv"
        Method for computing rolling regression parameters.
    lag : int, default 1
        Fixed lag length for ADF test (1-4 typical for daily data).
    min_nobs : int, optional
        Minimum observations required in each window.
    expanding : bool, default False
        If True, use expanding window instead of rolling.
    r : float, default 0.0001
        Discount rate for optimal stopping problem.
    c : float, default 0.001
        Transaction cost for entering/exiting trades.
    L : ndarray, optional
        Loss level array. If None, computed as ou_theta - stop_loss_factor * ou_sigma.
    stop_loss_factor : float, default 2.25
        Factor for computing loss level: L = ou_theta - stop_loss_factor * ou_sigma.
        Only used if L is None.
    compute_levels : bool, default True
        If True, compute optimal entry and exit levels after OU parameter estimation.
    use_analytical : bool, default True
        If True, use analytical F and G functions for faster level computation.

    Returns
    -------
    RollingPairsTradingResults
        Object containing:
        - Cointegration: alpha, beta, p_value, coint_t
        - OU parameters: ou_mu, ou_theta, ou_sigma, half_life
        - Trading levels: entry_level, exit_level (if compute_levels=True)

    Notes
    -----
    The spread is computed as: spread = y0 - beta*y1 - alpha

    For a valid mean-reverting OU process, the OLS coefficient must be in (0, 1).
    Windows where this condition is not met will have NaN for OU parameters.

    Half-life represents the time for the spread to revert halfway to its mean:
    half_life = ln(2) / ou_mu

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 500
    >>> y0 = np.cumsum(np.random.randn(n))
    >>> y1 = y0 + np.random.randn(n) * 0.5
    >>> results = rolling_pairs_trading(y0, y1, window=100, lag=2)
    >>> print(f"Mean p-value: {np.nanmean(results.p_value):.4f}")
    >>> print(f"Mean half-life: {np.nanmean(results.half_life):.1f} periods")

    References
    ----------
    Engle, R.F. and Granger, C.W. (1987). "Co-integration and Error Correction"
    Leung, T. and Li, X. (2015). "Optimal Mean Reversion Trading"
    """
    # Input validation
    y0 = np.asarray(y0, dtype=np.float64).squeeze()
    y1 = np.asarray(y1, dtype=np.float64).squeeze()

    if y0.ndim != 1 or y1.ndim != 1:
        raise ValueError("y0 and y1 must be 1-dimensional")
    if len(y0) != len(y1):
        raise ValueError("y0 and y1 must have the same length")

    nobs = len(y0)
    k_vars = 2  # Two variables for cointegration

    if window < lag + 5:
        raise ValueError(f"window must be at least lag + 5 = {lag + 5}")

    # =========================================================================
    # STAGE 1: Rolling cointegrating regression
    # y0 = alpha + beta*y1 + residuals (spread)
    # =========================================================================

    # Prepare exogenous: [y1, const] or just [y1] if no trend
    if trend == "n":
        exog = y1.reshape(-1, 1)
    else:
        exog = sm.add_constant(y1)  # [const, y1] -> need to reorder
        # add_constant puts const first, we want [y1, const] for consistency
        exog = exog[:, ::-1]  # Now [y1, const]

    # Run RollingOLS for cointegrating regression
    rolling_coint_model = RollingOLS(
        endog=y0,
        exog=exog,
        window=window,
        min_nobs=min_nobs,
        expanding=expanding,
    )
    rolling_coint_results = rolling_coint_model.fit(method=method, params_only=False)

    # Extract parameters
    params = rolling_coint_results.params
    beta = params[:, 0].copy()  # Coefficient on y1

    if trend == "n":
        alpha = np.zeros(nobs)  # No intercept
    else:
        alpha = params[:, 1].copy()  # Constant term

    # =========================================================================
    # STAGE 2: Rolling ADF test on spread (cointegration test)
    # STAGE 3: Rolling OU parameter estimation on spread
    #
    # Both stages use the spread: spread = y0 - beta*y1 - alpha
    # =========================================================================

    # Initialize output arrays
    coint_t = np.full(nobs, np.nan)
    p_value = np.full(nobs, np.nan)

    # OU parameters
    ou_mu = np.full(nobs, np.nan)
    ou_theta = np.full(nobs, np.nan)
    ou_sigma = np.full(nobs, np.nan)
    half_life = np.full(nobs, np.nan)

    # Optimal trading levels
    entry_level = np.full(nobs, np.nan)
    exit_level = np.full(nobs, np.nan)
    loss_level = np.full(nobs, np.nan)

    # Residual statistics
    residual_mean = np.full(nobs, np.nan)
    residual_std = np.full(nobs, np.nan)

    # Determine starting index
    if expanding:
        first_idx = max(min_nobs if min_nobs is not None else 2, lag + 5)
    else:
        first_idx = window

    # Precompute critical values for cointegration test
    if trend != "n":
        adf_nobs_approx = window - lag - 1
        crit_vals = mackinnoncrit(N=k_vars, regression=trend, nobs=adf_nobs_approx - 1)

    # =========================================================================
    # Main loop: For each window, compute spread, ADF test, and OU params
    # =========================================================================

    for t in range(first_idx - 1, nobs):
        # Skip if parameters not available
        if np.any(np.isnan(params[t])):
            continue

        # Get window bounds
        if expanding:
            w_start = 0
        else:
            w_start = t - window + 1
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
            adf_y, adf_X = _build_adf_matrices(spread_window, lag)
            adf_nobs = len(adf_y)

            if adf_nobs >= lag + 2:
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
                p_value[t] = mackinnonp(t_stat, regression=trend, N=k_vars)

        except (np.linalg.LinAlgError, Exception):
            pass

        # Check if p_value is greater than the threshold
        if p_value[t] > p_value_threshold:
            ou_mu[t] = np.nan
            ou_theta[t] = np.nan
            ou_sigma[t] = np.nan
            half_life[t] = np.nan
            continue

        # -----------------------------------------------------------------
        # OU Parameter Estimation on spread
        # Regression: spread_{t+1} = intercept + coef * spread_t + noise
        # -----------------------------------------------------------------
        try:
            if len(spread_window) < 3:
                continue

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
                mu = -np.log(coef) / DELTA_T
                theta = intercept / (1 - coef)

                # Compute sigma from residuals
                ou_resid = spread_next - X_ou @ ou_params
                n_resid = len(ou_resid)
                residual_var = (
                    np.sum(ou_resid**2) / (n_resid - 2) if n_resid > 2 else np.nan
                )

                if residual_var > 0 and not np.isnan(residual_var):
                    ou_residual_std = np.sqrt(residual_var)
                    # ou_residual_std^2 = sigma^2 * (1 - exp(-2*mu*dt)) / (2*mu)
                    factor = (1 - np.exp(-2 * mu * DELTA_T)) / (2 * mu)
                    if factor > 0:
                        sigma = ou_residual_std / np.sqrt(factor)

                        ou_mu[t] = mu
                        ou_theta[t] = theta
                        ou_sigma[t] = sigma
                        half_life[t] = np.log(2) / mu

        except (np.linalg.LinAlgError, Exception):
            pass

    # =========================================================================
    # STAGE 4: Compute loss level and optimal entry/exit levels for all valid OU parameters
    # =========================================================================
    # Find valid OU parameters (not NaN and positive)
    valid_mask = (
        ~np.isnan(ou_mu)
        & ~np.isnan(ou_theta)
        & ~np.isnan(ou_sigma)
        & (ou_mu > 0)
        & (ou_sigma > 0)
    )

    if np.any(valid_mask):
        # Extract valid OU parameters
        valid_mu = ou_mu[valid_mask]
        valid_theta = ou_theta[valid_mask]
        valid_sigma = ou_sigma[valid_mask]
        valid_residual_std = residual_std[valid_mask]
        valid_residual_mean = residual_mean[valid_mask]

        # Compute loss level L for all valid OU parameters
        if L is None:
            valid_L = valid_residual_mean - stop_loss_factor * valid_residual_std
        else:
            L_arr = np.asarray(L)
            if L_arr.size == 1:
                valid_L = np.full(len(valid_mu), float(L_arr))
            else:
                valid_L = L_arr[valid_mask]

        # Store loss level
        loss_level[valid_mask] = valid_L

        # Compute entry and exit levels if requested
        if compute_levels:
            # Compute exit levels first (needed for entry levels)
            try:
                valid_exit = stochastic.OrnsteinUhlenbeck.get_optimal_exit_level(
                    mu=valid_mu,
                    sigma=valid_sigma,
                    theta=valid_theta,
                    r=r,
                    c=c,
                    L=valid_L,
                    use_analytical=use_analytical,
                )

                # Compute entry levels (uses exit levels internally)
                valid_entry = stochastic.OrnsteinUhlenbeck.get_optimal_entry_level(
                    mu=valid_mu,
                    sigma=valid_sigma,
                    theta=valid_theta,
                    r=r,
                    c=c,
                    L=valid_L,
                    b_star=valid_exit,
                    use_analytical=use_analytical,
                )

                # Assign back to full arrays
                exit_level[valid_mask] = valid_exit
                entry_level[valid_mask] = valid_entry

            except Exception:
                # If computation fails, leave as NaN
                pass

    return RollingPairsTradingResults(
        coint_t=coint_t,
        p_value=p_value,
        alpha=alpha,
        beta=beta,
        ou_mu=ou_mu,
        ou_theta=ou_theta,
        ou_sigma=ou_sigma,
        half_life=half_life,
        entry_level=entry_level,
        exit_level=exit_level,
        loss_level=loss_level,
        residual_mean=residual_mean,
        residual_std=residual_std,
        usedlag=lag,
    )
