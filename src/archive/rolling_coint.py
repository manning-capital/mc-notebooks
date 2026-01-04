"""
Rolling Cointegration Test

Implements a rolling window version of the Engle-Granger cointegration test,
leveraging statsmodels' RollingOLS for efficient computation of both the
cointegrating regression and the ADF test on residuals.
"""

import numpy as np
from typing import Literal, Optional
from dataclasses import dataclass
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import mackinnonp, mackinnoncrit
from statsmodels.tsa.tsatools import add_trend, lagmat


@dataclass
class RollingCointegrationResults:
    """
    Results from rolling cointegration test.

    The cointegrating regression is: y0 = alpha + beta*y1 + [trend terms] + residuals

    Attributes
    ----------
    coint_t : ndarray
        Rolling t-statistics from ADF test on residuals
    pvalue : ndarray
        Rolling p-values for cointegration test
    crit_1pct : ndarray
        Critical values at 1% significance level
    crit_5pct : ndarray
        Critical values at 5% significance level
    crit_10pct : ndarray
        Critical values at 10% significance level
    usedlag : int
        Number of lags used in ADF test (fixed for rolling)
    alpha : ndarray
        Rolling intercept (constant term) from cointegrating regression.
        None if trend="n" (no constant).
    beta : ndarray
        Rolling slope coefficients (hedge ratios) on y1.
        Shape: (nobs,) for single regressor, (nobs, n_y1_vars) for multiple.
    """

    coint_t: np.ndarray
    pvalue: np.ndarray
    crit_1pct: np.ndarray
    crit_5pct: np.ndarray
    crit_10pct: np.ndarray
    usedlag: int
    alpha: np.ndarray  # Intercept
    beta: np.ndarray  # Slope / hedge ratio

    def __repr__(self):
        valid = ~np.isnan(self.coint_t)
        return (
            f"RollingCointegrationResults(\n"
            f"  n_valid_windows={valid.sum()},\n"
            f"  usedlag={self.usedlag},\n"
            f"  mean_t_stat={np.nanmean(self.coint_t):.4f},\n"
            f"  mean_pvalue={np.nanmean(self.pvalue):.4f},\n"
            f"  mean_alpha={np.nanmean(self.alpha) if self.alpha is not None else 'N/A':.4f},\n"
            f"  mean_beta={np.nanmean(self.beta):.4f}\n"
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


def rolling_coint(
    y0: np.ndarray,
    y1: np.ndarray,
    window: int,
    trend: Literal["c", "ct", "ctt", "n"] = "c",
    method: Literal["inv", "lstsq", "pinv"] = "inv",
    lag: int = 1,
    min_nobs: Optional[int] = None,
    expanding: bool = False,
) -> RollingCointegrationResults:
    """
    Perform rolling window Engle-Granger cointegration test.

    This function efficiently computes cointegration test statistics over rolling
    windows using RollingOLS for both:
    1. The cointegrating regression (y0 ~ y1 + trend)
    2. The ADF test on residuals

    Parameters
    ----------
    y0 : array_like, 1d
        The first element in cointegrated system (dependent variable).
    y1 : array_like, 1d or 2d
        The remaining elements in cointegrated system (independent variables).
        Can be 1d for single regressor or 2d for multiple regressors.
    window : int
        Size of the rolling window for the cointegrating regression.
    trend : {"c", "ct", "ctt", "n"}, default "c"
        The trend term included in cointegrating equation:
        * "c" : constant only
        * "ct" : constant and linear trend
        * "ctt" : constant, linear and quadratic trend
        * "n" : no constant, no trend
    method : {"inv", "lstsq", "pinv"}, default "inv"
        Method for computing rolling regression parameters:
        * "inv" : matrix inversion (fastest)
        * "lstsq" : least squares
        * "pinv" : pseudo-inverse
    lag : int, default 1
        Fixed lag length for ADF test. Using a fixed lag enables efficient
        rolling computation. Set based on data frequency (e.g., 1-4 for daily).
    min_nobs : int, optional
        Minimum number of observations required in each window.
        Defaults to the number of regressors if not specified.
    expanding : bool, default False
        If True, use expanding window instead of rolling window.

    Returns
    -------
    RollingCointegrationResults
        Object containing rolling test statistics, p-values, critical values,
        and hedge ratios.

    Notes
    -----
    The null hypothesis is no cointegration. Small p-values indicate rejection
    of the null, suggesting cointegration exists.

    The implementation uses two rolling OLS regressions:

    1. Cointegrating regression: y0 = β*y1 + trend + residuals
       Uses RollingOLS with the specified window.

    2. ADF on residuals: Δresid = ρ*resid_{t-1} + Σγ_i*Δresid_{t-i} + error
       Uses RollingOLS with window = (cointegration_window - lag - 1)
       The test statistic is t_ρ = ρ_hat / se(ρ_hat)

    Using a fixed lag (instead of autolag) is necessary for efficient rolling
    computation. For most applications, lag=1 to 4 works well.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 500
    >>> y0 = np.cumsum(np.random.randn(n))
    >>> y1 = y0 + np.random.randn(n) * 0.5  # Cointegrated series
    >>> results = rolling_coint(y0, y1, window=100, lag=2)
    >>> print(f"Mean p-value: {np.nanmean(results.pvalue):.4f}")

    References
    ----------
    Engle, R.F. and Granger, C.W. (1987). "Co-integration and Error Correction:
    Representation, Estimation, and Testing". Econometrica, 55(2), 251-276.
    """
    # Input validation and preparation
    y0 = np.asarray(y0, dtype=np.float64).squeeze()
    if y0.ndim != 1:
        raise ValueError("y0 must be 1-dimensional")

    y1 = np.asarray(y1, dtype=np.float64)
    if y1.ndim == 1:
        y1 = y1.reshape(-1, 1)
    elif y1.ndim != 2:
        raise ValueError("y1 must be 1d or 2d")

    if len(y0) != len(y1):
        raise ValueError("y0 and y1 must have the same length")

    nobs = len(y0)
    k_vars = y1.shape[1] + 1  # +1 for y0

    if window < lag + 5:
        raise ValueError(f"window must be at least lag + 5 = {lag + 5}")

    # Prepare exogenous variables with trend
    if trend == "n":
        exog = y1
    else:
        exog = add_trend(y1, trend=trend, prepend=False)

    n_exog = exog.shape[1]

    # =========================================================================
    # STAGE 1: Rolling cointegrating regression using RollingOLS
    # y0 = β*y1 + trend + residuals
    # =========================================================================
    rolling_coint_model = RollingOLS(
        endog=y0,
        exog=exog,
        window=window,
        min_nobs=min_nobs,
        expanding=expanding,
    )
    rolling_coint_results = rolling_coint_model.fit(method=method, params_only=False)

    # Compute residuals for EACH window
    # residuals[t] uses params[t] to compute y0[t] - exog[t] @ params[t]
    # But for ADF, we need the FULL window of residuals computed with params[t]

    # We'll compute residuals in a rolling manner
    # For window ending at t: resid[t-w+1:t+1] = y0[t-w+1:t+1] - exog[t-w+1:t+1] @ params[t]

    # Get rolling parameters (shape: nobs x n_exog)
    params = rolling_coint_results.params

    # Extract beta (coefficients on y1) and alpha (intercept)
    # Parameter order from add_trend with prepend=False: [y1_vars..., const, trend, trend^2]
    n_y1_vars = y1.shape[1]
    beta = params[:, :n_y1_vars].copy()
    if n_y1_vars == 1:
        beta = beta.squeeze()  # Make 1D for single regressor case

    # Extract alpha (intercept) - it's the first trend term after y1 variables
    if trend == "n":
        alpha = None  # No intercept
    else:
        alpha = params[:, n_y1_vars].copy()  # Constant term

    # =========================================================================
    # STAGE 2: Rolling ADF test on residuals using RollingOLS
    # For each window, compute residuals then run ADF
    #
    # Key insight: We can set up the ADF regression matrices globally and use
    # RollingOLS, but the residuals themselves depend on rolling params.
    #
    # Solution: Compute window-specific residuals, build ADF matrices, and
    # run the ADF OLS in a vectorized manner.
    # =========================================================================

    # Initialize output arrays
    coint_t = np.full(nobs, np.nan)
    pvalue = np.full(nobs, np.nan)
    crit_1pct = np.full(nobs, np.nan)
    crit_5pct = np.full(nobs, np.nan)
    crit_10pct = np.full(nobs, np.nan)

    # Determine starting index
    if expanding:
        first_idx = max(min_nobs if min_nobs is not None else n_exog, lag + 5)
    else:
        first_idx = window

    # Precompute critical values (they depend on nobs and trend, constant for fixed window)
    if trend != "n":
        # For rolling window, ADF nobs = window - lag - 1
        adf_nobs = window - lag - 1
        crit_vals = mackinnoncrit(N=k_vars, regression=trend, nobs=adf_nobs - 1)
    else:
        crit_vals = [np.nan, np.nan, np.nan]

    # =========================================================================
    # Efficient rolling ADF computation
    #
    # For each window ending at t:
    # 1. Compute window residuals using params[t]
    # 2. Build ADF matrices (Δresid ~ resid_lag + Δresid_lags)
    # 3. Run OLS and get t-statistic for ρ
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

        # Compute residuals for this window using THIS window's params
        y0_window = y0[w_start:w_end]
        exog_window = exog[w_start:w_end]
        resid_window = y0_window - exog_window @ params[t]

        # Check for degenerate cases
        if np.std(resid_window) < 1e-10:
            continue

        # Build ADF matrices
        try:
            adf_y, adf_X = _build_adf_matrices(resid_window, lag)
        except Exception:
            continue

        # Minimum observations for ADF regression
        adf_nobs = len(adf_y)
        if adf_nobs < lag + 2:
            continue

        # Run ADF OLS: adf_y ~ adf_X
        # We need params and standard errors
        try:
            # Using numpy lstsq for efficiency
            # X'X
            XtX = adf_X.T @ adf_X
            # X'y
            Xty = adf_X.T @ adf_y
            # (X'X)^{-1}
            XtX_inv = np.linalg.inv(XtX)
            # params = (X'X)^{-1} X'y
            adf_params = XtX_inv @ Xty

            # Residuals and SSR
            adf_resid = adf_y - adf_X @ adf_params
            ssr = np.sum(adf_resid**2)

            # Degrees of freedom
            df = adf_nobs - adf_X.shape[1]

            # MSE
            mse = ssr / df

            # Standard errors: sqrt(diag((X'X)^{-1} * mse))
            se = np.sqrt(np.diag(XtX_inv) * mse)

            # t-statistic for rho (first coefficient)
            t_stat = adf_params[0] / se[0]

            coint_t[t] = t_stat

            # P-value using MacKinnon approximation
            pvalue[t] = mackinnonp(t_stat, regression=trend, N=k_vars)

            # Critical values
            if trend != "n":
                if expanding:
                    crit_vals = mackinnoncrit(
                        N=k_vars, regression=trend, nobs=adf_nobs - 1
                    )
                crit_1pct[t] = crit_vals[0]
                crit_5pct[t] = crit_vals[1]
                crit_10pct[t] = crit_vals[2]

        except np.linalg.LinAlgError:
            continue

    return RollingCointegrationResults(
        coint_t=coint_t,
        pvalue=pvalue,
        crit_1pct=crit_1pct,
        crit_5pct=crit_5pct,
        crit_10pct=crit_10pct,
        usedlag=lag,
        alpha=alpha,
        beta=beta,
    )


def plot_rolling_coint_results(
    results: RollingCointegrationResults,
    title: str = "Rolling Cointegration Test Results",
    figsize: tuple = (12, 10),
):
    """
    Plot rolling cointegration test results.

    Parameters
    ----------
    results : RollingCointegrationResults
        Results from rolling_coint function
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting")

    beta = results.beta
    if beta.ndim == 1:
        beta = beta.reshape(-1, 1)
    n_y1 = beta.shape[1]

    # Add extra subplot for alpha if it exists
    n_plots = 2 + n_y1 + (1 if results.alpha is not None else 0)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)

    # Plot 1: Test statistic
    valid_crit = results.crit_5pct[~np.isnan(results.crit_5pct)]
    axes[0].plot(results.coint_t, label="Test Statistic", linewidth=1.5)
    if len(valid_crit) > 0:
        axes[0].axhline(
            y=valid_crit.mean(),
            color="r",
            linestyle="--",
            label="5% Critical Value",
            alpha=0.7,
        )
    axes[0].set_ylabel("ADF t-statistic")
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: P-value
    axes[1].plot(results.pvalue, label="P-value", linewidth=1.5, color="orange")
    axes[1].axhline(
        y=0.05, color="r", linestyle="--", label="5% Significance", alpha=0.7
    )
    axes[1].axhline(
        y=0.10, color="y", linestyle="--", label="10% Significance", alpha=0.7
    )
    axes[1].set_ylabel("P-value")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])

    # Plot 3+: Beta (slope / hedge ratio)
    plot_idx = 2
    for i in range(n_y1):
        axes[plot_idx].plot(
            beta[:, i],
            label=f"β_{i + 1} (slope/hedge ratio)",
            linewidth=1.5,
            color="green",
        )
        axes[plot_idx].set_ylabel(f"Beta {i + 1}")
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

    # Plot alpha (intercept) if present
    if results.alpha is not None:
        axes[plot_idx].plot(
            results.alpha, label="α (intercept)", linewidth=1.5, color="purple"
        )
        axes[plot_idx].set_ylabel("Alpha")
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")

    plt.tight_layout()
    return fig, axes
