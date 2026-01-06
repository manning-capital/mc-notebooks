from typing import Optional
import numpy as np
import statsmodels.api as sm
from scipy.integrate import quad
from scipy.special import pbdv, gammaln
from scipy.stats import norm
from typing import Literal


def newton_vectorized(
    f: callable,
    df_dx: callable,
    x0: np.ndarray | float,
    max_iter: int = 20,
    tol: float = 1e-8,
    step_direction: float | Literal["positive", "negative"] | None = None,
) -> np.ndarray | float:
    """
    Vectorized Newton's method for root finding.

    Parameters
    ----------
    f : callable
        Function f(x) to find roots of. Should accept array x and return array of same shape.
    df_dx : callable
        Derivative function df/dx. Should accept array x and return array of same shape.
    x0 : np.ndarray or float
        Initial guess(es) for the root(s)
    max_iter : int, optional
        Maximum number of iterations (default: 20)
    tol : float, optional
        Tolerance for convergence (default: 1e-8)
    step_direction : float, str, or None, optional
        Direction to move in terms of the derivative from the initial guess.
        - If None or "auto": Standard Newton direction (x - dx)
        - If float: Multiplier for the step direction. x = x - step_direction * dx
          - 1.0: Standard Newton (default behavior)
          - -1.0: Reverse direction
          - 0.5: Take smaller steps
        - If "positive": Move in positive direction when derivative is positive,
          negative direction when derivative is negative (x = x - sign(fpx) * dx)
        - If "negative": Move in negative direction when derivative is positive,
          positive direction when derivative is negative (x = x + sign(fpx) * dx)

    Returns
    -------
    np.ndarray or float
        Root(s) found by Newton's method. Returns scalar if input was scalar.
    """
    # Convert to array and remember if input was scalar
    x0_arr = np.asarray(x0)
    was_scalar = x0_arr.ndim == 0
    if was_scalar:
        x0_arr = x0_arr.reshape(1)

    x = x0_arr.copy()

    # Determine step direction multiplier
    if step_direction is None or step_direction == "auto":
        direction_mult = 1.0
    elif isinstance(step_direction, (int, float)):
        direction_mult = float(step_direction)
    elif step_direction == "positive":
        # Move in positive direction based on derivative sign
        # When f'(x) > 0, move in positive direction; when f'(x) < 0, move in negative direction
        direction_mult = None  # Will be set dynamically
    elif step_direction == "negative":
        # Move in negative direction based on derivative sign
        # When f'(x) > 0, move in negative direction; when f'(x) < 0, move in positive direction
        direction_mult = None  # Will be set dynamically
    else:
        raise ValueError(
            f"step_direction must be None, 'auto', a float, 'positive', or 'negative', got {step_direction}"
        )

    for _ in range(max_iter):
        fx = f(x)
        fpx = df_dx(x)

        # Ensure fx and fpx are arrays with same shape as x
        fx = np.asarray(fx)
        fpx = np.asarray(fpx)
        if fx.shape != x.shape:
            fx = np.broadcast_to(fx, x.shape)
        if fpx.shape != x.shape:
            fpx = np.broadcast_to(fpx, x.shape)

        # Avoid division by zero
        mask = np.abs(fpx) > 1e-12
        dx = np.zeros_like(x)
        dx[mask] = fx[mask] / fpx[mask]

        # Determine step direction multiplier for this iteration
        if direction_mult is None:
            if step_direction == "positive":
                # Move in direction of increasing x when derivative is positive
                # This means: if f'(x) > 0, use positive step; if f'(x) < 0, use negative step
                # Standard Newton is x = x - f/f', so we want to reverse when f' < 0
                current_direction = np.where(fpx > 0, 1.0, -1.0)
            elif step_direction == "negative":
                # Move in direction of decreasing x when derivative is positive
                # This means: if f'(x) > 0, use negative step; if f'(x) < 0, use positive step
                current_direction = np.where(fpx > 0, -1.0, 1.0)
            else:
                current_direction = 1.0
        else:
            current_direction = direction_mult

        # Ensure current_direction is properly shaped
        if isinstance(current_direction, np.ndarray):
            if current_direction.shape != x.shape:
                current_direction = np.broadcast_to(current_direction, x.shape)
        else:
            # Scalar - broadcast to x.shape
            current_direction = np.full(x.shape, current_direction)

        x = x - current_direction * dx

        # Check convergence
        step_size = np.abs(current_direction * dx)
        if np.max(step_size) < tol:
            break

    # Return scalar if input was scalar
    if was_scalar:
        return float(x.item())
    return x


def newton_with_multiple_guesses(
    f: callable,
    df_dx: callable,
    guesses: np.ndarray,
    max_iter: int = 20,
    tol: float = 1e-8,
    valid_range: tuple = (None, None),
) -> np.ndarray:
    """
    Newton's method with multiple initial guesses, returning the best result.

    Tries each guess and selects the one with the smallest residual |f(x)|.
    Only considers results within the valid_range if provided.

    Parameters
    ----------
    f : callable
        Function f(x) to find roots of (expects array input matching guesses[0].shape)
    df_dx : callable
        Derivative function df/dx (expects array input matching guesses[0].shape)
    guesses : np.ndarray
        Array of initial guesses to try (shape: [n_guesses, ...])
    max_iter : int, optional
        Maximum number of iterations per guess (default: 20)
    tol : float, optional
        Tolerance for convergence (default: 1e-8)
    valid_range : tuple, optional
        (lower_bound, upper_bound) for valid results. None means no bound.
        Only results in this range are considered. Default: (None, None)

    Returns
    -------
    np.ndarray
        Best root found (shape matches guesses[0])
    """
    n_guesses = guesses.shape[0]
    target_shape = guesses.shape[1:]
    results = np.zeros_like(guesses[0])
    residuals = np.full(target_shape, np.inf)

    lower_bound, upper_bound = valid_range
    if lower_bound is not None:
        lower_bound = np.asarray(lower_bound)
        if lower_bound.shape != target_shape:
            lower_bound = np.broadcast_to(lower_bound, target_shape)
    if upper_bound is not None:
        upper_bound = np.asarray(upper_bound)
        if upper_bound.shape != target_shape:
            upper_bound = np.broadcast_to(upper_bound, target_shape)

    # Try each guess
    for i in range(n_guesses):
        try:
            guess = guesses[i]
            # Run Newton's method with this guess
            result = newton_vectorized(f, df_dx, guess, max_iter=max_iter, tol=tol)

            # Ensure result is properly shaped
            result = np.asarray(result)
            if result.shape != target_shape:
                if result.ndim == 0:
                    # Scalar - broadcast to target shape
                    result = np.full(target_shape, result.item())
                else:
                    result = np.broadcast_to(result, target_shape)

            # Check if result is in valid range
            valid_mask = np.ones(target_shape, dtype=bool)
            if lower_bound is not None:
                valid_mask = valid_mask & (result > lower_bound)
            if upper_bound is not None:
                valid_mask = valid_mask & (result < upper_bound)

            # Only evaluate residual for valid results
            if np.any(valid_mask):
                # Evaluate function at result to get residual
                fx = f(result)
                fx = np.asarray(fx)
                if fx.shape != target_shape:
                    fx = np.broadcast_to(fx, target_shape)
                residual = np.abs(fx)

                # Only update where result is valid AND gives a better residual
                better_mask = valid_mask & (residual < residuals)
                results[better_mask] = result[better_mask]
                residuals[better_mask] = residual[better_mask]

        except Exception:
            # If this guess fails, continue to next guess
            continue

    # If all guesses failed (all residuals still inf), use first guess as fallback
    failed_mask = np.isinf(residuals)
    if np.any(failed_mask):
        results[failed_mask] = guesses[0][failed_mask]

    return results


from .base import (
    DELTA_T,
    StochasticModel,
    GeometricBrownianMotionResult,
    OrnsteinUhlenbeckResult,
)


class GeometricBrownianMotion(StochasticModel):
    """
    Geometric Brownian Motion process.
    """

    def __init__(self, params: GeometricBrownianMotionResult = None):
        super().__init__(params)

    def log_likelihood(self, X: np.ndarray) -> float:
        """
        Calculates the log-likelihood for a Geometric Brownian Motion model.

        The GBM SDE: dS_t = mu*S_t*dt + sigma*S_t*dW_t
        Log-returns follow: r ~ N((mu - 0.5*sigma²)*dt, sigma²*dt)

        Uses the global DELTA_T constant for time step.

        Args:
            X (np.ndarray): Array of asset prices

        Returns:
            float: The negative log-likelihood (for minimization)
        """
        # Calculate log-returns from prices
        log_returns = np.diff(np.log(X))
        n = len(log_returns)

        # Expected mean of log-returns
        expected_mean = (self.params.mu - 0.5 * self.params.sigma**2) * DELTA_T

        # Calculate log-likelihood terms
        term1 = -0.5 * n * np.log(2 * np.pi)
        term2 = -0.5 * n * np.log(self.params.sigma**2 * DELTA_T)
        term3 = -np.sum((log_returns - expected_mean) ** 2) / (
            2 * self.params.sigma**2 * DELTA_T
        )

        log_likelihood = term1 + term2 + term3

        # Return negative for minimization
        return -log_likelihood

    def simulate(self, N: int, N_simulated: int, X_0: float) -> np.ndarray:
        """
        Simulates GBM paths using the Euler-Maruyama scheme.

        Following the approach from:
        https://towardsdatascience.com/stochastic-processes-simulation-the-ornstein-uhlenbeck-process-e8bff820f3/

        GBM SDE: dS = mu*S*dt + sigma*S*dW
        Solution: S_t = S_0 * exp((mu - 0.5*sigma²)*t + sigma*W_t)

        Uses the global DELTA_T constant for time step.

        input: N - number of time steps
               N_simulated - number of paths to simulate
               X_0 - initial value
        returns: np.ndarray of shape (N_simulated, N) with simulated paths
        """
        # Initialize the simulated paths
        X_simulated = np.zeros((N_simulated, N))
        X_simulated[:, 0] = X_0

        # Simulate using the exact solution at each time step
        for i in range(1, N):
            X_simulated[:, i] = X_simulated[:, i - 1] * np.exp(
                (self.params.mu - 0.5 * self.params.sigma**2) * DELTA_T
                + self.params.sigma
                * np.sqrt(DELTA_T)
                * np.random.normal(0, 1, N_simulated)
            )

        return X_simulated

    def fit(self, X: np.ndarray) -> GeometricBrownianMotionResult:
        """
        Estimates Geometric Brownian Motion parameters from price data.

        Following the moment matching approach similar to:
        https://towardsdatascience.com/stochastic-processes-simulation-the-ornstein-uhlenbeck-process-e8bff820f3/

        The GBM SDE is: dS = mu*S*dt + sigma*S*dW
        Taking logs: d(log S) = (mu - 0.5*sigma²)*dt + sigma*dW

        Log-returns follow: r = log(S_t/S_{t-1}) ~ N((mu - 0.5*sigma²)*dt, sigma²*dt)

        Uses the global DELTA_T constant for time step.

        Parameters estimated by moment matching:
        - Var(r) = sigma²*DELTA_T  =>  sigma = sqrt(Var(r)/DELTA_T)
        - E[r] = (mu - 0.5*sigma²)*DELTA_T  =>  mu = E[r]/DELTA_T + 0.5*sigma²

        input: X - array-like price data
        returns: GeometricBrownianMotionResult with estimated mu and sigma
        """
        # Calculate log-returns
        log_returns = np.diff(np.log(X))

        # Estimate sigma from variance of log-returns
        # Var(r) = sigma²*DELTA_T
        sigma = np.sqrt(np.var(log_returns, ddof=1) / DELTA_T)

        # Estimate mu from mean of log-returns
        # E[r] = (mu - 0.5*sigma²)*DELTA_T
        mu = np.mean(log_returns) / DELTA_T + 0.5 * sigma**2

        # Update model parameters
        self.params = GeometricBrownianMotionResult(mu=mu, sigma=sigma)

        return self.params


class OrnsteinUhlenbeck(StochasticModel):
    """
    Ornstein-Uhlenbeck process.

    The Ornstein-Uhlenbeck process is defined by:

    dX_t = mu * (theta - X_t) dt + sigma dW_t

    where:
    - mu is the mean reversion rate (speed of mean reversion)
    - theta is the long-term mean (asymptotic mean)
    - sigma is the volatility parameter
    - W_t is a Wiener process

    This matches the standard parameterization used in Leung & Li (2015).
    """

    def __init__(self, params: OrnsteinUhlenbeckResult = None):
        super().__init__(params)

    def log_likelihood(self, X: np.ndarray) -> float:
        """
        Computes the log likelihood of the OU process.

        Uses the global DELTA_T constant for time step.
        """
        # Get the number of observations.
        n = len(X)

        # Get the lag and next values.
        X_lag = X[:-1]
        X_next = X[1:]

        # Get the tilde sigma.
        tilde_sigma = self.params.sigma * np.sqrt(
            (1 - np.exp(-2 * self.params.mu * DELTA_T)) / (2 * self.params.mu)
        )

        # Compute the log likelihood.
        log_likelihood = (
            -0.5 * np.log(2 * np.pi)
            - np.log(tilde_sigma)
            - 1
            / (2 * n * tilde_sigma**2)
            * np.sum(
                (
                    X_next
                    - X_lag * np.exp(-self.params.mu * DELTA_T)
                    - self.params.theta * (1 - np.exp(-self.params.mu * DELTA_T))
                )
                ** 2
            )
        )

        return -log_likelihood

    def fit(self, X: np.ndarray) -> OrnsteinUhlenbeckResult:
        """
        Estimates Ornstein-Uhlenbeck parameters from the given array using OLS regression
        on the exact discrete-time solution (not the Euler approximation).

        The exact OU discrete transition is:
        X_{t+dt} = theta*(1 - exp(-mu*dt)) + X_t*exp(-mu*dt) + sigma*sqrt((1-exp(-2*mu*dt))/(2*mu))*noise

        Letting a = exp(-mu*dt), we can write:
        X_{t+dt} = theta*(1 - a) + X_t*a + noise

        OLS regression of X_{t+1} on X_t gives:
        - intercept = theta*(1 - a)
        - coef = a = exp(-mu*dt)

        Therefore:
        - mu = -log(coef) / dt
        - theta = intercept / (1 - coef)

        input: X - array-like data to be fit as an OU process
        returns: OrnsteinUhlenbeckResult
        """
        # Regress X_{t+1} on X_t (not differences!)
        X_next = X[1:]
        X_lag = X[:-1]
        X_with_const = sm.add_constant(X_lag)

        # Fit OLS regression: X_{t+1} = intercept + coef*X_t
        model = sm.OLS(X_next, X_with_const)
        results = model.fit()

        # Extract coefficients: [intercept, coef]
        intercept = results.params[0]
        coef = results.params[1]

        # Extract OU parameters from exact solution
        # coef = exp(-mu*DELTA_T), which must be in (0, 1) for a valid mean-reverting OU process
        if not (0 < coef < 1):
            raise ValueError(
                f"Invalid OLS coefficient {coef:.6f}. For a mean-reverting OU process, "
                f"the coefficient must be in (0, 1) since it equals exp(-mu*DELTA_T). "
                f"This indicates the data does not follow an OU process or has insufficient variation."
            )

        mu = -np.log(coef) / DELTA_T
        theta = intercept / (1 - coef)

        # Get residual standard deviation
        # residuals = X_{t+1} - (theta*(1-a) + X_t*a)
        # Theoretical: sigma_residual = sigma * sqrt((1 - exp(-2*mu*dt)) / (2*mu))
        residual_std = np.sqrt(results.mse_resid)

        # Back out sigma from residual_std
        # residual_std^2 = sigma^2 * (1 - exp(-2*mu*dt)) / (2*mu)
        sigma = residual_std * np.sqrt(2 * mu / (1 - np.exp(-2 * mu * DELTA_T)))

        # Update the parameters.
        self.params = OrnsteinUhlenbeckResult(mu=mu, theta=theta, sigma=sigma)

        return self.params

    def simulate(self, N: int, N_simulated: int, X_0: float) -> np.ndarray:
        """
        Simulates the OU process.

        Uses the global DELTA_T constant for time step.
        """
        # Initialize the simulated process.
        X_simulated = np.zeros((N_simulated, N))
        X_simulated[:, 0] = X_0  # initial value

        # Simulate the process.
        for i in range(1, N):
            X_simulated[:, i] = (
                X_simulated[:, i - 1] * np.exp(-self.params.mu * DELTA_T)
                + self.params.theta * (1 - np.exp(-self.params.mu * DELTA_T))
                + self.params.sigma
                * np.sqrt(
                    (1 - np.exp(-2 * self.params.mu * DELTA_T)) / (2 * self.params.mu)
                )
                * np.random.normal(0, 1, N_simulated)
            )

        return X_simulated

    @staticmethod
    def f(
        u: float, x: float, mu: float, sigma: float, theta: float, r: float = 0.01
    ) -> float:
        """
        Computes the f function.
        f(u, r) = u^((r/mu) - 1) * e^(sqrt((2*mu)/(sigma^2)) * (x - theta) * u  - u^2 / 2)
        """
        return u ** ((r / mu) - 1) * np.exp(
            np.sqrt((2 * mu) / (sigma**2)) * (x - theta) * u - u**2 / 2
        )

    @staticmethod
    def F(
        x: float | np.ndarray,
        mu: float | np.ndarray,
        sigma: float | np.ndarray,
        theta: float | np.ndarray,
        r: float = 0.01,
        use_analytical: bool = False,
    ) -> float | np.ndarray:
        """
        Computes the F function.
        F(x, r) = integral f(u, x, r) du from 0 to infinity

        By default uses numerical integration with quad for accuracy.
        An experimental analytical approximation using parabolic cylinder functions
        is available with use_analytical=True, but has ~1-2% error that varies with parameters.

        Parameters
        ----------
        x : float or ndarray
            Single value or array of x values
        mu : float or ndarray
            Mean reversion speed parameter(s)
        sigma : float or ndarray
            Volatility parameter(s)
        theta : float or ndarray
            Long-term mean parameter(s)
        r : float, optional
            Discount rate (default: 0.01)
        use_analytical : bool, optional
            If True, use experimental analytical approximation (faster but less accurate).
            If False, use numerical integration with quad (default, most accurate).

        Returns
        -------
        float or ndarray
            Scalar if all inputs are scalar, otherwise array with shape
            determined by broadcasting the input arrays

        Notes
        -----
        The integral has a singularity at u=0 when r/mu < 1. The quad integrator
        handles this well using the Wynn epsilon algorithm.

        The analytical approximation provides 10-100x speedup but has accuracy issues
        (~1-2% error) that vary with the parameters, particularly with r/mu ratio.
        Use only if speed is critical and small errors are acceptable.
        """
        if not use_analytical:
            # Use numerical integration (default, most accurate)
            all_scalar = (
                np.isscalar(x)
                and np.isscalar(mu)
                and np.isscalar(sigma)
                and np.isscalar(theta)
            )

            if all_scalar:
                return quad(
                    lambda u: OrnsteinUhlenbeck.f(
                        u, x, mu=mu, sigma=sigma, theta=theta, r=r
                    ),
                    0,
                    np.inf,
                )[0]

            # For arrays, compute element-wise (arrays must have equal lengths)
            x_arr = np.atleast_1d(x)
            mu_arr = np.atleast_1d(mu)
            sigma_arr = np.atleast_1d(sigma)
            theta_arr = np.atleast_1d(theta)

            # Broadcast to same shape if needed, then compute element-wise
            x_arr, mu_arr, sigma_arr, theta_arr = np.broadcast_arrays(
                x_arr, mu_arr, sigma_arr, theta_arr
            )

            result = np.empty(x_arr.shape)
            for idx in np.ndindex(x_arr.shape):
                result[idx] = quad(
                    lambda u: OrnsteinUhlenbeck.f(
                        u,
                        x_arr[idx],
                        mu=mu_arr[idx],
                        sigma=sigma_arr[idx],
                        theta=theta_arr[idx],
                        r=r,
                    ),
                    0,
                    np.inf,
                )[0]

            if result.ndim == 0:
                return float(result)
            return result

        # Robust analytical approximation using parabolic cylinder functions
        # Always uses pbdv with corrected prefactor (no 2^power term)
        # Based on: ∫₀^∞ u^α exp(βu - u²/2) du = exp(β²/4) * Γ(α+1) * D_{-α-1}(-β)

        # Convert inputs to arrays and broadcast for element-wise computation
        x_arr = np.atleast_1d(x)
        mu_arr = np.atleast_1d(mu)
        sigma_arr = np.atleast_1d(sigma)
        theta_arr = np.atleast_1d(theta)

        # Broadcast to same shape if needed, then compute element-wise
        x_arr, mu_arr, sigma_arr, theta_arr = np.broadcast_arrays(
            x_arr, mu_arr, sigma_arr, theta_arr
        )

        # Compute parameters element-wise
        alpha = (r / mu_arr) - 1
        beta = np.sqrt(2 * mu_arr / sigma_arr**2) * (x_arr - theta_arr)

        # Initialize result array
        result = np.empty(x_arr.shape, dtype=float)

        # Always use pbdv with corrected prefactor (no 2^power term)
        # Special case: alpha ≈ 0 uses exact formula
        alpha_zero_mask = np.isclose(alpha, 0, atol=1e-10)
        non_zero_mask = ~alpha_zero_mask

        # Special case: alpha ≈ 0 (exact formula, vectorized)
        if np.any(alpha_zero_mask):
            result[alpha_zero_mask] = (
                np.exp(beta[alpha_zero_mask] ** 2 / 2)
                * np.sqrt(2 * np.pi)
                * norm.cdf(beta[alpha_zero_mask])
            )

        # All other cases: use parabolic cylinder function (vectorized)
        if np.any(non_zero_mask):
            alpha_nonzero = alpha[non_zero_mask]
            beta_nonzero = beta[non_zero_mask]

            # Compute nu for all cases
            nu_nonzero = -(alpha_nonzero + 1)

            # Compute D values - pbdv can handle arrays
            D_vals, _ = pbdv(nu_nonzero, -beta_nonzero)

            # Prefactor: exp(β²/4) * Γ(α+1) * D_{-α-1}(-β)
            # Note: The 2^power term cancels out (2^((α-1)/2) * 2^((1-α)/2) = 1)
            # So we use just exp(β²/4) * Γ(α+1)
            log_prefactor = gammaln(alpha_nonzero + 1) + beta_nonzero**2 / 4

            # Compute with pbdv (works for all practical cases)
            non_zero_result = np.exp(log_prefactor) * D_vals

            # Check for overflow/invalid results
            invalid_mask = ~np.isfinite(non_zero_result)
            if np.any(invalid_mask):
                # For overflow cases, set to a large but finite value
                # or raise warning - in practice these are extremely rare
                non_zero_result[invalid_mask] = np.inf

            # Assign back to result array
            result[non_zero_mask] = non_zero_result

        if result.ndim == 0:
            return float(result)

        return result

    @staticmethod
    def g(
        u: float, x: float, mu: float, sigma: float, theta: float, r: float = 0.01
    ) -> float:
        """
        Computes the g function.
        g(u, x, r) = u^((r/mu) - 1) * e^(sqrt((2*mu)/(sigma^2)) * (theta - x) * u  - u^2 / 2)
        """
        return u ** ((r / mu) - 1) * np.exp(
            np.sqrt((2 * mu) / (sigma**2)) * (theta - x) * u - u**2 / 2
        )

    @staticmethod
    def G(
        x: float | np.ndarray,
        mu: float | np.ndarray,
        sigma: float | np.ndarray,
        theta: float | np.ndarray,
        r: float = 0.01,
        use_analytical: bool = False,
    ) -> float | np.ndarray:
        """
        Computes the G function.
        G(x, r) = integral g(u, x, r) du from 0 to infinity

        By default uses numerical integration with quad for accuracy.
        An experimental analytical approximation using parabolic cylinder functions
        is available with use_analytical=True, but has ~1-2% error that varies with parameters.

        Parameters
        ----------
        x : float or ndarray
            Single value or array of x values
        mu : float or ndarray
            Mean reversion speed parameter(s)
        sigma : float or ndarray
            Volatility parameter(s)
        theta : float or ndarray
            Long-term mean parameter(s)
        r : float, optional
            Discount rate (default: 0.01)
        use_analytical : bool, optional
            If True, use experimental analytical approximation (faster but less accurate).
            If False, use numerical integration with quad (default, most accurate).

        Returns
        -------
        float or ndarray
            Scalar if all inputs are scalar, otherwise array with shape
            determined by broadcasting the input arrays

        Notes
        -----
        The integral has a singularity at u=0 when r/mu < 1. The quad integrator
        handles this well using the Wynn epsilon algorithm.

        The analytical approximation provides 10-100x speedup but has accuracy issues
        (~1-2% error) that vary with the parameters. Use only if speed is critical
        and small errors are acceptable.
        """
        if not use_analytical:
            # Use numerical integration (default, most accurate)
            all_scalar = (
                np.isscalar(x)
                and np.isscalar(mu)
                and np.isscalar(sigma)
                and np.isscalar(theta)
            )

            if all_scalar:
                return quad(
                    lambda u: OrnsteinUhlenbeck.g(
                        u, x, mu=mu, sigma=sigma, theta=theta, r=r
                    ),
                    0,
                    np.inf,
                )[0]

            # For arrays, compute element-wise (arrays must have equal lengths)
            x_arr = np.atleast_1d(x)
            mu_arr = np.atleast_1d(mu)
            sigma_arr = np.atleast_1d(sigma)
            theta_arr = np.atleast_1d(theta)

            # Broadcast to same shape if needed, then compute element-wise
            x_arr, mu_arr, sigma_arr, theta_arr = np.broadcast_arrays(
                x_arr, mu_arr, sigma_arr, theta_arr
            )

            result = np.empty(x_arr.shape)
            for idx in np.ndindex(x_arr.shape):
                result[idx] = quad(
                    lambda u: OrnsteinUhlenbeck.g(
                        u,
                        x_arr[idx],
                        mu=mu_arr[idx],
                        sigma=sigma_arr[idx],
                        theta=theta_arr[idx],
                        r=r,
                    ),
                    0,
                    np.inf,
                )[0]

            if result.ndim == 0:
                return float(result)
            return result

        # Robust analytical approximation using parabolic cylinder functions
        # Always uses pbdv with corrected prefactor (no 2^power term)
        # Based on: ∫₀^∞ u^α exp(βu - u²/2) du = exp(β²/4) * Γ(α+1) * D_{-α-1}(-β) where β = √(2μ/σ²)(θ-x) for G

        # Convert inputs to arrays and broadcast for element-wise computation
        x_arr = np.atleast_1d(x)
        mu_arr = np.atleast_1d(mu)
        sigma_arr = np.atleast_1d(sigma)
        theta_arr = np.atleast_1d(theta)

        # Broadcast to same shape if needed, then compute element-wise
        x_arr, mu_arr, sigma_arr, theta_arr = np.broadcast_arrays(
            x_arr, mu_arr, sigma_arr, theta_arr
        )

        # Compute parameters element-wise (note: theta - x for G, vs x - theta for F)
        alpha = (r / mu_arr) - 1
        beta = np.sqrt(2 * mu_arr / sigma_arr**2) * (theta_arr - x_arr)

        # Initialize result array
        result = np.empty(x_arr.shape, dtype=float)

        # Always use pbdv with corrected prefactor (no 2^power term)
        # Special case: alpha ≈ 0 uses exact formula
        alpha_zero_mask = np.isclose(alpha, 0, atol=1e-10)
        non_zero_mask = ~alpha_zero_mask

        # Special case: alpha ≈ 0 (exact formula, vectorized)
        if np.any(alpha_zero_mask):
            result[alpha_zero_mask] = (
                np.exp(beta[alpha_zero_mask] ** 2 / 2)
                * np.sqrt(2 * np.pi)
                * norm.cdf(beta[alpha_zero_mask])
            )

        # All other cases: use parabolic cylinder function (vectorized)
        if np.any(non_zero_mask):
            alpha_nonzero = alpha[non_zero_mask]
            beta_nonzero = beta[non_zero_mask]

            # Compute nu for all cases
            nu_nonzero = -(alpha_nonzero + 1)

            # Compute D values - pbdv can handle arrays
            D_vals, _ = pbdv(nu_nonzero, -beta_nonzero)

            # Prefactor: exp(β²/4) * Γ(α+1) * D_{-α-1}(-β)
            # Note: The 2^power term cancels out (2^((α-1)/2) * 2^((1-α)/2) = 1)
            # So we use just exp(β²/4) * Γ(α+1)
            log_prefactor = gammaln(alpha_nonzero + 1) + beta_nonzero**2 / 4

            # Compute with pbdv (works for all practical cases)
            non_zero_result = np.exp(log_prefactor) * D_vals

            # Check for overflow/invalid results
            invalid_mask = ~np.isfinite(non_zero_result)
            if np.any(invalid_mask):
                # For overflow cases, set to a large but finite value
                # or raise warning - in practice these are extremely rare
                non_zero_result[invalid_mask] = np.inf

            # Assign back to result array
            result[non_zero_mask] = non_zero_result

        if result.ndim == 0:
            return float(result)

        return result

    @staticmethod
    def F_prime(
        x: float | np.ndarray,
        mu: float | np.ndarray,
        sigma: float | np.ndarray,
        theta: float | np.ndarray,
        r: float = 0.01,
        h: float = 1e-6,
        use_analytical: bool = False,
    ) -> float | np.ndarray:
        """
        Computes the F_prime function.
        F_prime(x, r) = dF(x, r) / dx

        Uses finite difference approximation: (F(x+h) - F(x)) / h

        Parameters
        ----------
        x : float or ndarray
            Single value or array of x values
        mu : float or ndarray
            Mean reversion speed parameter(s)
        sigma : float or ndarray
            Volatility parameter(s)
        theta : float or ndarray
            Long-term mean parameter(s)
        r : float, optional
            Discount rate (default: 0.01)
        h : float, optional
            Step size for finite difference (default: 1e-6)
        use_analytical : bool, optional
            If True, use analytical F function. If False, use numerical (default: False)

        Returns
        -------
        float or ndarray
            Scalar if all inputs are scalar, otherwise array with shape
            determined by broadcasting the input arrays
        """
        return (
            OrnsteinUhlenbeck.F(
                x + h,
                mu=mu,
                sigma=sigma,
                theta=theta,
                r=r,
                use_analytical=use_analytical,
            )
            - OrnsteinUhlenbeck.F(
                x, mu=mu, sigma=sigma, theta=theta, r=r, use_analytical=use_analytical
            )
        ) / h

    @staticmethod
    def G_prime(
        x: float | np.ndarray,
        mu: float | np.ndarray,
        sigma: float | np.ndarray,
        theta: float | np.ndarray,
        r: float = 0.01,
        h: float = 1e-6,
        use_analytical: bool = False,
    ) -> float | np.ndarray:
        """
        Computes the G_prime function.
        G_prime(x, r) = dG(x, r) / dx

        Uses finite difference approximation: (G(x+h) - G(x)) / h

        Parameters
        ----------
        x : float or ndarray
            Single value or array of x values
        mu : float or ndarray
            Mean reversion speed parameter(s)
        sigma : float or ndarray
            Volatility parameter(s)
        theta : float or ndarray
            Long-term mean parameter(s)
        r : float, optional
            Discount rate (default: 0.01)
        h : float, optional
            Step size for finite difference (default: 1e-6)
        use_analytical : bool, optional
            If True, use analytical G function. If False, use numerical (default: False)

        Returns
        -------
        float or ndarray
            Scalar if all inputs are scalar, otherwise array with shape
            determined by broadcasting the input arrays
        """
        return (
            OrnsteinUhlenbeck.G(
                x + h,
                mu=mu,
                sigma=sigma,
                theta=theta,
                r=r,
                use_analytical=use_analytical,
            )
            - OrnsteinUhlenbeck.G(
                x, mu=mu, sigma=sigma, theta=theta, r=r, use_analytical=use_analytical
            )
        ) / h

    @staticmethod
    def V(
        x: float | np.ndarray,
        mu: float | np.ndarray,
        sigma: float | np.ndarray,
        theta: float | np.ndarray,
        r: float | np.ndarray,
        c: float | np.ndarray,
        exit_level: float | np.ndarray,
        use_analytical: bool = False,
    ) -> float | np.ndarray:
        """
        The value function V(x, r).
        """
        # Convert to arrays and broadcast to same shape
        x_arr = np.atleast_1d(x)
        mu_arr = np.atleast_1d(mu)
        sigma_arr = np.atleast_1d(sigma)
        theta_arr = np.atleast_1d(theta)
        r_arr = np.atleast_1d(r)
        c_arr = np.atleast_1d(c)
        exit_level_arr = np.atleast_1d(exit_level)

        # Broadcast all arrays to the same shape
        x_arr, mu_arr, sigma_arr, theta_arr, r_arr, c_arr, exit_level_arr = (
            np.broadcast_arrays(
                x_arr, mu_arr, sigma_arr, theta_arr, r_arr, c_arr, exit_level_arr
            )
        )

        # Create masks for element-wise operations
        left_mask = x_arr < exit_level_arr

        # Compute V_left for all elements (will be used where left_mask is True)
        F_x = OrnsteinUhlenbeck.F(
            x_arr, mu_arr, sigma_arr, theta_arr, r_arr, use_analytical=use_analytical
        )
        F_exit = OrnsteinUhlenbeck.F(
            exit_level_arr,
            mu_arr,
            sigma_arr,
            theta_arr,
            r_arr,
            use_analytical=use_analytical,
        )
        V_left = (exit_level_arr - c_arr) * F_x / F_exit

        # Compute V_right for all elements (will be used where left_mask is False)
        V_right = x_arr - c_arr

        # Use np.where for element-wise selection
        result = np.where(left_mask, V_left, V_right)

        # Return scalar if input was scalar
        if (
            np.isscalar(x)
            and np.isscalar(mu)
            and np.isscalar(sigma)
            and np.isscalar(theta)
            and np.isscalar(r)
            and np.isscalar(c)
            and np.isscalar(exit_level)
        ):
            return float(result)

        return result

    @staticmethod
    def V_prime(
        x: float | np.ndarray,
        mu: float | np.ndarray,
        sigma: float | np.ndarray,
        theta: float | np.ndarray,
        r: float | np.ndarray,
        c: float | np.ndarray,
        exit_level: float | np.ndarray,
        h: float = 1e-6,
        use_analytical: bool = False,
    ) -> float | np.ndarray:
        """
        The derivative of the value function V(x, r).
        """
        return (
            OrnsteinUhlenbeck.V(
                x + h, mu, sigma, theta, r, c, exit_level, use_analytical=use_analytical
            )
            - OrnsteinUhlenbeck.V(
                x - h, mu, sigma, theta, r, c, exit_level, use_analytical=use_analytical
            )
        ) / (2 * h)

    @staticmethod
    def C(
        c: float | np.ndarray,
        mu: float | np.ndarray,
        sigma: float | np.ndarray,
        theta: float | np.ndarray,
        r: float,
        L: float | np.ndarray,
        exit_level: float | np.ndarray,
        use_analytical: bool = False,
    ) -> float | np.ndarray:
        """
        The constant C in the OU process.

        Parameters
        ----------
        c : float or ndarray
            Transaction cost parameter(s)
        mu : float or ndarray
            Mean reversion speed parameter(s)
        sigma : float or ndarray
            Volatility parameter(s)
        theta : float or ndarray
            Long-term mean parameter(s)
        r : float
            Discount rate
        L : float or ndarray
            Loss level parameter(s)
        exit_level : float or ndarray
            Optimal exit level parameter(s)
        use_analytical : bool, optional
            If True, use analytical F and G functions (default: False)

        Returns
        -------
        float or ndarray
            Scalar if all inputs are scalar, otherwise array with shape
            determined by broadcasting the input arrays
        """
        return (
            (exit_level - c)
            * OrnsteinUhlenbeck.G(
                L, mu=mu, sigma=sigma, theta=theta, r=r, use_analytical=use_analytical
            )
            - (L - c)
            * OrnsteinUhlenbeck.G(
                exit_level,
                mu=mu,
                sigma=sigma,
                theta=theta,
                r=r,
                use_analytical=use_analytical,
            )
        ) / (
            OrnsteinUhlenbeck.F(
                exit_level,
                mu=mu,
                sigma=sigma,
                theta=theta,
                r=r,
                use_analytical=use_analytical,
            )
            * OrnsteinUhlenbeck.G(
                L, mu=mu, sigma=sigma, theta=theta, r=r, use_analytical=use_analytical
            )
            - OrnsteinUhlenbeck.F(
                L, mu=mu, sigma=sigma, theta=theta, r=r, use_analytical=use_analytical
            )
            * OrnsteinUhlenbeck.G(
                exit_level,
                mu=mu,
                sigma=sigma,
                theta=theta,
                r=r,
                use_analytical=use_analytical,
            )
        )

    @staticmethod
    def D(
        c: float | np.ndarray,
        mu: float | np.ndarray,
        sigma: float | np.ndarray,
        theta: float | np.ndarray,
        r: float,
        L: float | np.ndarray,
        exit_level: float | np.ndarray,
        use_analytical: bool = False,
    ) -> float | np.ndarray:
        """
        The optimal entry level d_D^*.

        Parameters
        ----------
        c : float or ndarray
            Transaction cost parameter(s)
        mu : float or ndarray
            Mean reversion speed parameter(s)
        sigma : float or ndarray
            Volatility parameter(s)
        theta : float or ndarray
            Long-term mean parameter(s)
        r : float
            Discount rate
        L : float or ndarray
            Loss level parameter(s)
        exit_level : float or ndarray
            Optimal exit level parameter(s)
        use_analytical : bool, optional
            If True, use analytical F and G functions (default: False)

        Returns
        -------
        float or ndarray
            Scalar if all inputs are scalar, otherwise array with shape
            determined by broadcasting the input arrays
        """
        return (
            (L - c)
            * OrnsteinUhlenbeck.F(
                exit_level,
                mu=mu,
                sigma=sigma,
                theta=theta,
                r=r,
                use_analytical=use_analytical,
            )
            - (exit_level - c)
            * OrnsteinUhlenbeck.F(
                L, mu=mu, sigma=sigma, theta=theta, r=r, use_analytical=use_analytical
            )
        ) / (
            OrnsteinUhlenbeck.F(
                exit_level,
                mu=mu,
                sigma=sigma,
                theta=theta,
                r=r,
                use_analytical=use_analytical,
            )
            * OrnsteinUhlenbeck.G(
                L, mu=mu, sigma=sigma, theta=theta, r=r, use_analytical=use_analytical
            )
            - OrnsteinUhlenbeck.F(
                L, mu=mu, sigma=sigma, theta=theta, r=r, use_analytical=use_analytical
            )
            * OrnsteinUhlenbeck.G(
                exit_level,
                mu=mu,
                sigma=sigma,
                theta=theta,
                r=r,
                use_analytical=use_analytical,
            )
        )

    @staticmethod
    def V_L(
        x: float | np.ndarray,
        c: float | np.ndarray,
        mu: float | np.ndarray,
        sigma: float | np.ndarray,
        theta: float | np.ndarray,
        r: float,
        L: float | np.ndarray,
        exit_level: float | np.ndarray,
        use_analytical: bool = False,
    ) -> float | np.ndarray:
        """
        The value function V_L(x, r).

        Parameters
        ----------
        x : float or ndarray
            Single value or array of x values
        c : float or ndarray
            Transaction cost parameter(s)
        mu : float or ndarray
            Mean reversion speed parameter(s)
        sigma : float or ndarray
            Volatility parameter(s)
        theta : float or ndarray
            Long-term mean parameter(s)
        r : float
            Discount rate
        L : float or ndarray
            Loss level parameter(s)
        exit_level : float or ndarray
            Optimal exit level parameter(s)
        use_analytical : bool, optional
            If True, use analytical F and G functions (default: False)

        Returns
        -------
        float or ndarray
            Scalar if all inputs are scalar, otherwise array with shape
            determined by broadcasting the input arrays
        """
        # Convert to arrays for element-wise operations
        x_arr = np.atleast_1d(x)
        c_arr = np.atleast_1d(c)
        mu_arr = np.atleast_1d(mu)
        sigma_arr = np.atleast_1d(sigma)
        theta_arr = np.atleast_1d(theta)
        L_arr = np.atleast_1d(L)
        exit_level_arr = np.atleast_1d(exit_level)

        # Broadcast all arrays to the same shape
        x_arr, c_arr, mu_arr, sigma_arr, theta_arr, L_arr, exit_level_arr = (
            np.broadcast_arrays(
                x_arr, c_arr, mu_arr, sigma_arr, theta_arr, L_arr, exit_level_arr
            )
        )

        # Element-wise condition: x < exit_level and x > L
        condition = (x_arr < exit_level_arr) & (x_arr > L_arr)

        # Compute both branches
        waiting_value = OrnsteinUhlenbeck.C(
            c=c_arr,
            mu=mu_arr,
            sigma=sigma_arr,
            theta=theta_arr,
            r=r,
            L=L_arr,
            exit_level=exit_level_arr,
            use_analytical=use_analytical,
        ) * OrnsteinUhlenbeck.F(
            x_arr,
            mu=mu_arr,
            sigma=sigma_arr,
            theta=theta_arr,
            r=r,
            use_analytical=use_analytical,
        ) + OrnsteinUhlenbeck.D(
            c=c_arr,
            mu=mu_arr,
            sigma=sigma_arr,
            theta=theta_arr,
            r=r,
            L=L_arr,
            exit_level=exit_level_arr,
            use_analytical=use_analytical,
        ) * OrnsteinUhlenbeck.G(
            x_arr,
            mu=mu_arr,
            sigma=sigma_arr,
            theta=theta_arr,
            r=r,
            use_analytical=use_analytical,
        )
        immediate_value = x_arr - c_arr

        # Use np.where for element-wise selection
        result = np.where(condition, waiting_value, immediate_value)

        # Return scalar if input was scalar
        if (
            np.isscalar(x)
            and np.isscalar(c)
            and np.isscalar(mu)
            and np.isscalar(sigma)
            and np.isscalar(theta)
            and np.isscalar(L)
            and np.isscalar(exit_level)
        ):
            return float(result)

        return result

    @staticmethod
    def V_L_prime(
        x: float | np.ndarray,
        c: float | np.ndarray,
        mu: float | np.ndarray,
        sigma: float | np.ndarray,
        theta: float | np.ndarray,
        r: float,
        L: float | np.ndarray,
        exit_level: float | np.ndarray,
        h: float = 1e-6,
        use_analytical: bool = False,
    ) -> float | np.ndarray:
        """
        The derivative of the value function V_L(x, r).

        Uses finite difference approximation: (V_L(x+h) - V_L(x)) / h

        Parameters
        ----------
        x : float or ndarray
            Single value or array of x values
        c : float or ndarray
            Transaction cost parameter(s)
        mu : float or ndarray
            Mean reversion speed parameter(s)
        sigma : float or ndarray
            Volatility parameter(s)
        theta : float or ndarray
            Long-term mean parameter(s)
        r : float
            Discount rate
        L : float or ndarray
            Loss level parameter(s)
        exit_level : float or ndarray
            Optimal exit level parameter(s)
        h : float, optional
            Step size for finite difference (default: 1e-6)
        use_analytical : bool, optional
            If True, use analytical F and G functions (default: False)

        Returns
        -------
        float or ndarray
            Scalar if all inputs are scalar, otherwise array with shape
            determined by broadcasting the input arrays
        """
        return (
            OrnsteinUhlenbeck.V_L(
                x + h,
                c=c,
                mu=mu,
                sigma=sigma,
                theta=theta,
                r=r,
                L=L,
                exit_level=exit_level,
                use_analytical=use_analytical,
            )
            - OrnsteinUhlenbeck.V_L(
                x,
                c=c,
                mu=mu,
                sigma=sigma,
                theta=theta,
                r=r,
                L=L,
                exit_level=exit_level,
                use_analytical=use_analytical,
            )
        ) / h

    @staticmethod
    def get_optimal_exit_level(
        mu: float | np.ndarray,
        sigma: float | np.ndarray,
        theta: float | np.ndarray,
        r: float = 0.01,
        c: float | np.ndarray = 0.001,
        L: Optional[float | np.ndarray] = None,
        use_analytical: bool = False,
    ) -> float | np.ndarray:
        """
        Computes the optimal exit level (exit_level) for an OU process.
        Uses Newton's method for root finding.

        Parameters
        ----------
        mu : float or ndarray
            Mean reversion speed parameter(s)
        sigma : float or ndarray
            Volatility parameter(s)
        theta : float or ndarray
            Long-term mean parameter(s)
        r : float, optional
            Discount rate (default: 0.01)
        c : float or ndarray, optional
            Transaction cost to enter the trade (default: 0.001)
        L : float or ndarray, optional
            Loss level to exit the trade. If None, defaults to theta - 2*sigma (default: None)
        use_analytical : bool, optional
            If True, use analytical F and G functions (default: False)

        Returns
        -------
        float or ndarray
            Optimal exit level (exit_level) - should be well above theta.
            Scalar if all inputs are scalar, otherwise array with shape
            determined by broadcasting the input arrays
        """
        # Convert to arrays and broadcast
        mu_arr = np.atleast_1d(mu)
        sigma_arr = np.atleast_1d(sigma)
        theta_arr = np.atleast_1d(theta)
        c_arr = np.atleast_1d(c) if not np.isscalar(c) else np.array([c])

        # Broadcast all arrays to the same shape
        mu_arr, sigma_arr, theta_arr, c_arr = np.broadcast_arrays(
            mu_arr, sigma_arr, theta_arr, c_arr
        )

        # Handle L
        if L is None:
            L_arr = theta_arr - 2 * sigma_arr
        else:
            L_arr = np.atleast_1d(L) if not np.isscalar(L) else np.array([L])
            L_arr = np.broadcast_to(L_arr, mu_arr.shape)

        # Choose initial guess
        initial_guess_arr = theta_arr + 10 * sigma_arr

        # Pre-compute L-dependent values (constant for all elements)
        F_L = OrnsteinUhlenbeck.F(
            L_arr, mu_arr, sigma_arr, theta_arr, r, use_analytical=use_analytical
        )
        G_L = OrnsteinUhlenbeck.G(
            L_arr, mu_arr, sigma_arr, theta_arr, r, use_analytical=use_analytical
        )
        # Ensure F_L and G_L are arrays with correct shape
        F_L = np.asarray(F_L)
        G_L = np.asarray(G_L)
        if F_L.shape != mu_arr.shape:
            F_L = np.broadcast_to(F_L, mu_arr.shape)
        if G_L.shape != mu_arr.shape:
            G_L = np.broadcast_to(G_L, mu_arr.shape)

        # Define vectorized function to find root of: f(b) = f_left - f_right
        def f_exit(b: float | np.ndarray, h: float = 1e-6) -> float | np.ndarray:
            # Ensure b is properly shaped and broadcast with other arrays
            b_arr = np.asarray(b)
            if b_arr.shape != mu_arr.shape:
                # Broadcast b to match mu_arr shape
                b_arr = np.broadcast_to(b_arr, mu_arr.shape)

            G_b = OrnsteinUhlenbeck.G(
                b_arr, mu_arr, sigma_arr, theta_arr, r, use_analytical=use_analytical
            )
            F_prime_b = OrnsteinUhlenbeck.F_prime(
                b_arr,
                mu_arr,
                sigma_arr,
                theta_arr,
                r,
                h=h,
                use_analytical=use_analytical,
            )
            F_b = OrnsteinUhlenbeck.F(
                b_arr, mu_arr, sigma_arr, theta_arr, r, use_analytical=use_analytical
            )
            G_prime_b = OrnsteinUhlenbeck.G_prime(
                b_arr,
                mu_arr,
                sigma_arr,
                theta_arr,
                r,
                h=h,
                use_analytical=use_analytical,
            )

            f_left = ((L_arr - c_arr) * G_b - (b_arr - c_arr) * F_prime_b) + (
                (b_arr - c_arr) * F_L - (L_arr - c_arr) * F_b
            ) * G_prime_b

            f_right = G_b * F_L - G_L * F_b

            return f_left - f_right

        # Define vectorized derivative using numerical differentiation
        def df_exit_dx(b: float | np.ndarray, h: float = 1e-6) -> float | np.ndarray:
            b_arr = np.asarray(b)
            if b_arr.shape != mu_arr.shape:
                b_arr = np.broadcast_to(b_arr, mu_arr.shape)
            return (f_exit(b_arr + h, h) - f_exit(b_arr - h, h)) / (2 * h)

        # Use Newton's method to find result
        # Exit level must be > theta
        result = newton_vectorized(
            f_exit,
            df_exit_dx,
            initial_guess_arr,
            max_iter=50,
            tol=1e-8,
            step_direction="negative",
        )

        # Ensure result is properly shaped
        result = np.asarray(result)
        if result.shape != mu_arr.shape:
            result = np.broadcast_to(result, mu_arr.shape)

        # Return scalar if input was scalar
        if (
            np.isscalar(mu)
            and np.isscalar(sigma)
            and np.isscalar(theta)
            and np.isscalar(c)
            and (L is None or np.isscalar(L))
        ):
            # Extract single element before converting to float
            if isinstance(result, np.ndarray):
                return float(result.item())
            return float(result)

        return result

    @staticmethod
    def get_optimal_entry_level(
        mu: float | np.ndarray,
        sigma: float | np.ndarray,
        theta: float | np.ndarray,
        r: float = 0.01,
        c: float | np.ndarray = 0.001,
        L: Optional[float | np.ndarray] = None,
        h: float = 1e-6,
        initial_guess: Optional[float | np.ndarray] = None,
        exit_level: Optional[float | np.ndarray] = None,
        use_analytical: bool = False,
    ) -> float | np.ndarray:
        """
        Computes the optimal entry level (d_star) for an OU process.
        Uses Newton's method for root finding.

        Parameters
        ----------
        mu : float or ndarray
            Mean reversion speed parameter(s)
        sigma : float or ndarray
            Volatility parameter(s)
        theta : float or ndarray
            Long-term mean parameter(s)
        r : float, optional
            Discount rate (default: 0.01)
        c : float or ndarray, optional
            Transaction cost to enter the trade (default: 0.001)
        L : float or ndarray, optional
            Loss level to exit the trade. If None, defaults to theta - 2*sigma (default: None)
        h : float, optional
            Step size for numerical differentiation (default: 1e-6)
        initial_guess : ndarray, optional
            Starting point(s) for root finding. Should be a vector of the same length/shape as
            the broadcasted mu, sigma, and theta arrays. Each element will be used as the initial
            guess for the corresponding parameter set. If None, defaults to multiple guesses
            based on L and theta (default: None)
        exit_level : float or ndarray, optional
            Pre-computed optimal exit level. If None, will be computed (default: None)
        use_analytical : bool, optional
            If True, use analytical F and G functions (default: False)

        Returns
        -------
        float or ndarray
            Optimal entry level (d_star) - should be below theta.
            Scalar if all inputs are scalar, otherwise array with shape
            determined by broadcasting the input arrays
        """
        # Convert to arrays and broadcast
        mu_arr = np.atleast_1d(mu)
        sigma_arr = np.atleast_1d(sigma)
        theta_arr = np.atleast_1d(theta)
        c_arr = np.atleast_1d(c) if not np.isscalar(c) else np.array([c])

        # Broadcast all arrays to the same shape
        mu_arr, sigma_arr, theta_arr, c_arr = np.broadcast_arrays(
            mu_arr, sigma_arr, theta_arr, c_arr
        )

        # Handle L
        if L is None:
            L_arr = theta_arr - 2 * sigma_arr
        else:
            L_arr = np.atleast_1d(L) if not np.isscalar(L) else np.array([L])
            L_arr = np.broadcast_to(L_arr, mu_arr.shape)

        # Get exit_level first (needed for d_star calculation) if not provided
        if exit_level is None:
            exit_level_arr = OrnsteinUhlenbeck.get_optimal_exit_level(
                mu=mu_arr,
                sigma=sigma_arr,
                theta=theta_arr,
                r=r,
                c=c_arr,
                L=L_arr,
                h=h,
                use_analytical=use_analytical,
            )
            exit_level_arr = np.atleast_1d(exit_level_arr)
        else:
            exit_level_arr = (
                np.atleast_1d(exit_level)
                if not np.isscalar(exit_level)
                else np.array([exit_level])
            )
            exit_level_arr = np.broadcast_to(exit_level_arr, mu_arr.shape)

        # Handle initial_guess
        # Entry level should be between L and theta
        if initial_guess is None:
            # Default guess: middle of [L, theta]
            range_size = theta_arr - L_arr
            initial_guess_arr = L_arr + 0.5 * range_size
        else:
            # initial_guess should be a vector of the same shape as broadcasted arrays
            initial_guess_arr = np.asarray(initial_guess)
            if initial_guess_arr.ndim == 0:
                # Scalar provided - broadcast to match shape
                initial_guess_arr = np.broadcast_to(initial_guess_arr, mu_arr.shape)
            else:
                # Vector provided - ensure it can be broadcast to match shape
                try:
                    initial_guess_arr = np.broadcast_to(
                        initial_guess_arr, mu_arr.shape
                    ).copy()
                except ValueError as e:
                    raise ValueError(
                        f"initial_guess must be broadcastable to shape {mu_arr.shape} "
                        f"(matching mu, sigma, theta after broadcasting), got shape {initial_guess_arr.shape}"
                    ) from e
            # Ensure initial guess is between L and theta
            initial_guess_arr = np.maximum(initial_guess_arr, L_arr + 0.001)
            initial_guess_arr = np.minimum(initial_guess_arr, theta_arr - 0.001)

        # Define vectorized function to find root of: f(d) = f_left - f_right
        def f_entry(d):
            # Ensure d is properly shaped and broadcast with other arrays
            d_arr = np.asarray(d)
            if d_arr.shape != mu_arr.shape:
                # Broadcast d to match mu_arr shape
                d_arr = np.broadcast_to(d_arr, mu_arr.shape)

            G_d = OrnsteinUhlenbeck.G(
                d_arr, mu_arr, sigma_arr, theta_arr, r, use_analytical=use_analytical
            )
            V_prime_d = OrnsteinUhlenbeck.V_L_prime(
                d_arr,
                c_arr,
                mu_arr,
                sigma_arr,
                theta_arr,
                r,
                L_arr,
                exit_level_arr,
                h,
                use_analytical=use_analytical,
            )
            G_prime_d = OrnsteinUhlenbeck.G_prime(
                d_arr,
                mu_arr,
                sigma_arr,
                theta_arr,
                r,
                h=h,
                use_analytical=use_analytical,
            )
            V_d = OrnsteinUhlenbeck.V_L(
                d_arr,
                c_arr,
                mu_arr,
                sigma_arr,
                theta_arr,
                r,
                L_arr,
                exit_level_arr,
                use_analytical=use_analytical,
            )

            f_left = G_d * (V_prime_d - 1)
            f_right = G_prime_d * (V_d - d_arr - c_arr)

            return f_left - f_right

        # Define vectorized derivative using numerical differentiation
        def df_entry_dx(d):
            d_arr = np.asarray(d)
            if d_arr.shape != mu_arr.shape:
                d_arr = np.broadcast_to(d_arr, mu_arr.shape)
            h_num = 1e-6
            return (f_entry(d_arr + h_num) - f_entry(d_arr - h_num)) / (2 * h_num)

        # Use Newton's method to find result
        # Entry level must be between L and theta
        result = newton_vectorized(
            f_entry,
            df_entry_dx,
            initial_guess_arr,
            max_iter=50,
            tol=1e-8,
            step_direction="positive",
        )
        # Validate result is in valid range
        result = np.asarray(result)
        if result.shape != mu_arr.shape:
            result = np.broadcast_to(result, mu_arr.shape)
        # If result is not in [L, theta], it's invalid - use analytical as fallback
        invalid_mask = (result <= L_arr) | (result >= theta_arr)
        if np.any(invalid_mask):
            analytical_result = OrnsteinUhlenbeck.get_optimal_entry_level(
                mu=mu_arr[invalid_mask] if mu_arr.ndim > 0 else mu,
                sigma=sigma_arr[invalid_mask] if sigma_arr.ndim > 0 else sigma,
                theta=theta_arr[invalid_mask] if theta_arr.ndim > 0 else theta,
                r=r,
                c=c_arr[invalid_mask] if c_arr.ndim > 0 else c,
                L=L_arr[invalid_mask] if L_arr.ndim > 0 else L,
                exit_level=exit_level_arr[invalid_mask]
                if exit_level_arr.ndim > 0
                else exit_level_arr,
                use_analytical=True,
            )
            if isinstance(analytical_result, np.ndarray):
                result[invalid_mask] = analytical_result
            else:
                result[invalid_mask] = analytical_result

        # Ensure result is properly shaped
        result = np.asarray(result)
        if result.shape != mu_arr.shape:
            result = np.broadcast_to(result, mu_arr.shape)

        # Return scalar if input was scalar
        if (
            np.isscalar(mu)
            and np.isscalar(sigma)
            and np.isscalar(theta)
            and np.isscalar(c)
            and (L is None or np.isscalar(L))
            and (exit_level is None or np.isscalar(exit_level))
        ):
            # Extract single element before converting to float
            if isinstance(result, np.ndarray):
                return float(result.item())
            return float(result)

        return result
