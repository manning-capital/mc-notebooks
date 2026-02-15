import warnings

import numpy as np
import statsmodels.api as sm
from scipy.integrate import quad
from scipy.special import pbdv, gammaln
from scipy.stats import norm
from typing import Literal

# Default step size for finite difference approximations in prime calculations
H = 1e-6

# Threshold below which the OU process is considered degenerate.
# When mu < threshold, the process degenerates to Brownian motion (no mean reversion),
# and the F/G integrals diverge due to r/mu terms.
# When sigma < threshold, the process becomes deterministic (no stochastic component).
DEGENERATE_OU_THRESHOLD = 1e-6


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
    def f_integrand(
        u: float,
        x: float,
        mu: float,
        sigma: float,
        theta: float,
        derivative: int = 0,
        r: float = 0.01,
    ) -> float:
        """
        Computes the f function for various derivatives with respect to x based on the base function f(u, r).
        f(u, r) = u^((r/mu) - 1) * e^(sqrt((2*mu)/(sigma^2)) * (x - theta) * u  - u^2 / 2)
        """
        alpha = (r / mu) - 1 + derivative
        beta = np.sqrt((2 * mu) / (sigma**2)) * (x - theta)
        return (beta**derivative) * (u**alpha) * np.exp(beta * u - u**2 / 2)

    @staticmethod
    def g_integrand(
        u: float,
        x: float,
        mu: float,
        sigma: float,
        theta: float,
        derivative: int = 0,
        r: float = 0.01,
    ) -> float:
        """
        Computes the f function for various derivatives with respect to x based on the base function f(u, r).
        f(u, r) = u^((r/mu) - 1) * e^(sqrt((2*mu)/(sigma^2)) * (x - theta) * u  - u^2 / 2)
        """
        alpha = (r / mu) - 1 + derivative
        beta = np.sqrt((2 * mu) / (sigma**2)) * (theta - x)
        return (beta**derivative) * (u**alpha) * np.exp(beta * u - u**2 / 2)

    @staticmethod
    def f_prime(
        u: float, x: float, mu: float, sigma: float, theta: float, r: float = 0.01
    ) -> float:
        """
        Computes the derivative of f with respect to x.
        f'(u, r) = sqrt((2*mu)/(sigma^2)) * u^((r/mu) - 1) * e^(sqrt((2*mu)/(sigma^2)) * (x - theta) * u  - u^2 / 2)
        """
        return (
            np.sqrt((2 * mu) / (sigma**2))
            * u ** (r / mu)
            * np.exp(np.sqrt((2 * mu) / (sigma**2)) * (x - theta) * u - u**2 / 2)
        )

    @staticmethod
    def F(
        x: float | np.ndarray,
        mu: float | np.ndarray,
        sigma: float | np.ndarray,
        theta: float | np.ndarray,
        r: float = 0.01,
        derivative: int = 0,
        use_analytical: bool = True,
    ) -> float | np.ndarray:
        """
        Computes the F function or its nth derivative with respect to x.
            derivative=0: F(x, r)   = integral f(u, x, r) du from 0 to infinity
            derivative=1: F'(x, r)  = dF/dx
            derivative=2: F''(x, r) = d²F/dx²

        By default uses numerical integration with quad for accuracy.
        An experimental analytical approximation using parabolic cylinder functions
        is available with use_analytical=True, but has ~1-2% error that varies with parameters.

        Parameters
        ----------
        x : float or ndarray
            Absolute position values.
        mu : float or ndarray
            Mean reversion speed parameter(s)
        sigma : float or ndarray
            Volatility parameter(s)
        theta : float or ndarray
            Mean reversion level(s)
        r : float, optional
            Discount rate (default: 0.01)
        derivative : int, optional
            Order of derivative with respect to x (default: 0)
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
        mu_arr = np.atleast_1d(mu)
        sigma_arr = np.atleast_1d(sigma)
        x_arr = np.atleast_1d(x)
        theta_arr = np.atleast_1d(theta)
        mu_arr, sigma_arr, x_arr, theta_arr = np.broadcast_arrays(
            mu_arr, sigma_arr, x_arr, theta_arr
        )

        alpha = (r / mu_arr) - 1
        scale = np.sqrt(2 * mu_arr / sigma_arr**2)
        beta = scale * (x_arr - theta_arr)

        return OrnsteinUhlenbeck._compute_integral(
            alpha, scale, beta, derivative=derivative, use_analytical=use_analytical
        )

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
    def g_prime(
        u: float, x: float, mu: float, sigma: float, theta: float, r: float = 0.01
    ) -> float:
        """
        Computes the derivative of g with respect to x.
        g'(u, x, r) = -sqrt((2*mu)/(sigma^2)) * u^((r/mu) - 2) * e^(sqrt((2*mu)/(sigma^2)) * (theta - x) * u  - u^2 / 2)
        """
        return (
            -np.sqrt((2 * mu) / (sigma**2))
            * u ** ((r / mu) - 2)
            * np.exp(np.sqrt((2 * mu) / (sigma**2)) * (theta - x) * u - u**2 / 2)
        )

    @staticmethod
    def _compute_integral(
        alpha: float | np.ndarray,
        scale: float | np.ndarray,
        beta: float | np.ndarray,
        derivative: int = 0,
        use_analytical: bool = True,
    ) -> float | np.ndarray:
        """
        General-purpose integral of the form:
            scale^derivative * integral_0^inf u^(alpha + derivative) * exp(beta*u - u^2/2) du

        This is the shared computation underlying F, G, and their derivatives.
        The derivative parameter controls both the alpha offset and scaling factor,
        matching the pattern from f_integrand/g_integrand:
            derivative=0: integral u^alpha * exp(beta*u - u^2/2) du           (F, G)
            derivative=1: scale * integral u^(alpha+1) * exp(beta*u - u^2/2) du (F', G')
            derivative=2: scale^2 * integral u^(alpha+2) * exp(beta*u - u^2/2) du (F'', G'')

        Callers provide the base alpha = (r/mu) - 1 and the appropriate scale and beta:
            F family:  scale = sqrt(2mu/sigma^2), beta = scale * (x - theta)
            G family:  scale = sqrt(2mu/sigma^2), beta = scale * (theta - x)

        Parameters
        ----------
        alpha : float or ndarray
            Base power of u in the integrand (before derivative offset)
        scale : float or ndarray
            Scale factor for derivatives (sqrt(2*mu/sigma^2))
        beta : float or ndarray
            Coefficient of u in the exponential
        derivative : int, optional
            Order of derivative (default: 0). Adds to alpha and scales by scale^derivative.
        use_analytical : bool, optional
            If True, use parabolic cylinder function approximation.
            If False, use numerical integration with quad (default).

        Returns
        -------
        float or ndarray
        """
        # Apply derivative: shift alpha and compute scale factor
        effective_alpha = alpha + derivative if derivative != 0 else alpha
        scale_factor = scale**derivative if derivative != 0 else 1.0

        if not use_analytical:
            all_scalar = (
                np.isscalar(effective_alpha)
                and np.isscalar(beta)
                and np.isscalar(scale_factor)
            )

            if all_scalar:
                val = quad(
                    lambda u: u**effective_alpha * np.exp(beta * u - u**2 / 2),
                    0,
                    np.inf,
                )[0]
                return scale_factor * val

            alpha_arr = np.atleast_1d(effective_alpha)
            beta_arr = np.atleast_1d(beta)
            scale_factor_arr = np.atleast_1d(scale_factor)
            alpha_arr, beta_arr, scale_factor_arr = np.broadcast_arrays(
                alpha_arr, beta_arr, scale_factor_arr
            )

            result = np.empty(alpha_arr.shape)
            for idx in np.ndindex(alpha_arr.shape):
                result[idx] = (
                    scale_factor_arr[idx]
                    * quad(
                        lambda u, a=alpha_arr[idx], b=beta_arr[idx]: u**a
                        * np.exp(b * u - u**2 / 2),
                        0,
                        np.inf,
                    )[0]
                )

            if result.ndim == 0:
                return float(result)
            return result

        # Analytical path using parabolic cylinder functions
        # Based on: integral_0^inf u^alpha exp(beta*u - u^2/2) du
        #         = exp(beta^2/4) * Gamma(alpha+1) * D_{-alpha-1}(-beta)
        alpha_arr = np.atleast_1d(effective_alpha)
        beta_arr = np.atleast_1d(beta)
        scale_factor_arr = np.atleast_1d(scale_factor)
        alpha_arr, beta_arr, scale_factor_arr = np.broadcast_arrays(
            alpha_arr, beta_arr, scale_factor_arr
        )

        result = np.empty(alpha_arr.shape, dtype=float)

        alpha_zero_mask = np.isclose(alpha_arr, 0, atol=1e-10)
        non_zero_mask = ~alpha_zero_mask

        # Special case: alpha ~ 0 (exact formula)
        if np.any(alpha_zero_mask):
            result[alpha_zero_mask] = (
                scale_factor_arr[alpha_zero_mask]
                * np.exp(beta_arr[alpha_zero_mask] ** 2 / 2)
                * np.sqrt(2 * np.pi)
                * norm.cdf(beta_arr[alpha_zero_mask])
            )

        if np.any(non_zero_mask):
            a_nz = alpha_arr[non_zero_mask]
            b_nz = beta_arr[non_zero_mask]
            s_nz = scale_factor_arr[non_zero_mask]
            nu_nz = -(a_nz + 1)

            D_vals, _ = pbdv(nu_nz, -b_nz)
            log_prefactor = gammaln(a_nz + 1) + b_nz**2 / 4

            non_zero_result = np.zeros_like(D_vals)
            D_is_nan = np.isnan(D_vals)
            D_is_posinf = np.isposinf(D_vals)
            D_is_neginf = np.isneginf(D_vals)
            D_is_finite = ~(D_is_nan | D_is_posinf | D_is_neginf)

            if np.any(D_is_nan):
                prefactor_is_zero = log_prefactor[D_is_nan] == -np.inf
                non_zero_result[D_is_nan] = np.where(prefactor_is_zero, 0.0, np.nan)

            if np.any(D_is_posinf):
                prefactor_is_zero = log_prefactor[D_is_posinf] == -np.inf
                non_zero_result[D_is_posinf] = np.where(
                    prefactor_is_zero, np.nan, np.inf
                )

            if np.any(D_is_neginf):
                prefactor_is_zero = log_prefactor[D_is_neginf] == -np.inf
                non_zero_result[D_is_neginf] = np.where(
                    prefactor_is_zero, np.nan, -np.inf
                )

            if np.any(D_is_finite):
                prefactor = np.exp(log_prefactor[D_is_finite])
                non_zero_result[D_is_finite] = prefactor * D_vals[D_is_finite]
                finite_overflow = ~np.isfinite(non_zero_result[D_is_finite])
                if np.any(finite_overflow):
                    indices = np.where(D_is_finite)[0][finite_overflow]
                    signs = np.sign(prefactor[finite_overflow]) * np.sign(
                        D_vals[indices]
                    )
                    non_zero_result[indices] = signs * np.inf

            result[non_zero_mask] = s_nz * non_zero_result

        if result.ndim == 0:
            return float(result)
        return result

    @staticmethod
    def G(
        x: float | np.ndarray,
        mu: float | np.ndarray,
        sigma: float | np.ndarray,
        theta: float | np.ndarray,
        r: float = 0.01,
        derivative: int = 0,
        use_analytical: bool = True,
    ) -> float | np.ndarray:
        """
        Computes the G function or its nth derivative with respect to x.
            derivative=0: G(x, r)   = integral g(u, x, r) du from 0 to infinity
            derivative=1: G'(x, r)  = dG/dx
            derivative=2: G''(x, r) = d²G/dx²

        By default uses numerical integration with quad for accuracy.
        An experimental analytical approximation using parabolic cylinder functions
        is available with use_analytical=True, but has ~1-2% error that varies with parameters.

        Parameters
        ----------
        x : float or ndarray
            Absolute position values.
        mu : float or ndarray
            Mean reversion speed parameter(s)
        sigma : float or ndarray
            Volatility parameter(s)
        theta : float or ndarray
            Mean reversion level(s)
        r : float, optional
            Discount rate (default: 0.01)
        derivative : int, optional
            Order of derivative with respect to x (default: 0)
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
        mu_arr = np.atleast_1d(mu)
        sigma_arr = np.atleast_1d(sigma)
        x_arr = np.atleast_1d(x)
        theta_arr = np.atleast_1d(theta)
        mu_arr, sigma_arr, x_arr, theta_arr = np.broadcast_arrays(
            mu_arr, sigma_arr, x_arr, theta_arr
        )

        alpha = (r / mu_arr) - 1
        scale = -1 * np.sqrt(2 * mu_arr / sigma_arr**2)
        beta = scale * (x_arr - theta_arr)

        return OrnsteinUhlenbeck._compute_integral(
            alpha, scale, beta, derivative=derivative, use_analytical=use_analytical
        )

    @staticmethod
    def V(
        x: float | np.ndarray,
        mu: float | np.ndarray,
        sigma: float | np.ndarray,
        theta: float | np.ndarray,
        r: float | np.ndarray,
        c: float | np.ndarray,
        exit_level: float | np.ndarray,
        use_analytical: bool = True,
    ) -> float | np.ndarray:
        """
        The value function V(x, r).

        Parameters
        ----------
        x : float or ndarray
            Absolute position values.
        mu : float or ndarray
            Mean reversion speed parameter(s)
        sigma : float or ndarray
            Volatility parameter(s)
        theta : float or ndarray
            Mean reversion level(s)
        r : float or ndarray
            Discount rate(s)
        c : float or ndarray
            Transaction cost(s)
        exit_level : float or ndarray
            Exit level(s) as absolute positions
        use_analytical : bool, optional
            If True, use analytical F function (default: False)
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
        V_left = ((exit_level_arr - c_arr) / F_exit) * F_x

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
        use_analytical: bool = True,
    ) -> float | np.ndarray:
        """
        The derivative of the value function V(x, r) with respect to x.

        For x < exit_level: V'(x) = [(exit_level - c) / F(exit_level)] * F'(x)
        For x >= exit_level: V'(x) = 1

        Parameters
        ----------
        x : float or ndarray
            Absolute position values.
        mu : float or ndarray
            Mean reversion speed parameter(s)
        sigma : float or ndarray
            Volatility parameter(s)
        theta : float or ndarray
            Mean reversion level(s)
        r : float or ndarray
            Discount rate(s)
        c : float or ndarray
            Transaction cost(s)
        exit_level : float or ndarray
            Exit level(s) as absolute positions
        use_analytical : bool, optional
            If True, use analytical F function (default: False)

        Returns
        -------
        float or ndarray
            Derivative dV/dx
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

        # Compute V'_left for all elements (will be used where left_mask is True)
        # V'(x) = [(exit_level - c) / F(exit_level)] * F'(x)
        F_prime_x = OrnsteinUhlenbeck.F(
            x_arr,
            mu_arr,
            sigma_arr,
            theta_arr,
            r_arr,
            derivative=1,
            use_analytical=use_analytical,
        )
        F_exit = OrnsteinUhlenbeck.F(
            exit_level_arr,
            mu_arr,
            sigma_arr,
            theta_arr,
            r_arr,
            use_analytical=use_analytical,
        )
        V_prime_left = ((exit_level_arr - c_arr) / F_exit) * F_prime_x

        # For x >= exit_level: V'(x) = d/dx(x - c) = 1
        V_prime_right = np.ones_like(x_arr)

        # Use np.where for element-wise selection
        result = np.where(left_mask, V_prime_left, V_prime_right)

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
    def V_double_prime(
        x: float | np.ndarray,
        mu: float | np.ndarray,
        sigma: float | np.ndarray,
        theta: float | np.ndarray,
        r: float | np.ndarray,
        c: float | np.ndarray,
        exit_level: float | np.ndarray,
        use_analytical: bool = True,
    ) -> float | np.ndarray:
        """
        The second derivative of the value function V(x, r) with respect to x.

        For x < exit_level: V''(x) = [(exit_level - c) / F(exit_level)] * F''(x)
        For x >= exit_level: V''(x) = 0

        Parameters
        ----------
        x : float or ndarray
            Absolute position values.
        mu : float or ndarray
            Mean reversion speed parameter(s)
        sigma : float or ndarray
            Volatility parameter(s)
        theta : float or ndarray
            Mean reversion level(s)
        r : float or ndarray
            Discount rate(s)
        c : float or ndarray
            Transaction cost(s)
        exit_level : float or ndarray
            Exit level(s) as absolute positions
        use_analytical : bool, optional
            If True, use analytical F function (default: False)

        Returns
        -------
        float or ndarray
            Second derivative d²V/dx²
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

        # Compute V''_left for all elements (will be used where left_mask is True)
        # V''(x) = [(exit_level - c) / F(exit_level)] * F''(x)
        F_double_prime_x = OrnsteinUhlenbeck.F(
            x_arr,
            mu_arr,
            sigma_arr,
            theta_arr,
            r_arr,
            derivative=2,
            use_analytical=use_analytical,
        )
        F_exit = OrnsteinUhlenbeck.F(
            exit_level_arr,
            mu_arr,
            sigma_arr,
            theta_arr,
            r_arr,
            use_analytical=use_analytical,
        )
        V_double_prime_left = (exit_level_arr - c_arr) * F_double_prime_x / F_exit

        # For x >= exit_level: V''(x) = d²/dx²(x - c) = 0
        V_double_prime_right = np.zeros_like(x_arr)

        # Use np.where for element-wise selection
        result = np.where(left_mask, V_double_prime_left, V_double_prime_right)

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
    def C(
        c: float | np.ndarray,
        mu: float | np.ndarray,
        sigma: float | np.ndarray,
        r: float,
        L: float | np.ndarray,
        exit_level: float | np.ndarray,
        use_analytical: bool = True,
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
        r : float
            Discount rate
        L : float or ndarray
            Loss level as spread (L - theta)
        exit_level : float or ndarray
            Exit level as spread (exit_level - theta)
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
                L, mu=mu, sigma=sigma, theta=0, r=r, use_analytical=use_analytical
            )
            - (L - c)
            * OrnsteinUhlenbeck.G(
                exit_level,
                mu=mu,
                sigma=sigma,
                theta=0,
                r=r,
                use_analytical=use_analytical,
            )
        ) / (
            OrnsteinUhlenbeck.F(
                exit_level,
                mu=mu,
                sigma=sigma,
                theta=0,
                r=r,
                use_analytical=use_analytical,
            )
            * OrnsteinUhlenbeck.G(
                L, mu=mu, sigma=sigma, theta=0, r=r, use_analytical=use_analytical
            )
            - OrnsteinUhlenbeck.F(
                L, mu=mu, sigma=sigma, theta=0, r=r, use_analytical=use_analytical
            )
            * OrnsteinUhlenbeck.G(
                exit_level,
                mu=mu,
                sigma=sigma,
                theta=0,
                r=r,
                use_analytical=use_analytical,
            )
        )

    @staticmethod
    def D(
        c: float | np.ndarray,
        mu: float | np.ndarray,
        sigma: float | np.ndarray,
        r: float,
        L: float | np.ndarray,
        exit_level: float | np.ndarray,
        use_analytical: bool = True,
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
        r : float
            Discount rate
        L : float or ndarray
            Loss level as spread (L - theta)
        exit_level : float or ndarray
            Exit level as spread (exit_level - theta)
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
                theta=0,
                r=r,
                use_analytical=use_analytical,
            )
            - (exit_level - c)
            * OrnsteinUhlenbeck.F(
                L, mu=mu, sigma=sigma, theta=0, r=r, use_analytical=use_analytical
            )
        ) / (
            OrnsteinUhlenbeck.F(
                exit_level,
                mu=mu,
                sigma=sigma,
                theta=0,
                r=r,
                use_analytical=use_analytical,
            )
            * OrnsteinUhlenbeck.G(
                L, mu=mu, sigma=sigma, theta=0, r=r, use_analytical=use_analytical
            )
            - OrnsteinUhlenbeck.F(
                L, mu=mu, sigma=sigma, theta=0, r=r, use_analytical=use_analytical
            )
            * OrnsteinUhlenbeck.G(
                exit_level,
                mu=mu,
                sigma=sigma,
                theta=0,
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
        r: float,
        L: float | np.ndarray,
        exit_level: float | np.ndarray,
        use_analytical: bool = True,
    ) -> float | np.ndarray:
        """
        The value function V_L(x, r).

        Parameters
        ----------
        x : float or ndarray
            Spread values (x - theta). MUST be pre-normalized relative to theta.
        c : float or ndarray
            Transaction cost parameter(s)
        mu : float or ndarray
            Mean reversion speed parameter(s)
        sigma : float or ndarray
            Volatility parameter(s)
        r : float
            Discount rate
        L : float or ndarray
            Loss level as spread (L - theta)
        exit_level : float or ndarray
            Exit level as spread (exit_level - theta)
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
        L_arr = np.atleast_1d(L)
        exit_level_arr = np.atleast_1d(exit_level)

        # Broadcast all arrays to the same shape
        x_arr, c_arr, mu_arr, sigma_arr, L_arr, exit_level_arr = np.broadcast_arrays(
            x_arr, c_arr, mu_arr, sigma_arr, L_arr, exit_level_arr
        )

        # Element-wise condition: x < exit_level and x > L
        condition = (x_arr < exit_level_arr) & (x_arr > L_arr)

        # Compute both branches
        waiting_value = OrnsteinUhlenbeck.C(
            c=c_arr,
            mu=mu_arr,
            sigma=sigma_arr,
            r=r,
            L=L_arr,
            exit_level=exit_level_arr,
            use_analytical=use_analytical,
        ) * OrnsteinUhlenbeck.F(
            x_arr,
            mu=mu_arr,
            sigma=sigma_arr,
            theta=0,
            r=r,
            use_analytical=use_analytical,
        ) + OrnsteinUhlenbeck.D(
            c=c_arr,
            mu=mu_arr,
            sigma=sigma_arr,
            r=r,
            L=L_arr,
            exit_level=exit_level_arr,
            use_analytical=use_analytical,
        ) * OrnsteinUhlenbeck.G(
            x_arr,
            mu=mu_arr,
            sigma=sigma_arr,
            theta=0,
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
        r: float,
        L: float | np.ndarray,
        exit_level: float | np.ndarray,
        h: float | None = None,
        use_analytical: bool = True,
    ) -> float | np.ndarray:
        """
        The derivative of the value function V_L(x, r).

        Uses finite difference approximation: (V_L(x+h) - V_L(x)) / h

        Parameters
        ----------
        x : float or ndarray
            Spread values (x - theta). MUST be pre-normalized relative to theta.
        c : float or ndarray
            Transaction cost parameter(s)
        mu : float or ndarray
            Mean reversion speed parameter(s)
        sigma : float or ndarray
            Volatility parameter(s)
        r : float
            Discount rate
        L : float or ndarray
            Loss level as spread (L - theta)
        exit_level : float or ndarray
            Exit level as spread (exit_level - theta)
        h : float or None, optional
            Step size for finite difference (default: None, uses module constant H)
        use_analytical : bool, optional
            If True, use analytical F and G functions (default: False)

        Returns
        -------
        float or ndarray
            Scalar if all inputs are scalar, otherwise array with shape
            determined by broadcasting the input arrays
        """
        if h is None:
            h = H
        return (
            OrnsteinUhlenbeck.V_L(
                x + h,
                c=c,
                mu=mu,
                sigma=sigma,
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
                r=r,
                L=L,
                exit_level=exit_level,
                use_analytical=use_analytical,
            )
        ) / h

    @staticmethod
    def get_optimal_exit_level(
        mu: np.ndarray,
        sigma: np.ndarray,
        theta: np.ndarray,
        discount_rate: float = 0.01,
        transaction_cost: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
        n_grid: int = 1000,
        degenerate_threshold: float = DEGENERATE_OU_THRESHOLD,
    ):
        """
        Vectorized computation of the Ornstein-Uhlenbeck optimal exit level given process parameters.

        Uses a grid search over spreads to find initial guesses, then refines with Newton's method.

        All arguments must be 1D numpy arrays of the same shape.
        Returns: 1D numpy array of exit_levels, or np.nan for positions where
        mu or sigma < degenerate_threshold (degenerate OU process) or any input is NaN/inf.
        """
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)
        theta = np.asarray(theta)

        # Initialize result with NaN
        result = np.full_like(mu, np.nan)

        # Filter out degenerate OU processes and invalid inputs.
        # mu < threshold: degenerates to Brownian motion (no mean reversion, r/mu diverges)
        # sigma < threshold: degenerates to deterministic process (no stochastic component)
        valid = (
            (mu >= degenerate_threshold)
            & (sigma >= degenerate_threshold)
            & np.isfinite(mu)
            & np.isfinite(sigma)
            & np.isfinite(theta)
        )
        if not np.any(valid):
            return result

        mu_v = mu[valid]
        sigma_v = sigma[valid]
        theta_v = theta[valid]
        r = discount_rate
        c = transaction_cost

        # Function f(x) operating on absolute positions
        def f_exit_level(x, mu, sigma, theta, r, c):
            return (x - c) * OrnsteinUhlenbeck.F(
                x, mu, sigma, theta=theta, r=r, derivative=1, use_analytical=True
            ) - OrnsteinUhlenbeck.F(x, mu, sigma, theta=theta, r=r, use_analytical=True)

        # Derivative of f(x) using analytical second derivative
        def f_prime_exit_level(x, mu, sigma, theta, r, c):
            return (x - c) * OrnsteinUhlenbeck.F(
                x, mu, sigma, theta=theta, r=r, derivative=2, use_analytical=True
            )

        # Grid search for initial guesses (in absolute position space)
        # For single parameter set, create 1D grid; for multiple, create 2D grid
        if len(mu_v) == 1:
            # Single parameter: create 1D grid for efficiency
            x_grid = np.linspace(float(theta_v), float(theta_v + 100 * sigma_v), n_grid)
            f_grid = f_exit_level(x_grid, mu_v, sigma_v, theta_v, r, c)
            fp_grid = f_prime_exit_level(x_grid, mu_v, sigma_v, theta_v, r, c)
            # Filter: f(x) > 0 and f'(x) > 0 (both must be positive)
            f_search = np.where((f_grid > 0) & (fp_grid > 0), f_grid, np.inf)
            x0 = np.array([x_grid[np.argmin(f_search)]])
        else:
            # Multiple parameters: create 2D grid
            # x_grid shape: (n_grid, n_params)
            x_grid = np.linspace(theta_v, theta_v + 100 * sigma_v, n_grid)
            f_grid = f_exit_level(x_grid, mu_v, sigma_v, theta_v, r, c)
            fp_grid = f_prime_exit_level(x_grid, mu_v, sigma_v, theta_v, r, c)
            # Filter: f(x) > 0 and f'(x) > 0 (both must be positive)
            f_search = np.where((f_grid > 0) & (fp_grid > 0), f_grid, np.inf)
            x0 = x_grid[np.argmin(f_search, axis=0), np.arange(len(mu_v))]

        # Newton's method
        x = x0
        active = np.ones(len(x), dtype=bool)
        for _ in range(max_iter):
            if not np.any(active):
                break
            f_val = f_exit_level(
                x[active], mu_v[active], sigma_v[active], theta_v[active], r, c
            )
            fp_val = f_prime_exit_level(
                x[active], mu_v[active], sigma_v[active], theta_v[active], r, c
            )
            safe = (fp_val != 0) & np.isfinite(fp_val) & np.isfinite(f_val)
            step = np.zeros_like(f_val)
            np.divide(f_val, fp_val, out=step, where=safe)
            x[active] = x[active] - step
            active_idx = np.flatnonzero(active)
            # NaN out elements that hit invalid values
            x[active_idx[~safe]] = np.nan
            done = (np.abs(f_val) < tol) | ~safe
            active[active_idx[done]] = False

        # Set non-converged values to NaN
        n_unconverged = np.sum(active)
        if n_unconverged > 0:
            warnings.warn(
                f"get_optimal_exit_level: {n_unconverged} roots did not converge "
                f"within {max_iter} iterations."
            )
            x[active] = np.nan

        # Result is already in absolute positions
        result[valid] = x
        return result

    @staticmethod
    def get_optimal_entry_level(
        mu: np.ndarray,
        sigma: np.ndarray,
        theta: np.ndarray,
        exit_level: np.ndarray,
        discount_rate: float,
        transaction_cost: float,
        max_iter: int = 1000,
        tol: float = 1e-6,
        n_grid: int = 1000,
        degenerate_threshold: float = DEGENERATE_OU_THRESHOLD,
    ):
        """
        Compute optimal entry level using grid search for initial guess then Newton's method.

        All arguments must be 1D numpy arrays of the same shape.
        Returns: 1D numpy array of entry levels (absolute values), or np.nan where
        mu or sigma < degenerate_threshold (degenerate OU process) or any input is NaN/inf.
        """
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)
        theta = np.asarray(theta)
        exit_level = np.asarray(exit_level)

        # Initialize result with NaN
        result = np.full_like(mu, np.nan)

        # Filter out degenerate OU processes and invalid inputs.
        # mu < threshold: degenerates to Brownian motion (no mean reversion, r/mu diverges)
        # sigma < threshold: degenerates to deterministic process (no stochastic component)
        valid = (
            (mu >= degenerate_threshold)
            & (sigma >= degenerate_threshold)
            & np.isfinite(mu)
            & np.isfinite(sigma)
            & np.isfinite(theta)
            & np.isfinite(exit_level)
        )
        if not np.any(valid):
            return result

        mu_v = mu[valid]
        sigma_v = sigma[valid]
        theta_v = theta[valid]
        exit_level_v = exit_level[valid]
        r = discount_rate
        c = transaction_cost

        # Function f(x) operating on absolute positions
        def f_entry_level(x, mu, sigma, theta, r, c, exit_level):
            return (
                OrnsteinUhlenbeck.G(
                    x, mu, sigma, theta=theta, r=r, derivative=1, use_analytical=True
                )
                * (
                    OrnsteinUhlenbeck.V(
                        x,
                        mu,
                        sigma,
                        theta=theta,
                        r=r,
                        c=c,
                        exit_level=exit_level,
                        use_analytical=True,
                    )
                    - x
                    - c
                )
            ) - (
                OrnsteinUhlenbeck.G(x, mu, sigma, theta=theta, r=r, use_analytical=True)
                * (
                    OrnsteinUhlenbeck.V_prime(
                        x,
                        mu,
                        sigma,
                        theta=theta,
                        r=r,
                        c=c,
                        exit_level=exit_level,
                        use_analytical=True,
                    )
                    - 1
                )
            )

        # Derivative of f(x) using analytical second derivative
        def f_prime_entry_level(x, mu, sigma, theta, r, c, exit_level):
            return (
                OrnsteinUhlenbeck.G(
                    x, mu, sigma, theta=theta, r=r, derivative=2, use_analytical=True
                )
                * (
                    OrnsteinUhlenbeck.V(
                        x,
                        mu,
                        sigma,
                        theta=theta,
                        r=r,
                        c=c,
                        exit_level=exit_level,
                        use_analytical=True,
                    )
                    - x
                    - c
                )
            ) - (
                OrnsteinUhlenbeck.G(x, mu, sigma, theta=theta, r=r, use_analytical=True)
                * (
                    OrnsteinUhlenbeck.V_double_prime(
                        x,
                        mu,
                        sigma,
                        theta=theta,
                        r=r,
                        c=c,
                        exit_level=exit_level,
                        use_analytical=True,
                    )
                )
            )

        # Grid search for initial guesses (in absolute position space)
        # For single parameter set, create 1D grid; for multiple, create 2D grid
        if len(mu_v) == 1:
            # Single parameter: create 1D grid for efficiency
            x_grid = np.linspace(float(theta_v - 100 * sigma_v), float(theta_v), n_grid)
            f_grid = f_entry_level(x_grid, mu_v, sigma_v, theta_v, r, c, exit_level_v)
            fp_grid = f_prime_entry_level(
                x_grid, mu_v, sigma_v, theta_v, r, c, exit_level_v
            )
            # Filter: f(x) < 0 and f'(x) > 0
            f_search = np.where((f_grid < 0) & (fp_grid > 0), f_grid, np.inf)
            x0 = np.array([x_grid[np.argmin(f_search)]])
        else:
            # Multiple parameters: create 2D grid
            # x_grid shape: (n_grid, n_params)
            x_grid = np.linspace(theta_v - 100 * sigma_v, theta_v, n_grid)
            f_grid = f_entry_level(x_grid, mu_v, sigma_v, theta_v, r, c, exit_level_v)
            fp_grid = f_prime_entry_level(
                x_grid, mu_v, sigma_v, theta_v, r, c, exit_level_v
            )
            # Filter: f(x) < 0 and f'(x) > 0
            f_search = np.where((f_grid < 0) & (fp_grid > 0), f_grid, np.inf)
            x0 = x_grid[np.argmin(f_search, axis=0), np.arange(len(mu_v))]

        # Newton's method
        x = x0
        active = np.ones(len(x), dtype=bool)
        for _ in range(max_iter):
            if not np.any(active):
                break
            f_val = f_entry_level(
                x[active],
                mu_v[active],
                sigma_v[active],
                theta_v[active],
                r,
                c,
                exit_level_v[active],
            )
            fp_val = f_prime_entry_level(
                x[active],
                mu_v[active],
                sigma_v[active],
                theta_v[active],
                r,
                c,
                exit_level_v[active],
            )
            safe = (fp_val != 0) & np.isfinite(fp_val) & np.isfinite(f_val)
            step = np.zeros_like(f_val)
            np.divide(f_val, fp_val, out=step, where=safe)
            x[active] = x[active] - step
            active_idx = np.flatnonzero(active)
            # NaN out elements that hit invalid values
            x[active_idx[~safe]] = np.nan
            done = (np.abs(f_val) < tol) | ~safe
            active[active_idx[done]] = False

        # Set non-converged values to NaN
        n_unconverged = np.sum(active)
        if n_unconverged > 0:
            warnings.warn(
                f"get_optimal_entry_level: {n_unconverged} roots did not converge "
                f"within {max_iter} iterations."
            )
            x[active] = np.nan

        # Result is already in absolute positions
        result[valid] = x
        return result
