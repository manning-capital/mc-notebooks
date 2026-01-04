from typing import Optional
import numpy as np
import statsmodels.api as sm
from scipy.integrate import quad
from scipy.special import pbdv, gammaln
from scipy.stats import norm
from scipy.optimize import fsolve, brentq

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
    def C(
        c: float | np.ndarray,
        mu: float | np.ndarray,
        sigma: float | np.ndarray,
        theta: float | np.ndarray,
        r: float,
        L: float | np.ndarray,
        b_star: float | np.ndarray,
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
        b_star : float or ndarray
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
            (b_star - c)
            * OrnsteinUhlenbeck.G(
                L, mu=mu, sigma=sigma, theta=theta, r=r, use_analytical=use_analytical
            )
            - (L - c)
            * OrnsteinUhlenbeck.G(
                b_star,
                mu=mu,
                sigma=sigma,
                theta=theta,
                r=r,
                use_analytical=use_analytical,
            )
        ) / (
            OrnsteinUhlenbeck.F(
                b_star,
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
                b_star,
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
        b_star: float | np.ndarray,
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
        b_star : float or ndarray
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
                b_star,
                mu=mu,
                sigma=sigma,
                theta=theta,
                r=r,
                use_analytical=use_analytical,
            )
            - (b_star - c)
            * OrnsteinUhlenbeck.F(
                L, mu=mu, sigma=sigma, theta=theta, r=r, use_analytical=use_analytical
            )
        ) / (
            OrnsteinUhlenbeck.F(
                b_star,
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
                b_star,
                mu=mu,
                sigma=sigma,
                theta=theta,
                r=r,
                use_analytical=use_analytical,
            )
        )

    @staticmethod
    def V(
        x: float | np.ndarray,
        c: float | np.ndarray,
        mu: float | np.ndarray,
        sigma: float | np.ndarray,
        theta: float | np.ndarray,
        r: float,
        L: float | np.ndarray,
        b_star: float | np.ndarray,
        use_analytical: bool = False,
    ) -> float | np.ndarray:
        """
        The value function V(x, r).

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
        b_star : float or ndarray
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
        b_star_arr = np.atleast_1d(b_star)

        # Broadcast all arrays to the same shape
        x_arr, c_arr, mu_arr, sigma_arr, theta_arr, L_arr, b_star_arr = (
            np.broadcast_arrays(
                x_arr, c_arr, mu_arr, sigma_arr, theta_arr, L_arr, b_star_arr
            )
        )

        # Element-wise condition: x < b_star and x > L
        condition = (x_arr < b_star_arr) & (x_arr > L_arr)

        # Compute both branches
        waiting_value = OrnsteinUhlenbeck.C(
            c=c_arr,
            mu=mu_arr,
            sigma=sigma_arr,
            theta=theta_arr,
            r=r,
            L=L_arr,
            b_star=b_star_arr,
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
            b_star=b_star_arr,
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
            and np.isscalar(b_star)
        ):
            return float(result)

        return result

    @staticmethod
    def V_prime(
        x: float | np.ndarray,
        c: float | np.ndarray,
        mu: float | np.ndarray,
        sigma: float | np.ndarray,
        theta: float | np.ndarray,
        r: float,
        L: float | np.ndarray,
        b_star: float | np.ndarray,
        h: float = 1e-6,
        use_analytical: bool = False,
    ) -> float | np.ndarray:
        """
        The derivative of the value function V(x, r).

        Uses finite difference approximation: (V(x+h) - V(x)) / h

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
        b_star : float or ndarray
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
            OrnsteinUhlenbeck.V(
                x + h,
                c=c,
                mu=mu,
                sigma=sigma,
                theta=theta,
                r=r,
                L=L,
                b_star=b_star,
                use_analytical=use_analytical,
            )
            - OrnsteinUhlenbeck.V(
                x,
                c=c,
                mu=mu,
                sigma=sigma,
                theta=theta,
                r=r,
                L=L,
                b_star=b_star,
                use_analytical=use_analytical,
            )
        ) / h

    @staticmethod
    def _find_root_grid(
        f_func,
        search_low: np.ndarray,
        search_high: np.ndarray,
        n_grid: int = 50,
        use_brentq: bool = True,
    ) -> np.ndarray:
        """
        Find roots using a grid-based approach to find brackets, then brentq for refinement.

        Parameters
        ----------
        f_func : callable
            Function that takes a 2D array (n_elements, n_grid) and returns
            function values of shape (n_elements, n_grid)
        search_low : ndarray
            Lower bounds for search, shape (n_elements,)
        search_high : ndarray
            Upper bounds for search, shape (n_elements,)
        n_grid : int, optional
            Number of grid points for bracket finding (default: 50)
        use_brentq : bool, optional
            If True, use brentq to refine roots (default: True)

        Returns
        -------
        ndarray
            Root values, shape (n_elements,)
        """
        n_elements = search_low.shape[0]

        # Create a scalar function wrapper for each element
        # We need to create element-specific functions for brentq
        def create_element_func(i):
            """Create a function for element i that can be used with brentq"""

            def element_f(x):
                # Create a grid with just this one value for this element
                # We need to create a full grid but only evaluate at the right position
                grid_single = np.zeros((n_elements, 1))
                grid_single[i, 0] = float(x)
                # Evaluate only for this element's grid
                result = f_func(grid_single)
                return float(result[i, 0])

            return element_f

        # Create grid for all elements to find brackets
        # Shape: (n_elements, n_grid)
        grid = np.array(
            [
                np.linspace(search_low[i], search_high[i], n_grid)
                for i in range(n_elements)
            ]
        )

        # Evaluate function at all grid points (vectorized)
        # Shape: (n_elements, n_grid)
        f_values = f_func(grid)

        # Ensure f_values has correct shape
        if f_values.shape != (n_elements, n_grid):
            raise ValueError(
                f"f_func returned shape {f_values.shape}, expected ({n_elements}, {n_grid})"
            )

        # Find zero crossings for each element
        roots = np.full(n_elements, np.nan)

        for i in range(n_elements):
            f_vals = np.asarray(f_values[i, :]).flatten()
            grid_vals = np.asarray(grid[i, :]).flatten()

            # Find sign changes
            signs = np.sign(f_vals)
            sign_changes = np.where(np.diff(signs) != 0)[0]

            if len(sign_changes) > 0:
                # Use first sign change as bracket
                idx = int(sign_changes[0])
                bracket_low = float(grid_vals[idx])
                bracket_high = float(grid_vals[idx + 1])

                if use_brentq:
                    # Use brentq to find precise root in bracket
                    try:
                        element_f = create_element_func(i)
                        root = brentq(
                            element_f, bracket_low, bracket_high, xtol=1e-8, maxiter=100
                        )
                        roots[i] = root
                    except (ValueError, RuntimeError):
                        # Fallback to linear interpolation if brentq fails
                        f1, f2 = float(f_vals[idx]), float(f_vals[idx + 1])
                        if abs(f2 - f1) > 1e-15:
                            root = bracket_low - f1 * (bracket_high - bracket_low) / (
                                f2 - f1
                            )
                        else:
                            root = (bracket_low + bracket_high) / 2
                        roots[i] = root
                else:
                    # Linear interpolation
                    f1, f2 = float(f_vals[idx]), float(f_vals[idx + 1])
                    if abs(f2 - f1) > 1e-15:
                        root = bracket_low - f1 * (bracket_high - bracket_low) / (
                            f2 - f1
                        )
                    else:
                        root = (bracket_low + bracket_high) / 2
                    roots[i] = root
            else:
                # No sign change found, use point closest to zero
                abs_f_vals = np.abs(f_vals)
                # Handle NaN/inf values
                abs_f_vals = np.where(np.isfinite(abs_f_vals), abs_f_vals, np.inf)
                min_idx = int(np.argmin(abs_f_vals))
                # Ensure index is valid
                min_idx = max(0, min(min_idx, len(grid_vals) - 1))
                roots[i] = float(grid_vals[min_idx])

        return roots

    @staticmethod
    def get_optimal_exit_level(
        mu: float | np.ndarray,
        sigma: float | np.ndarray,
        theta: float | np.ndarray,
        r: float = 0.01,
        c: float | np.ndarray = 0.001,
        L: Optional[float | np.ndarray] = None,
        h: float = 1e-6,
        initial_guess: Optional[float | np.ndarray] = None,
        use_analytical: bool = False,
    ) -> float | np.ndarray:
        """
        Computes the optimal exit level (b_star) for an OU process.
        Uses Brent's method (brentq) for root finding.

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
        initial_guess : float or ndarray, optional
            Starting point for root finding. If None, defaults to theta + 0.1*sigma (default: None)
        use_analytical : bool, optional
            If True, use analytical F and G functions (default: False)

        Returns
        -------
        float or ndarray
            Optimal exit level (b_star) - should be well above theta.
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

        # Handle initial_guess
        if initial_guess is None:
            initial_guess_arr = theta_arr + 0.1 * sigma_arr
        else:
            initial_guess_arr = (
                np.atleast_1d(initial_guess)
                if not np.isscalar(initial_guess)
                else np.array([initial_guess])
            )
            initial_guess_arr = np.broadcast_to(initial_guess_arr, mu_arr.shape)

        # Process each element using brentq
        n_elements = mu_arr.size
        result = np.empty(mu_arr.shape, dtype=float)

        # Flatten for iteration
        mu_flat = mu_arr.flatten()
        sigma_flat = sigma_arr.flatten()
        theta_flat = theta_arr.flatten()
        c_flat = c_arr.flatten()
        L_flat = L_arr.flatten()
        initial_guess_flat = initial_guess_arr.flatten()

        for idx in np.ndindex(mu_arr.shape):
            i = np.ravel_multi_index(idx, mu_arr.shape)
            mu_val = float(mu_flat[i])
            sigma_val = float(sigma_flat[i])
            theta_val = float(theta_flat[i])
            c_val = float(c_flat[i])
            L_val = float(L_flat[i])
            initial_guess_val = float(initial_guess_flat[i])

            # Pre-compute L-dependent values (constant for this element)
            F_L = OrnsteinUhlenbeck.F(
                L_val, mu_val, sigma_val, theta_val, r, use_analytical=use_analytical
            )
            G_L = OrnsteinUhlenbeck.G(
                L_val, mu_val, sigma_val, theta_val, r, use_analytical=use_analytical
            )

            # Define the function to find root of
            def f(b):
                b_val = float(b)
                G_b = OrnsteinUhlenbeck.G(
                    b_val,
                    mu_val,
                    sigma_val,
                    theta_val,
                    r,
                    use_analytical=use_analytical,
                )
                F_prime_b = OrnsteinUhlenbeck.F_prime(
                    b_val,
                    mu_val,
                    sigma_val,
                    theta_val,
                    r,
                    h=h,
                    use_analytical=use_analytical,
                )
                F_b = OrnsteinUhlenbeck.F(
                    b_val,
                    mu_val,
                    sigma_val,
                    theta_val,
                    r,
                    use_analytical=use_analytical,
                )
                G_prime_b = OrnsteinUhlenbeck.G_prime(
                    b_val,
                    mu_val,
                    sigma_val,
                    theta_val,
                    r,
                    h=h,
                    use_analytical=use_analytical,
                )

                f_left = ((L_val - c_val) * G_b - (b_val - c_val) * F_prime_b) + (
                    (b_val - c_val) * F_L - (L_val - c_val) * F_b
                ) * G_prime_b

                f_right = G_b * F_L - G_L * F_b

                return f_left - f_right

            # Define search bounds
            bracket_low = max(theta_val + 0.001, L_val + 0.001, theta_val + 0.01)
            bracket_high = min(theta_val + 2 * sigma_val, theta_val + 0.5 * sigma_val)

            # Try multiple bracket ranges
            bracket_ranges = [
                (theta_val + 0.01, theta_val + 0.05),
                (theta_val + 0.05, theta_val + 0.15),
                (theta_val + 0.1 * sigma_val, theta_val + 0.3 * sigma_val),
                (bracket_low, bracket_high),
            ]

            found = False
            for br_low, br_high in bracket_ranges:
                br_low = max(br_low, theta_val + 0.001, L_val + 0.001)
                br_high = min(br_high, theta_val + 2 * sigma_val)

                if br_low >= br_high:
                    continue

                # Check if bracket has opposite signs
                f_low = f(br_low)
                f_high = f(br_high)

                if f_low * f_high < 0:
                    try:
                        b_result = brentq(f, br_low, br_high, xtol=1e-8, maxiter=100)
                        if (
                            b_result > theta_val
                            and b_result > L_val
                            and b_result < theta_val + 2 * sigma_val
                        ):
                            result[idx] = b_result
                            found = True
                            break
                    except (ValueError, RuntimeError):
                        continue

            if not found:
                # Fallback: use fsolve with initial guess
                try:
                    b_result = fsolve(f, initial_guess_val, xtol=1e-8, maxfev=200)[0]
                    result[idx] = max(
                        min(b_result, theta_val + 2 * sigma_val),
                        theta_val + 0.01,
                        L_val + 0.01,
                    )
                except:
                    result[idx] = initial_guess_val

        # Return scalar if input was scalar
        if (
            np.isscalar(mu)
            and np.isscalar(sigma)
            and np.isscalar(theta)
            and np.isscalar(c)
            and (L is None or np.isscalar(L))
            and (initial_guess is None or np.isscalar(initial_guess))
        ):
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
        b_star: Optional[float | np.ndarray] = None,
        use_analytical: bool = False,
    ) -> float | np.ndarray:
        """
        Computes the optimal entry level (d_star) for an OU process.
        Uses Brent's method (brentq) for root finding.

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
        initial_guess : float or ndarray, optional
            Starting point for root finding. If None, defaults to (L + theta) / 2 (default: None)
        b_star : float or ndarray, optional
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

        # Get b_star first (needed for d_star calculation) if not provided
        if b_star is None:
            b_star_arr = OrnsteinUhlenbeck.get_optimal_exit_level(
                mu=mu_arr,
                sigma=sigma_arr,
                theta=theta_arr,
                r=r,
                c=c_arr,
                L=L_arr,
                h=h,
                use_analytical=use_analytical,
            )
            b_star_arr = np.atleast_1d(b_star_arr)
        else:
            b_star_arr = (
                np.atleast_1d(b_star) if not np.isscalar(b_star) else np.array([b_star])
            )
            b_star_arr = np.broadcast_to(b_star_arr, mu_arr.shape)

        # Handle initial_guess
        if initial_guess is None:
            initial_guess_arr = (L_arr + theta_arr) / 2
        else:
            initial_guess_arr = (
                np.atleast_1d(initial_guess)
                if not np.isscalar(initial_guess)
                else np.array([initial_guess])
            )
            initial_guess_arr = np.broadcast_to(initial_guess_arr, mu_arr.shape)

        # Process each element using brentq
        n_elements = mu_arr.size
        result = np.empty(mu_arr.shape, dtype=float)

        # Flatten for iteration
        mu_flat = mu_arr.flatten()
        sigma_flat = sigma_arr.flatten()
        theta_flat = theta_arr.flatten()
        c_flat = c_arr.flatten()
        L_flat = L_arr.flatten()
        b_star_flat = b_star_arr.flatten()
        initial_guess_flat = initial_guess_arr.flatten()

        for idx in np.ndindex(mu_arr.shape):
            i = np.ravel_multi_index(idx, mu_arr.shape)
            mu_val = float(mu_flat[i])
            sigma_val = float(sigma_flat[i])
            theta_val = float(theta_flat[i])
            c_val = float(c_flat[i])
            L_val = float(L_flat[i])
            b_star_val = float(b_star_flat[i])
            initial_guess_val = float(initial_guess_flat[i])

            # Define the function to find root of
            def f(d):
                d_val = float(d)
                G_d = OrnsteinUhlenbeck.G(
                    d_val,
                    mu_val,
                    sigma_val,
                    theta_val,
                    r,
                    use_analytical=use_analytical,
                )
                V_prime_d = OrnsteinUhlenbeck.V_prime(
                    d_val,
                    c_val,
                    mu_val,
                    sigma_val,
                    theta_val,
                    r,
                    L_val,
                    b_star_val,
                    h,
                    use_analytical=use_analytical,
                )
                G_prime_d = OrnsteinUhlenbeck.G_prime(
                    d_val,
                    mu_val,
                    sigma_val,
                    theta_val,
                    r,
                    h=h,
                    use_analytical=use_analytical,
                )
                V_d = OrnsteinUhlenbeck.V(
                    d_val,
                    c_val,
                    mu_val,
                    sigma_val,
                    theta_val,
                    r,
                    L_val,
                    b_star_val,
                    use_analytical=use_analytical,
                )

                f_left = G_d * (V_prime_d - 1)
                f_right = G_prime_d * (V_d - d_val - c_val)

                return f_left - f_right

            # Define search bounds - d_star should be between L and theta
            bracket_low = max(L_val + 0.001, L_val + 0.01 * sigma_val)
            bracket_high = min(theta_val - 0.001, theta_val - 0.01 * sigma_val)

            # Try multiple bracket ranges
            bracket_ranges = [
                (L_val + 0.01 * sigma_val, theta_val - 0.01 * sigma_val),
                (L_val + 0.05 * sigma_val, theta_val - 0.05 * sigma_val),
                (L_val + 0.1 * sigma_val, theta_val - 0.1 * sigma_val),
                (bracket_low, bracket_high),
            ]

            found = False
            for br_low, br_high in bracket_ranges:
                br_low = max(br_low, L_val + 0.001)
                br_high = min(br_high, theta_val - 0.001)

                if br_low >= br_high:
                    continue

                # Check if bracket has opposite signs
                f_low = f(br_low)
                f_high = f(br_high)

                if f_low * f_high < 0:
                    try:
                        d_result = brentq(f, br_low, br_high, xtol=1e-8, maxiter=100)
                        if L_val < d_result < theta_val:
                            result[idx] = d_result
                            found = True
                            break
                    except (ValueError, RuntimeError):
                        continue

            if not found:
                # Fallback: use fsolve with initial guess
                try:
                    d_result = fsolve(f, initial_guess_val, xtol=1e-8, maxfev=200)[0]
                    result[idx] = max(min(d_result, theta_val - 0.001), L_val + 0.001)
                except:
                    result[idx] = initial_guess_val

        # Return scalar if input was scalar
        if (
            np.isscalar(mu)
            and np.isscalar(sigma)
            and np.isscalar(theta)
            and np.isscalar(c)
            and (L is None or np.isscalar(L))
            and (initial_guess is None or np.isscalar(initial_guess))
            and (b_star is None or np.isscalar(b_star))
        ):
            return float(result)

        return result
