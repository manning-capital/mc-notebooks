from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

# Constants: These are the parameters that are used to estimate the GBM and OU processes they should never be changed.
DELTA_T = 1


class StochasticModelResult(ABC):
    """
    Base class for stochastic model results.
    """

    @abstractmethod
    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def to_dict(self) -> dict:
        """
        Convert the results to a dictionary.
        """
        pass


@dataclass
class GeometricBrownianMotionResult(StochasticModelResult):
    """
    Results for the Geometric Brownian Motion (GBM) process.

    Parameters
    ----------
    mu : float
        Drift parameter (μ) in the SDE: dS = mu * S * dt + sigma * S * dW
    sigma : float
        Volatility parameter (σ) in the SDE: dS = mu * S * dt + sigma * S * dW
    """

    mu: float  # drift parameter
    sigma: float  # volatility parameter

    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma
        self.params = {"mu": self.mu, "sigma": self.sigma}

    def to_dict(self) -> dict:
        """
        Convert the results to a dictionary.
        """
        return {"mu": self.mu, "sigma": self.sigma}


@dataclass
class OrnsteinUhlenbeckResult(StochasticModelResult):
    """
    Results for the Ornstein-Uhlenbeck process.

    Parameters
    ----------
    mu : float
        Mean reversion parameter (μ) in the SDE: dX = mu * (theta - X) dt + sigma * dW
    theta : float
        Asymptotic mean (θ) in the SDE: dX = mu * (theta - X) dt + sigma * dW
    sigma : float
        Brownian motion scale (σ) in the SDE: dX = mu * (theta - X) dt + sigma * dW
    """

    mu: float  # mean reversion parameter
    theta: float  # asymptotic mean
    sigma: float  # Brownian motion scale (standard deviation)

    def __init__(
        self,
        mu: float,
        theta: float,
        sigma: float,
    ):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.params = {
            "mu": self.mu,
            "theta": self.theta,
            "sigma": self.sigma,
        }

    def to_dict(self) -> dict:
        return {
            "mu": self.mu,
            "theta": self.theta,
            "sigma": self.sigma,
        }


class StochasticModel(ABC):
    """
    Base class for stochastic models.
    """

    __params: StochasticModelResult

    def __init__(self, params: StochasticModelResult = None):
        self.__params = params

    @property
    def params(self) -> StochasticModelResult:
        """
        Get the parameters of the model.
        """
        if self.__params is None:
            raise ValueError("Parameters are not set for the model.")
        return self.__params

    @params.setter
    def params(self, params: StochasticModelResult):
        """
        Set the parameters of the model.
        """
        self.__params = params

    @abstractmethod
    def log_likelihood(self, X: np.ndarray) -> float:
        """
        Compute the log likelihood of the model.
        """
        pass

    @abstractmethod
    def fit(self, X: np.ndarray):
        """
        Fit the model to the data.
        """
        pass

    @abstractmethod
    def simulate(self, N: int, N_simulated: int, X_0: float) -> np.ndarray:
        """
        Simulate the model.
        """
        pass
