import numpy as np
from numpy.typing import NDArray
from typing import Literal, Optional
from dataclasses import dataclass


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
    entry_level: NDArray[np.float64]  # 1D array of float64
    exit_level: NDArray[np.float64]  # 1D array of float64
    loss_level: NDArray[np.float64]  # 1D array of float64


class RollingCointegration:
    """
    Rolling cointegration test.
    """

    def __init__(
        self,
        y0: np.ndarray,
        y1: np.ndarray,
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

    def fit(self, method: Literal["inv", "lstsq", "pinv"] = "inv"):
        """
        Fit the rolling cointegration model.
        """
        # TODO: Implement this.
        pass


class RollingOrnsteinUhlenbeck:
    """
    Rolling Ornstein-Uhlenbeck parameter estimation.
    """

    def __init__(
        self,
        alpha: np.ndarray,
        beta: np.ndarray,
        p_value: np.ndarray,
        loss_level: np.ndarray,
        y0: np.ndarray,
        y1: np.ndarray,
        window: int,
        r: float = 0.0001,
        c: float = 0.001,
        p_value_threshold: float = 0.05,
        trend: Literal["c", "ct", "ctt", "n"] = "c",
        lag: int = 1,
        min_nobs: Optional[int] = None,
        expanding: bool = False,
    ):
        self.alpha = alpha
        self.beta = beta
        self.p_value = p_value
        self.loss_level = loss_level
        self.y0 = y0
        self.y1 = y1
        self.window = window
        self.r = r
        self.c = c
        self.p_value_threshold = p_value_threshold
        self.trend = trend
        self.lag = lag
        self.min_nobs = min_nobs
        self.expanding = expanding

    def fit(self, method: Literal["inv", "lstsq", "pinv"] = "inv"):
        """
        Fit the rolling Ornstein-Uhlenbeck model.
        """
        # TODO: Implement this.
        pass
