"""Dask-compatible helper functions for stochastic analysis."""

import datetime as dt
import pandas as pd

from .rolling_v2 import RollingCointegration, RollingOrnsteinUhlenbeck


def rolling_ornstein_uhlenbeck(df: pd.DataFrame, window: dt.timedelta) -> pd.DataFrame:
    """Apply rolling Ornstein-Uhlenbeck to a DataFrame and return merged result with provider_asset_group_id as index."""
    # Copy the input DataFrame.
    output_df = df.copy()

    # Compute rolling Ornstein-Uhlenbeck
    cointegration_result = RollingCointegration(
        y0=output_df["close_1"].to_numpy(),
        y1=output_df["close_2"].to_numpy(),
        window=int(window.total_seconds() // 60),
    ).fit()

    # Create DataFrame with cointegration results indexed by timestamp
    timestamp_values = (
        output_df["timestamp"].values
        if hasattr(output_df["timestamp"], "values")
        else output_df["timestamp"]
    )
    cointegration_df = pd.DataFrame(
        {
            "beta": cointegration_result.beta,
            "pvalue": cointegration_result.pvalue,
            "residual_mean": cointegration_result.residual_mean,
            "residual_std": cointegration_result.residual_std,
        },
        index=timestamp_values,
    )
    cointegration_df.dropna(inplace=True)

    # Merge with original DataFrame
    output_df = output_df.merge(
        cointegration_df,
        left_on="timestamp",
        right_index=True,
        how="inner",
    )

    # Compute the Ornstein-Uhlenbeck parameters
    ou_result = RollingOrnsteinUhlenbeck(
        beta=cointegration_result.beta,
        y0=df["close_1"].to_numpy(),
        y1=df["close_2"].to_numpy(),
        window=int(window.total_seconds() // 60),
    ).fit()

    # Create DataFrame with OU results indexed by timestamp
    ou_df = pd.DataFrame(
        {
            "mu": ou_result.mu,
            "sigma": ou_result.sigma,
            "theta": ou_result.theta,
            "half_life": ou_result.half_life,
        },
        index=timestamp_values,
    )
    ou_df.dropna(inplace=True)

    # Merge with OU results
    output_df = output_df.merge(
        ou_df,
        left_on="timestamp",
        right_index=True,
        how="inner",
    )

    # Reset index to ensure clean state
    output_df = output_df.reset_index(drop=True)

    return output_df.set_index("provider_asset_group_id")