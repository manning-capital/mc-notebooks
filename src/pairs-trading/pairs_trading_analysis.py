"""
Pairs Trading Analysis Module

This module provides functionality to analyze cointegrated pairs of assets
using Dask for parallel processing. The main entry point is a Prefect flow
named 'save-pairs-trading-attributes' that performs the analysis and saves
results to the ProviderAssetGroupAttribute table.

Database Schema Overview:
-------------------------
This module works with the following database tables (ORM models):

1. ProviderAssetGroup (models.ProviderAssetGroup)
   - Represents groups of asset pairs for pairs trading analysis
   - Primary key: id

2. ProviderAssetGroupMember (models.ProviderAssetGroupMember)
   - Defines the two assets in each pairs trading group
   - Links: provider_asset_group_id -> ProviderAssetGroup.id
   - Fields: provider_id, from_asset_id, to_asset_id, order (1 or 2)
   - Each group has exactly 2 members (order=1 and order=2)

3. ProviderAssetMarket (models.ProviderAssetMarket)
   - Contains historical market data (OHLCV) for asset pairs
   - Links to assets via: provider_id, from_asset_id, to_asset_id
   - Fields: timestamp, open, high, low, close, volume
   - Indexed by timestamp for efficient time-series queries

4. ProviderAssetGroupAttribute (models.ProviderAssetGroupAttribute)
   - Stores computed cointegration statistics for each group
   - Links: provider_asset_group_id -> ProviderAssetGroup.id
   - Fields: timestamp, lookback_window_seconds, cointegration_p_value,
            linear_fit_alpha, linear_fit_beta, linear_fit_mse,
            linear_fit_r_squared, linear_fit_r_squared_adj,
            ou_mu (mean reversion level), ou_theta (mean reversion speed),
            ou_sigma (volatility)
   - Results from this analysis are written to this table
"""

import datetime as dt
import sys
from pathlib import Path
from typing import Optional

import coiled
import dask.dataframe as dd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dask import delayed
from dask.diagnostics import ProgressBar
from prefect import flow, get_run_logger
from sqlalchemy import select
from sqlalchemy.orm import Session
from statsmodels.tsa.stattools import coint

# Handle imports from the workspace
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

import mc_postgres_db.models as models
from mc_postgres_db.prefect.tasks import get_engine, set_data
from src.utils.stochastic_models import OrnsteinUhlenbeck

# Configuration constants
DEFAULT_N_WORKERS = 20
DEFAULT_MAX_GROUPS = 5000


@delayed
def load_pairs_trading_frame_chunk(
    start: dt.datetime,
    end: dt.datetime,
    members_chunk: pd.DataFrame,
    market_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load the pairs trading frame for a chunk of provider asset groups.
    Returns only the essential columns needed for cointegration analysis.

    This function joins data from ProviderAssetGroupMember with ProviderAssetMarket
    to create time-series pairs of closing prices for cointegration testing.

    Args:
        start: Start datetime (timezone-naive)
        end: End datetime (timezone-naive)
        members_chunk: Pre-filtered DataFrame from ProviderAssetGroupMember table
                      containing group membership data (provider_id, from_asset_id,
                      to_asset_id, order)
        market_data: Broadcasted DataFrame from ProviderAssetMarket table
                    containing historical price data

    Returns:
        pandas DataFrame indexed by provider_asset_group_id with columns:
            - timestamp: DateTime of the observation
            - close_1: Closing price of the first asset (order=1)
            - close_2: Closing price of the second asset (order=2)
    """
    # Generate timeframe using pd.date_range
    time_frame = pd.DataFrame({"timestamp": pd.date_range(start, end, freq="1min")})

    # Cross join with members_chunk (already filtered)
    full_frame = time_frame.merge(members_chunk, how="cross")
    full_frame = full_frame.sort_values("timestamp")

    # Merge_asof to get historical market data
    full_market_frame = pd.merge_asof(
        full_frame,
        market_data,
        on="timestamp",
        by=["provider_id", "from_asset_id", "to_asset_id"],
        direction="backward",
    )

    # Split by order and create pairs - only keep essential columns
    close_1 = full_market_frame[full_market_frame["order"] == 1][
        ["timestamp", "provider_asset_group_id", "close"]
    ].rename(columns={"close": "close_1"})
    close_2 = full_market_frame[full_market_frame["order"] == 2][
        ["timestamp", "provider_asset_group_id", "close"]
    ].rename(columns={"close": "close_2"})

    # Merge to create pairs - only timestamp, close_1, close_2
    pairs = pd.merge(
        close_1, close_2, on=["timestamp", "provider_asset_group_id"], how="inner"
    )

    # Keep only essential columns
    pairs = pairs[["provider_asset_group_id", "timestamp", "close_1", "close_2"]]

    # Set index to provider_asset_group_id
    pairs = pairs.set_index("provider_asset_group_id")

    return pairs


def get_pairs_trading_frame(
    start: dt.datetime,
    end: dt.datetime,
    provider_asset_group_ids: list[int],
    members_data: pd.DataFrame,
    market_data_future,
    n_workers: int = 10,
) -> dd.DataFrame:
    """
    Get the pairs trading frame with only essential columns for cointegration analysis.

    This function parallelizes the loading of pairs trading data across multiple
    ProviderAssetGroup IDs using Dask delayed execution.

    Args:
        start: Start datetime (timezone-naive)
        end: End datetime (timezone-naive)
        provider_asset_group_ids: List of IDs from ProviderAssetGroup table to process
        members_data: Pre-loaded DataFrame from ProviderAssetGroupMember table
        market_data_future: Broadcasted future containing ProviderAssetMarket data
        n_workers: Number of parallel workers for task distribution

    Returns:
        Dask DataFrame indexed by provider_asset_group_id (from ProviderAssetGroup)
        with columns:
            - timestamp: DateTime of the observation
            - close_1: Closing price of the first asset (order=1)
            - close_2: Closing price of the second asset (order=2)
    """
    # Split provider asset groups into chunks
    n_chunks = min(n_workers, len(provider_asset_group_ids))
    group_chunks = np.array_split(provider_asset_group_ids, n_chunks)

    # Create delayed tasks with filtered member chunks
    delayed_dfs = []
    for chunk in group_chunks:
        # Filter members data for this specific chunk
        members_chunk = members_data[
            members_data["provider_asset_group_id"].isin(chunk.tolist())
        ].copy()

        delayed_dfs.append(
            load_pairs_trading_frame_chunk(
                start, end, members_chunk, market_data_future
            )
        )

    # Define minimal schema
    meta = pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns]"),
            "close_1": pd.Series(dtype="float64"),
            "close_2": pd.Series(dtype="float64"),
        }
    )
    meta.index = pd.Index([], name="provider_asset_group_id", dtype="int64")

    # Convert to Dask DataFrame
    pairs_trading_frame = dd.from_delayed(delayed_dfs, meta=meta)

    # Set index to provider_asset_group_id
    pairs_trading_frame = pairs_trading_frame.set_index(
        "provider_asset_group_id", sorted=True
    )

    return pairs_trading_frame


def get_cointegrated_stats(df: pd.DataFrame) -> pd.Series:
    """
    Calculate cointegration statistics for a pair of assets.

    Performs OLS linear regression on the two price series and fits an
    Ornstein-Uhlenbeck process to the residuals to model mean reversion behavior.

    These statistics are written to the ProviderAssetGroupAttribute table.

    Args:
        df: DataFrame with close_1 and close_2 columns (closing prices from
            ProviderAssetMarket for the two assets in a pair)

    Returns:
        Series with the following statistics:
            - linear_fit_alpha: Intercept of the linear regression
            - linear_fit_beta: Slope of the linear regression
            - linear_fit_mse: Mean squared error of the fit
            - linear_fit_r_squared: R-squared value
            - linear_fit_r_squared_adj: Adjusted R-squared value
            - ou_mu: Mean reversion level (long-term mean of residuals)
            - ou_theta: Mean reversion speed (rate of return to mean)
            - ou_sigma: Volatility of the residuals
    """
    # Compute the linear regression
    X = df["close_1"].to_numpy()
    y = df["close_2"].to_numpy()
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()

    # Get the residuals
    linear_fit_alpha = results.params[0]
    linear_fit_beta = results.params[1]
    linear_fit_mse = results.mse_total
    linear_fit_r_squared = results.rsquared
    linear_fit_r_squared_adj = results.rsquared_adj
    residuals = results.resid

    # Get the cointegration stats
    ou_params = OrnsteinUhlenbeck().fit(residuals)

    return pd.Series(
        [
            linear_fit_alpha,
            linear_fit_beta,
            linear_fit_mse,
            linear_fit_r_squared,
            linear_fit_r_squared_adj,
            ou_params.mu,
            ou_params.theta,
            ou_params.sigma,
        ],
        index=[
            "linear_fit_alpha",
            "linear_fit_beta",
            "linear_fit_mse",
            "linear_fit_r_squared",
            "linear_fit_r_squared_adj",
            "ou_mu",
            "ou_theta",
            "ou_sigma",
        ],
        dtype=float,
    )


@flow(name="save-pairs-trading-attributes")
def save_pairs_trading_attributes(
    date: Optional[dt.date] = None,
    lookback_days: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prefect flow: Run comprehensive pairs trading analysis for a given date and lookback period.

    This function performs the following workflow:
    1. Retrieves postgres_url from Prefect Secret block
    2. Loads ProviderAssetGroup IDs to analyze
    3. Retrieves historical market data from ProviderAssetMarket
    4. Retrieves group membership from ProviderAssetGroupMember
    5. Creates pairs trading frame with aligned time series for each pair
    6. Computes cointegration p-values using Engle-Granger test
    7. For cointegrated pairs (p < 0.001), calculates detailed statistics:
       - Linear regression parameters (alpha, beta, R²)
       - Ornstein-Uhlenbeck parameters (mu, theta, sigma)
    8. Writes results to ProviderAssetGroupAttribute table

    Args:
        date: The date to run the analysis for (default: current date)
        lookback_days: Number of days of historical data to use (default: 30)

    Returns:
        Tuple of (cointegration_p_values_df, cointegrated_pairs_stats_df):
            - cointegration_p_values_df: DataFrame with p_value for each group
            - cointegrated_pairs_stats_df: DataFrame with detailed statistics for
              cointegrated pairs (empty if no pairs pass the cointegration test)

    Configuration:
        - Uses Coiled cluster with configuration from flow decorator
        - PostgreSQL URL retrieved from Prefect Secret 'postgres-url'
        - Uses DEFAULT_MAX_GROUPS and DEFAULT_N_WORKERS constants for processing
    """
    logger = get_run_logger()

    # Use current date if not specified
    if date is None:
        date = dt.datetime.now(dt.timezone.utc).date()

    # Get the database engine.
    engine = get_engine()

    # Calculate date range
    end = dt.datetime.combine(date, dt.time.min)
    start = end - dt.timedelta(days=lookback_days)
    start_naive = start.replace(tzinfo=None).replace(second=0, microsecond=0)
    end_naive = end.replace(tzinfo=None).replace(second=0, microsecond=0)
    logger.info(f"Analysis Date: {date}")
    logger.info(f"Start: {start}, End: {end}")
    logger.info(f"Lookback Days: {lookback_days}")

    # Load provider asset group IDs from ProviderAssetGroup table
    # Each group represents a potential pairs trading opportunity
    with Session(engine) as session:
        provider_asset_group_ids = session.scalars(
            select(models.ProviderAssetGroup.id).limit(DEFAULT_MAX_GROUPS)
        ).all()

    logger.info(
        f"Provider asset group ids (count: {len(provider_asset_group_ids)}): {provider_asset_group_ids}"
    )

    # Load historical market data from ProviderAssetMarket table
    # This contains OHLCV data for all asset pairs in the date range
    logger.info("Loading market data...")
    market_data = pd.read_sql(
        select(
            models.ProviderAssetMarket.timestamp,
            models.ProviderAssetMarket.provider_id,
            models.ProviderAssetMarket.from_asset_id,
            models.ProviderAssetMarket.to_asset_id,
            models.ProviderAssetMarket.close,
        )
        .where(models.ProviderAssetMarket.timestamp.between(start_naive, end_naive))
        .order_by(models.ProviderAssetMarket.timestamp),
        engine,
    )
    logger.info(f"Market data loaded: {len(market_data)} rows")

    # Load group membership from ProviderAssetGroupMember table
    # Each group has exactly 2 members (order=1 and order=2)
    logger.info("Loading provider asset group members...")
    members_data = pd.read_sql(
        select(
            models.ProviderAssetGroupMember.provider_asset_group_id,
            models.ProviderAssetGroupMember.order,
            models.ProviderAssetGroupMember.provider_id,
            models.ProviderAssetGroupMember.from_asset_id,
            models.ProviderAssetGroupMember.to_asset_id,
        ).where(
            models.ProviderAssetGroupMember.provider_asset_group_id.in_(
                provider_asset_group_ids
            )
        ),
        engine,
    )
    logger.info(f"Members data loaded: {len(members_data)} rows")

    # Setup Coiled cluster
    logger.info(f"Setting up Coiled cluster with {DEFAULT_N_WORKERS} workers...")
    cluster = coiled.Cluster(
        name="pairs-trading-analysis",
        n_workers=DEFAULT_N_WORKERS,
        region="us-east-1",
        software="ghcr.io/manning-capital/mc-notebooks:main",
        worker_memory="16GB",
        worker_cpu=2,
        shutdown_on_close=True,
    )

    client = cluster.get_client()
    logger.info(f"Cluster ready: {client}")

    try:
        # Broadcast market data to workers
        logger.info("Broadcasting market data to workers...")
        market_data_future = client.scatter(market_data, broadcast=True)
        logger.info("Market data broadcasted successfully")

        # Get pairs trading frame
        logger.info("Creating pairs trading frame...")
        pairs_trading_frame = get_pairs_trading_frame(
            start_naive,
            end_naive,
            provider_asset_group_ids,
            members_data,
            market_data_future,
            DEFAULT_N_WORKERS,
        )

        # Compute cointegration p-values
        logger.info("Computing cointegration p-values...")
        cointegration_p_values = pairs_trading_frame.groupby("provider_asset_group_id")[
            ["close_1", "close_2"]
        ].apply(
            lambda df: pd.Series(
                coint(df["close_1"], df["close_2"])[1], index=["p_value"]
            ),
            meta={"p_value": pd.Series([], dtype=float)},
        )

        with ProgressBar():
            cointegration_p_values_computed = cointegration_p_values.compute()

        logger.info(
            f"Cointegration analysis complete: {len(cointegration_p_values_computed)} pairs analyzed"
        )

        # Filter for cointegrated pairs
        cointegrated_provider_asset_group_ids = cointegration_p_values_computed.loc[
            cointegration_p_values_computed["p_value"] < 0.001
        ].index.tolist()

        logger.info(
            f"Cointegrated provider asset group ids (count: {len(cointegrated_provider_asset_group_ids)}): "
            f"{cointegrated_provider_asset_group_ids}"
        )

        if len(cointegrated_provider_asset_group_ids) == 0:
            logger.info("No cointegrated pairs found. Skipping stats computation.")
            cluster.close(force_shutdown=True)
            return cointegration_p_values_computed, pd.DataFrame()

        # Rebroadcast market data for cointegrated pairs analysis
        logger.info("Re-broadcasting market data for cointegrated pairs...")
        market_data_future = client.scatter(market_data, broadcast=True)

        # Get pairs trading frame for cointegrated pairs
        logger.info("Creating pairs trading frame for cointegrated pairs...")
        cointegrated_pairs_trading_frame = get_pairs_trading_frame(
            start_naive,
            end_naive,
            cointegrated_provider_asset_group_ids,
            members_data,
            market_data_future,
            DEFAULT_N_WORKERS,
        )

        # Compute cointegrated pairs stats
        logger.info("Computing cointegrated pairs statistics...")
        cointegrated_pairs_trading_stats = cointegrated_pairs_trading_frame.groupby(
            "provider_asset_group_id"
        )[["close_1", "close_2"]].apply(
            lambda df: get_cointegrated_stats(df),
            meta={
                "linear_fit_alpha": pd.Series([], dtype=float),
                "linear_fit_beta": pd.Series([], dtype=float),
                "linear_fit_mse": pd.Series([], dtype=float),
                "linear_fit_r_squared": pd.Series([], dtype=float),
                "linear_fit_r_squared_adj": pd.Series([], dtype=float),
                "ou_mu": pd.Series([], dtype=float),
                "ou_theta": pd.Series([], dtype=float),
                "ou_sigma": pd.Series([], dtype=float),
            },
        )

        with ProgressBar():
            cointegrated_pairs_trading_stats_computed = (
                cointegrated_pairs_trading_stats.compute()
            )

        logger.info(
            f"Statistics computed for {len(cointegrated_pairs_trading_stats_computed)} cointegrated pairs"
        )

        # Prepare data for database
        toset = cointegration_p_values_computed.merge(
            cointegrated_pairs_trading_stats_computed, left_index=True, right_index=True
        ).reset_index()
        toset = toset.rename(columns={"p_value": "cointegration_p_value"})
        toset["lookback_window_seconds"] = lookback_days * 24 * 60 * 60
        toset["timestamp"] = end_naive
        toset = toset[
            [
                "timestamp",
                "provider_asset_group_id",
                "lookback_window_seconds",
                "cointegration_p_value",
                "linear_fit_alpha",
                "linear_fit_beta",
                "linear_fit_mse",
                "linear_fit_r_squared",
                "linear_fit_r_squared_adj",
                "ou_mu",
                "ou_theta",
                "ou_sigma",
            ]
        ]

        # Write results to ProviderAssetGroupAttribute table
        # This table stores the computed cointegration statistics for each group
        logger.info("Writing results to ProviderAssetGroupAttribute table...")
        set_data(
            engine,
            models.ProviderAssetGroupAttribute.__tablename__,
            toset,
            operation_type="upsert",
        )
        logger.info("Results written to database successfully")
    finally:
        # Cleanup
        logger.info("Closing cluster...")
        cluster.close(force_shutdown=True)
        logger.info("Analysis complete!")
