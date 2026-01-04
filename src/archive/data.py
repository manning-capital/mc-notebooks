import numpy as np
import pandas as pd


def freq_to_window(base_freq: str, window_freq: str) -> int:
    return int(
        np.floor(
            int(pd.Timedelta(window_freq).total_seconds() / 60)
            / int(pd.Timedelta(base_freq).total_seconds() / 60)
        )
    )


def align_series(
    df_1: pd.DataFrame, df_2: pd.DataFrame, freq: str | int = "1min"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Check for required columns.
    if "timestamp" not in df_1.columns or "timestamp" not in df_2.columns:
        raise ValueError("Timestamp column is required.")
    if "close" not in df_1.columns or "close" not in df_2.columns:
        raise ValueError("Close column is required.")

    # If either series is empty, return the empty series.
    if df_1.empty:
        return df_1, df_1
    if df_2.empty:
        return df_2, df_2

    # Get the start and end dates of the series.
    start = max(
        df_1["timestamp"].min(),
        df_2["timestamp"].min(),
    )
    end = min(
        df_1["timestamp"].max(),
        df_2["timestamp"].max(),
    )

    # Forward fill the series and join to a time-frame for missing values.
    time_df = pd.DataFrame(
        {"timestamp": pd.date_range(start=start, end=end, freq=freq)}
    )
    df_1 = df_1.sort_values(by="timestamp").reset_index(drop=True)
    df_2 = df_2.sort_values(by="timestamp").reset_index(drop=True)
    df_1 = pd.merge_asof(time_df, df_1, on="timestamp", direction="backward")
    df_2 = pd.merge_asof(time_df, df_2, on="timestamp", direction="backward")
    df_1 = df_1.ffill()
    df_2 = df_2.ffill()

    # Drop rows with missing values.
    df_1 = df_1.dropna()
    df_2 = df_2.dropna()

    return df_1, df_2
