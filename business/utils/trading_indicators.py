import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


def calculate_rsi_with_ma(
    prices, rsi_period=14, ma_type="SMA", ma_length=None, bb=False, bb_stddev=2.0
):
    """
    Calculates the RSI and applies a smoothing moving average (MA) on the RSI.

    Args:
        prices (pd.Series): Series of closing prices.
        rsi_period (int): Number of periods for the RSI calculation (default is 14).
        ma_type (str): Type of moving average to apply. Supported values are:
            "None", "SMA", "EMA", "SMMA" (or "RMA"), "WMA".
            (VWMA is not supported because RSI has no volume data.)
        ma_length (int): The length/period for the moving average. If None, defaults to rsi_period.
        bb (bool): If True and if ma_type is "SMA", Bollinger Bands will be computed.
        bb_stddev (float): The number of standard deviations for Bollinger Bands.

    Returns:
        If bb is False:
            tuple (rsi, rsi_ma)
        If bb is True (and ma_type is "SMA"):
            tuple (rsi, rsi_ma, upper_band, lower_band)
    """
    # Calculate the RSI from prices
    rsi = calculate_rsi(prices, period=rsi_period)

    # Use the provided ma_length or default to rsi_period
    if ma_length is None:
        ma_length = rsi_period

    # Compute the smoothing MA based on the selected type (case-insensitive)
    ma_type = ma_type.upper()
    if ma_type == "NONE":
        rsi_ma = None
    elif ma_type == "SMA":
        rsi_ma = rsi.rolling(window=ma_length, min_periods=ma_length).mean()
    elif ma_type == "EMA":
        rsi_ma = rsi.ewm(span=ma_length, adjust=False).mean()
    elif ma_type in ["SMMA", "RMA"]:
        # SMMA (or RMA) can be approximated by an EWMA with alpha = 1/ma_length
        rsi_ma = rsi.ewm(alpha=1 / ma_length, adjust=False).mean()
    elif ma_type == "WMA":
        # Weighted Moving Average: weights increase linearly (1,2,...,ma_length)
        weights = np.arange(1, ma_length + 1)
        rsi_ma = rsi.rolling(window=ma_length, min_periods=ma_length).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    elif ma_type == "VWMA":
        raise ValueError(
            "VWMA is not supported for RSI because there is no volume data."
        )
    else:
        raise ValueError(f"Unsupported MA type: {ma_type}")

    # If Bollinger Bands are requested (only when using SMA), calculate them
    if bb and ma_type == "SMA":
        rsi_std = rsi.rolling(window=ma_length, min_periods=ma_length).std()
        upper_band = rsi_ma + bb_stddev * rsi_std
        lower_band = rsi_ma - bb_stddev * rsi_std
        return rsi, rsi_ma, upper_band, lower_band
    else:
        return rsi, rsi_ma


def calculate_rsi(prices, period=14):
    """
    Calculates the Relative Strength Index (RSI) for a given list of prices.

    Args:
        prices: A list of closing prices.
        period: The number of periods for the moving average (default: 14).

    Returns:
        A pandas Series containing the RSI values for each price.
    """

    delta = prices.diff()
    delta = delta.dropna()  # Remove NaN from the first difference
    up, down = delta.clip(lower=0), delta.clip(
        upper=0, lower=None
    )  # Separate gains and losses

    ema_up = up.ewm(
        alpha=1 / period, min_periods=period
    ).mean()  # Exponential Moving Average for gains
    ema_down = (
        down.abs().ewm(alpha=1 / period, min_periods=period).mean()
    )  # EMA for absolute losses

    rs = ema_up / ema_down  # Average gain / Average loss
    rsi = 100 - 100 / (1 + rs)  # Calculate RSI

    return rsi


def moving_average(series, window, ma_type="SMA"):
    """
    Computes a moving average for a given pandas Series.

    Args:
        series (pd.Series): Data to smooth.
        window (int): The number of periods.
        ma_type (str): Type of moving average. Supported: "SMA" or "EMA".

    Returns:
        pd.Series: The moving average.
    """
    if ma_type.upper() == "SMA":
        return series.rolling(window=window, min_periods=window).mean()
    elif ma_type.upper() == "EMA":
        return series.ewm(span=window, adjust=False).mean()
    else:
        raise ValueError("Unsupported moving average type. Use 'SMA' or 'EMA'.")


def calculate_macd(series, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculates the MACD indicator components.

    Args:
        series (pd.Series): Price series (typically 'close').
        fast_period (int): Fast EMA period.
        slow_period (int): Slow EMA period.
        signal_period (int): Signal line EMA period.

    Returns:
        tuple of pd.Series: (macd_line, signal_line, macd_histogram)
    """
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def calculate_bollinger_bands(series, window=20, num_std=2):
    """
    Calculates Bollinger Bands for a given price series.

    Args:
        series (pd.Series): Price series (typically 'close').
        window (int): Window period for the SMA.
        num_std (float): Number of standard deviations for the bands.

    Returns:
        tuple of pd.Series: (middle_band, upper_band, lower_band)
    """
    middle_band = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    upper_band = middle_band + num_std * std
    lower_band = middle_band - num_std * std
    return middle_band, upper_band, lower_band


def calculate_stochastic(df, k_period=14, d_period=3):
    """
    Calculates the Stochastic Oscillator.

    Args:
        df (pd.DataFrame): DataFrame with columns 'high', 'low', and 'close'.
        k_period (int): Look-back period for %K.
        d_period (int): Smoothing period for %D.

    Returns:
        tuple of pd.Series: (%K, %D)
    """
    # Lowest low and highest high over the look-back period
    low_min = df["low"].rolling(window=k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(window=k_period, min_periods=k_period).max()

    # %K calculation: scaled between 0 and 100
    stoch_k = 100 * ((df["close"] - low_min) / (high_max - low_min))
    stoch_d = stoch_k.rolling(window=d_period, min_periods=d_period).mean()
    return stoch_k, stoch_d


def calculate_fibonacci_retracement(df, high_col="high", low_col="low"):
    """
    Calculate Fibonacci retracement levels based on high and low prices.

    Parameters:
    - df (pd.DataFrame): DataFrame containing price data with 'high' and 'low' columns.
    - high_col (str): Column name for high prices.
    - low_col (str): Column name for low prices.

    Returns:
    - dict: Fibonacci retracement levels.
    """
    # Find the highest high and lowest low
    highest_high = df[high_col].max()
    lowest_low = df[low_col].min()

    price_range = highest_high - lowest_low

    # Fibonacci retracement levels
    levels = {
        "0.0%": highest_high,
        "23.6%": highest_high - 0.236 * price_range,
        "38.2%": highest_high - 0.382 * price_range,
        "50.0%": highest_high - 0.5 * price_range,
        "61.8%": highest_high - 0.618 * price_range,
        "78.6%": highest_high - 0.786 * price_range,
        "100.0%": lowest_low,
        "Highest High": highest_high,
        "Lowest Low": lowest_low,
    }

    return levels


def find_swings(df, high_col="high", low_col="low", window=5):
    """
    Identify swing highs and lows using a rolling window approach.

    Parameters:
    - df (pd.DataFrame): DataFrame with high and low columns.
    - window (int): Number of candles to consider for a swing.

    Returns:
    - pd.DataFrame: Original DataFrame with 'swing_high' and 'swing_low' markers.
    """
    df = df.copy()

    # Swing High: Highest within window, with peaks
    df["swing_high"] = (
        (df[high_col] == df[high_col].rolling(window=window, center=True).max())
        & (df[high_col].shift(1) < df[high_col])
        & (df[high_col].shift(-1) < df[high_col])
    )

    # Swing Low: Lowest within window, with troughs
    df["swing_low"] = (
        (df[low_col] == df[low_col].rolling(window=window, center=True).min())
        & (df[low_col].shift(1) > df[low_col])
        & (df[low_col].shift(-1) > df[low_col])
    )

    return df


def calculate_fibonacci_from_swings(df, high_col="high", low_col="low", suffix=""):
    """Calculate Fibonacci retracement levels iteratively across the entire DataFrame.

    This function iterates row by row, ensuring that Fibonacci levels update dynamically.

    Parameters:
    - df: DataFrame containing price data.
    - high_col: Column name for high prices.
    - low_col: Column name for low prices.
    - suffix: A string to append to Fibonacci level column names (for multi-timeframe tracking).

    Returns:
    - DataFrame with Fibonacci levels assigned per row.
    """

    def find_local_low(df, current_idx, lookback=40, check_range=10):
        """Find the lowest low within a lookback period and return the surrounding confirmed local low."""
        start_idx = max(0, current_idx - lookback)

        # Ensure the slice is not empty
        if start_idx >= current_idx:
            return None  # No valid range to search in

        low_values = df[low_col][start_idx:current_idx]

        if low_values.empty:
            return None  # No valid values to search in

        low_idx = low_values.idxmin()

        # Check the surrounding range to see if it's truly the lowest nearby
        check_start = max(0, low_idx - check_range)
        check_end = min(len(df), low_idx + check_range)

        if df[low_col].loc[low_idx] == df[low_col].loc[check_start:check_end].min():
            return low_idx
        else:
            # If it's not the lowest, move back slightly and search again
            return find_local_low(df, low_idx, lookback=10, check_range=check_range)

    def find_local_high(df, low_idx, current_idx, prev_high_idx=None):
        """Find the highest high between low_idx and current_idx.
        If the high is the last row, use the previous valid local high.
        """
        # Ensure a valid range exists
        if low_idx >= current_idx:
            return prev_high_idx  # No valid range, return previous high if available

        high_values = df[high_col][low_idx:current_idx]

        if high_values.empty:
            return prev_high_idx  # No valid values, return previous high

        high_idx = high_values.idxmax()

        # If the detected high is the last row, revert to previous valid high
        if high_idx == df.index[-1]:
            return (
                prev_high_idx if prev_high_idx is not None else high_idx
            )  # Use previous high if available

        return high_idx

    fib_columns = {
        "fib_261_8": None,
        "fib_161_8": None,
        "fib_100": None,
        "fib_78_6": None,
        "fib_61_8": None,
        "fib_50": None,
        "fib_38_2": None,
        "fib_23_6": None,
        "fib_0": None,
        "swing_high_val": None,
        "swing_low_val": None,
        "trend": None,
    }

    # Add suffix to Fibonacci level column names
    fib_columns = {key + suffix: None for key in fib_columns}
    df = df.assign(**{col: np.nan for col in fib_columns.keys()})

    prev_fib_levels = None  # Store the last known Fibonacci levels

    for current_idx in df.index:
        # Find local low and high
        recent_low_idx = find_local_low(df, current_idx, lookback=40, check_range=10)
        if recent_low_idx is None:
            continue  # Skip if no valid low is found

        recent_high_idx = find_local_high(df, recent_low_idx, current_idx)

        # If high is the last row, use the previous valid Fibonacci levels
        if recent_high_idx == df.index[-1]:
            if prev_fib_levels:
                df.loc[current_idx, list(prev_fib_levels.keys())] = list(
                    prev_fib_levels.values()
                )
            continue

        # Check for deeper lows and adjust
        for extended_lookback, threshold in [(40, 0.3), (80, 1)]:
            deeper_low_idx = find_local_low(
                df, recent_low_idx, lookback=extended_lookback, check_range=10
            )
            if (
                deeper_low_idx
                and df.loc[recent_low_idx, low_col] - df.loc[deeper_low_idx, low_col]
                > (df.loc[recent_high_idx, high_col] - df.loc[recent_low_idx, low_col])
                * threshold
            ):
                recent_low_idx = deeper_low_idx
                recent_high_idx = find_local_high(df, recent_low_idx, current_idx)

        swing_low = df.loc[recent_low_idx, low_col]
        swing_high = df.loc[recent_high_idx, high_col]
        price_range = swing_high - swing_low

        # If no valid retracement range, skip
        if price_range == 0:
            continue

        # Calculate Fibonacci levels
        is_uptrend = recent_high_idx > recent_low_idx

        if is_uptrend:
            fib_levels = {
                f"fib_261_8{suffix}": swing_low + 2.618 * price_range,
                f"fib_161_8{suffix}": swing_low + 1.618 * price_range,
                f"fib_100{suffix}": swing_high,
                f"fib_78_6{suffix}": swing_low + 0.786 * price_range,
                f"fib_61_8{suffix}": swing_low + 0.618 * price_range,
                f"fib_50{suffix}": swing_low + 0.5 * price_range,
                f"fib_38_2{suffix}": swing_low + 0.382 * price_range,
                f"fib_23_6{suffix}": swing_low + 0.236 * price_range,
                f"fib_0{suffix}": swing_low,
            }
        else:  # Downtrend case
            fib_levels = {
                f"fib_261_8{suffix}": swing_high - 2.618 * price_range,
                f"fib_161_8{suffix}": swing_high - 1.618 * price_range,
                f"fib_100{suffix}": swing_low,
                f"fib_78_6{suffix}": swing_high - 0.786 * price_range,
                f"fib_61_8{suffix}": swing_high - 0.618 * price_range,
                f"fib_50{suffix}": swing_high - 0.5 * price_range,
                f"fib_38_2{suffix}": swing_high - 0.382 * price_range,
                f"fib_23_6{suffix}": swing_high - 0.236 * price_range,
                f"fib_0{suffix}": swing_high,
            }

        # Add metadata about swings
        fib_levels[f"swing_high_val{suffix}"] = swing_high
        fib_levels[f"swing_low_val{suffix}"] = swing_low
        fib_levels[f"trend{suffix}"] = "Uptrend" if is_uptrend else "Downtrend"

        # Store the latest valid Fibonacci levels for future reference
        prev_fib_levels = fib_levels.copy()

        # Assign values to each row up to the current row
        df.loc[recent_low_idx:current_idx, list(fib_levels.keys())] = list(
            fib_levels.values()
        )
    return df
