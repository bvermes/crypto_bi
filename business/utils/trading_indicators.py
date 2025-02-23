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


def calculate_fibonacci_from_swings(df, high_col="high", low_col="low", window=5):
    """Calculate Fibonacci retracement levels and store them in the DataFrame."""
    df = find_swings(df, high_col, low_col, window)

    recent_high_idx = df[df["swing_high"]].index.max()
    recent_low_idx = df[df["swing_low"]].index.max()

    if pd.isna(recent_high_idx) or pd.isna(recent_low_idx):
        print("Not enough swing points for Fibonacci calculation.")
        return df

    # Determine trend direction
    if recent_high_idx > recent_low_idx:
        swing_high = df.loc[recent_high_idx, high_col]
        swing_low = df.loc[recent_low_idx, low_col]
        trend = "Uptrend"
    else:
        swing_high = df.loc[recent_high_idx, high_col]
        swing_low = df.loc[recent_low_idx, low_col]
        trend = "Downtrend"

    price_range = swing_high - swing_low

    # Calculate Fibonacci levels and store them as columns
    fib_levels = {
        "fib_100": swing_high,
        "fib_78_6": swing_high - 0.786 * price_range,
        "fib_61_8": swing_high - 0.618 * price_range,
        "fib_50": swing_high - 0.5 * price_range,
        "fib_38_2": swing_high - 0.382 * price_range,
        "fib_23_6": swing_high - 0.236 * price_range,
        "fib_0": swing_low,
        "swing_high_val": swing_high,
        "swing_low_val": swing_low,
        "trend": trend,
    }

    # Add Fibonacci levels to the DataFrame for the range between swings
    min_idx = min(recent_high_idx, recent_low_idx)
    max_idx = max(recent_high_idx, recent_low_idx)

    for level, value in fib_levels.items():
        df[level] = np.nan
        df.loc[min_idx:max_idx, level] = value

    return df
