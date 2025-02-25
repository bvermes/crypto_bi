import business.utils.trading_indicators as ti
import pandas as pd


def very_low_price(symbol, df, last_valid_row):
    """Check if the price is very low based on signal, RSI, and price change conditions."""
    last_close = df.iloc[-1]["close"]
    first_close = df.iloc[0]["close"]
    time_diff = (df.iloc[-1]["timestamp"] - df.iloc[0]["timestamp"]).days
    price_change = abs((last_close - first_close) / first_close)

    # Calculate RSI and Moving Average
    rsi, rsi_ma = ti.calculate_rsi_with_ma(df["close"])

    # 1️ Condition: If last_valid_row is not empty, check signal-based threshold
    signal_based_low = False
    if not last_valid_row.empty:
        last_signal = last_valid_row["Notified Signal Value"].values[0]
        signal_based_low = last_close < last_signal * 0.95

    # 2️ Condition: RSI-Based low detection
    rsi_based_low = (rsi.iloc[-1] < 15) or (
        (rsi_ma is not None) and (rsi_ma.iloc[-1] - rsi.iloc[-1] > 30)
    )

    # 3️ Condition: Price change >5% within a week
    time_based_low = (time_diff < 7) and (price_change > 0.05)

    # Final decision
    if signal_based_low or rsi_based_low or time_based_low:
        return True, f"{symbol} very low price: {last_close}", last_close
    else:
        return False, f"{symbol} price is not low: {last_close}", last_close


def very_high_price(symbol, df, last_valid_row):
    """Check if the price is very high based on signal, RSI, and price change conditions."""
    last_close = df.iloc[-1]["close"]
    first_close = df.iloc[0]["close"]
    time_diff = (df.iloc[-1]["timestamp"] - df.iloc[0]["timestamp"]).days
    price_change = abs((last_close - first_close) / first_close)

    # Calculate RSI and Moving Average
    rsi, rsi_ma = ti.calculate_rsi_with_ma(df["close"])

    # 1️ Condition: If last_valid_row is not empty, check signal-based threshold
    signal_based_high = False
    if not last_valid_row.empty:
        last_signal = last_valid_row["Notified Signal Value"].values[0]
        signal_based_high = last_close > last_signal * 1.05

    # 2️ Condition: RSI-Based high detection
    rsi_based_high = (rsi.iloc[-1] > 85) or (
        (rsi_ma is not None) and (rsi.iloc[-1] - rsi_ma.iloc[-1] > 40)
    )

    # 3️ Condition: Price change >5% within a week
    time_based_high = (time_diff < 7) and (price_change > 0.05)

    # Final decision
    if signal_based_high or rsi_based_high or time_based_high:
        return True, f"{symbol} very high price: {last_close}", last_close
    else:
        return False, f"{symbol} price is not high: {last_close}", last_close
