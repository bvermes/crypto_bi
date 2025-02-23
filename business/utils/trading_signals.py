def very_low_price(symbol, df, price_threshold):
    last_close = df.iloc[-1]["close"]
    if last_close < price_threshold:
        return (True, f"{symbol} very low price: {last_close}")
    else:
        return (False, f"{symbol} price is not low: {last_close}")


def very_high_price(symbol, df, price_threshold):
    last_close = df.iloc[-1]["close"]
    if last_close > price_threshold:
        return (True, f"{symbol} very high price: {last_close}")
    else:
        return (False, f"{symbol} price is not high: {last_close}")


def trading_strategy_1():
    pass
