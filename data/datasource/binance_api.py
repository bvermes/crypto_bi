import requests
from datetime import datetime, timedelta


def get_binance_klines(symbol, interval, start_time=None, end_time=None, limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    if start_time:
        params["startTime"] = start_time

    if end_time:
        params["endTime"] = end_time
    response = requests.get(url, params=params)
    data = response.json()
    return data
