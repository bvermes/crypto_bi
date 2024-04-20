from tvDatafeed import TvDatafeed, Interval


def get_data(symbol, exchange, interval, n_bars):
    tv = TvDatafeed()
    # index
    print("f")
    nifty_index_data = tv.get_hist(
        symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars
    )
    # futures continuous contract
    print(nifty_index_data.tail())
    # crudeoil
    # crudeoil_data = tv.get_hist(symbol='CRUDEOIL',exchange='MCX',interval=Interval.in_1_hour,n_bars=5000,fut_contract=1)
    # downloading data for extended market hours
    # extended_price_data = tv.get_hist(symbol="EICHERMOT",exchange="NSE",interval=Interval.in_1_hour,n_bars=500, extended_session=False)


if __name__ == "__main__":
    print("m")
    get_data(
        symbol="BTCUSDT", exchange="BINANCE", interval=Interval.in_1_hour, n_bars=100000
    )
    print("m")
