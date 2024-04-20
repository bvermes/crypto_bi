from tvDatafeed import TvDatafeed, Interval


def get_data(symbol, exchange, interval, n_bars):
    tv = TvDatafeed()
    # index
    print('f')
    nifty_index_data = tv.get_hist(symbol='NIFTY',exchange='NSE',interval=Interval.in_1_hour,n_bars=10000)
    # futures continuous contract
    nifty_futures_data = tv.get_hist(symbol='NIFTY',exchange='NSE',interval=Interval.in_1_hour,n_bars=1000,fut_contract=1)
    print(nifty_futures_data.head())
    # crudeoil
    #crudeoil_data = tv.get_hist(symbol='CRUDEOIL',exchange='MCX',interval=Interval.in_1_hour,n_bars=5000,fut_contract=1)
    # downloading data for extended market hours
    #extended_price_data = tv.get_hist(symbol="EICHERMOT",exchange="NSE",interval=Interval.in_1_hour,n_bars=500, extended_session=False)


if __name__ == "__main__":
    print('m')
    get_data(symbol = 'NIFTY', exchange = 'NSE', interval=Interval.in_1_hour, n_bars=10000)
    print('m')