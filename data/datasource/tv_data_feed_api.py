from tvDatafeed import TvDatafeed, Interval
from dotenv import load_dotenv
import os
load_dotenv()
username = os.environ.get('TV_USERNAME')
password = os.environ.get('TV_PASSWORD')
tv = TvDatafeed(username, password)
#

""" 
    in_1_minute = "1"
    in_3_minute = "3"
    in_5_minute = "5"
    in_15_minute = "15"
    in_30_minute = "30"
    in_45_minute = "45"
    in_1_hour = "1H"
    in_2_hour = "2H"
    in_3_hour = "3H"
    in_4_hour = "4H"
    in_daily = "1D"
    in_weekly = "1W"
    in_monthly = "1M" 
    """
    
def get_data(symbol, exchange, interval, n_bars):
    nifty_index_data = tv.get_hist(
        symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars
    )
    return nifty_index_data


if __name__ == "__main__":
    print("m")
    get_data(
        symbol="BTCUSDT", exchange="BINANCE", interval=Interval.in_1_hour, n_bars=100000
    )
    print("m")
