from presentation.power_bi import open_power_bi_dashboard
from db_util import insert_data_to_table, select_from_table, update_data_in_db, delete_table
from data.datasource.tv_data_feed_api import get_data
from tvDatafeed import TvDatafeed, Interval
from etl.tv_data_hourly_etl import run_etl as hourly_etl
from etl.tv_data_daily_etl import run_etl as daily_etl
from etl.tv_data_weekly_etl import run_etl as weekly_etl

def add_new_pair(pair, exchange):
    hourly_data = get_data(
        symbol=pair, exchange="BINANCE", interval=Interval.in_1_hour, n_bars=100000
    )
    daily_data = get_data(symbol=pair, exchange="BINANCE", interval=Interval.in_daily, n_bars=100000)
    weekly_data = get_data(symbol=pair, exchange="BINANCE", interval=Interval.in_weekly, n_bars=100000)
    data = {
        "hourly": hourly_data,
        "daily": daily_data,
        "weekly": weekly_data,
    }
    return data

def remove_pair(pair, exchange):
    try:
        delete_table(f"{pair}_hourly")
    except:
        print(f"Table {pair}_hourly doesn't exist. Maybe it was already deleted or wasn't added yet.")

    try:
        delete_table(f"{pair}_daily")
    except:
        print(f"Table {pair}_daily doesn't exist. Maybe it was already deleted or wasn't added yet.")

    try:
        delete_table(f"{pair}_weekly")
    except:
        print(f"Table {pair}_weekly doesn't exist. Maybe it was already deleted or wasn't added yet.")
    return

def refresh_all_data():
    symbols = select_from_table("symbols", "TRUE")
    for symbol in symbols:
        hourly_data = get_data(
            symbol=symbol, exchange="BINANCE", interval=Interval.in_1_hour, n_bars=100000
        )
        daily_data = get_data(symbol=symbol, exchange="BINANCE", interval=Interval.in_daily, n_bars=100000)
        weekly_data = get_data(symbol=symbol, exchange="BINANCE", interval=Interval.in_weekly, n_bars=100000)
        hourly_etl.run_etl(hourly_data, symbol)
        daily_etl.run_etl(daily_data, symbol)
        weekly_etl.run_etl(weekly_data, symbol)

def open_powerbi_dashboard():
    open_power_bi_dashboard()


def new_unemployment_rate(month, rate):
    rows = select_from_table("unemployment_rates", f"date = {month}")
    if (len(rows) > 0):
        print("Data already exists for this month. Updating the rate.")
        try:
            update_data_in_db("unemployment_rates", f"overall_rate = {rate}", f"date = {month}")
        except:
            print("Error updating the data.")
    return

def rerun_models():
    pass   