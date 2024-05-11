from presentation.power_bi import open_power_bi_dashboard
from business.utils.db_util import insert_data_to_table, select_from_table, update_data_in_db, delete_table,delete_data_from_table
from data.datasource.tv_data_feed_api import get_data
from tvDatafeed import TvDatafeed, Interval
from business.etl.tv_data_hourly_etl import run_etl as hourly_etl
from business.etl.tv_data_daily_etl import run_etl as daily_etl
from business.etl.tv_data_weekly_etl import run_etl as weekly_etl
import pandas as pd
from business.model.neural_network import NeuralNetwork


def add_new_pair(pair, exchange):
    hourly_data = get_data(
        symbol=pair, exchange=exchange, interval=Interval.in_1_hour, n_bars=100000
    )
    daily_data = get_data(symbol=pair, exchange=exchange, interval=Interval.in_daily, n_bars=100000)
    weekly_data = get_data(symbol=pair, exchange=exchange, interval=Interval.in_weekly, n_bars=100000)
    data = {
        "hourly": hourly_data,
        "daily": daily_data,
        "weekly": weekly_data,
    }
    hourly_etl(hourly_data, pair)
    daily_etl(daily_data, pair)
    weekly_etl(weekly_data, pair)
    insert_data_to_table("symbols", pd.DataFrame(data=[[pair, exchange]], columns=["pair", "exchange"]))

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
        
    delete_data_from_table("symbols", f"pair = {pair}")
    return

def refresh_all_data():
    symbols = select_from_table("symbols", "TRUE")
    for index, symbol_item in symbols.iterrows():
        hourly_data = get_data(
            symbol=symbol_item.get('pair'), exchange=symbol_item.get('exchange'), interval=Interval.in_1_hour, n_bars=100000
        )
        daily_data = get_data(symbol=symbol_item['pair'], exchange=symbol_item['exchange'], interval=Interval.in_daily, n_bars=100000)
        weekly_data = get_data(symbol=symbol_item['pair'], exchange=symbol_item['exchange'], interval=Interval.in_weekly, n_bars=100000)
        hourly_etl(hourly_data, symbol_item['pair'])
        daily_etl(daily_data, symbol_item['pair'])
        weekly_etl(weekly_data, symbol_item['pair'])

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

def rerun_models(pair):
    df = select_from_table(f"{pair}_hourly", "TRUE")
    nn = NeuralNetwork(df)
    nn.fit_model()
    
    