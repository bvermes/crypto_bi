import psycopg2
import csv
import os
import pandas as pd
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
grandparent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.append(grandparent_dir)


from business.utils.db_util import (
    create_table,
    etl_load_dataframe_to_database,
    select_from_table
)


import pandas as pd


def _fill_null_with_average(df):
    for col in df.columns:
        if df[col].isnull().any():
            null_indices = df[col][df[col].isnull()].index
            for idx in null_indices:
                prev_value = df.at[idx - 1, col]
                next_value = df.at[idx + 1, col]
                avg_value = (prev_value + next_value) / 2
                df.at[idx, col] = avg_value

    return df

def _load_dataframe_to_database(df, table_name):
    db_df = select_from_table(table_name=table_name, where_clause="TRUE")
    merged = df.merge(db_df, on='datetime', how='outer', indicator=True)
    rows_only_in_df = merged[merged['_merge'] == 'left_only'].drop(columns='_merge')
    etl_load_dataframe_to_database(rows_only_in_df, table_name)
    

def run_etl(hourly_df, symbol):
    table_name = "{symbol}_hourly".format(symbol=symbol)
    columns = ["datetime TIMESTAMP null",
               "open numeric(20, 2) NOT null",
               "high numeric(20, 2) NOT null",
               "low numeric(20, 2) NOT null", 
               "close numeric(20, 2) NOT null",
               "volume numeric(20, 2) NOT null"]
    df = hourly_df
    df = df[["datetime", "open", "high", "low", "close", "volume"]]

    df = _fill_null_with_average(df)
    if (select_from_table(table_name=table_name, where_clause="TRUE").empty):
        create_table(table_name, columns)
        
    _load_dataframe_to_database(df, table_name)
    
    
