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
    load_dataframe_to_database,
    select_from_table,
    check_table_existance
)


import pandas as pd


def fill_null_with_average(df):
    for col in df.columns:
        # Check if there are null values in the column
        if df[col].isnull().any():
            # Find null values
            null_indices = df[col][df[col].isnull()].index

            # Iterate over null indices
            for idx in null_indices:
                # Calculate average of previous and next row values
                prev_value = df.at[idx - 1, col]
                next_value = df.at[idx + 1, col]
                avg_value = (prev_value + next_value) / 2

                # Fill null value with average
                df.at[idx, col] = avg_value

    return df

def run_etl():
    table_name = "unemployment_rates"
    columns = ["date DATE null", "overall_rate numeric(20, 2) NOT null"]
    csv_file = os.path.join(
        grandparent_dir, "data", "datasource", "raw", "df_unemployment_rates.csv"
    )
    df = pd.read_csv(csv_file)
    df = df[["date", "overall_rate"]]

    df = fill_null_with_average(df)

    if (check_table_existance(table_name=table_name) == False):
            create_table(table_name, columns)
    load_dataframe_to_database(df, table_name)
    
if __name__ == "__main__":
    run_etl()
    
