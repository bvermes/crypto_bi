import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os
import csv

load_dotenv()


def _create_connection():
    return psycopg2.connect(
        host="localhost",
        database="postgres",
        user=os.environ.get("POSTGRES_USER"),
        password=os.environ.get("POSTGRES_PASSWORD"),
        port=5432,
    )


def create_table_if_not_exists(table_name, columns):
    conn = _create_connection()
    cur = conn.cursor()
    create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {", ".join(columns)}
        );
    """

    cur.execute(create_table_query)
    conn.commit()

    cur.close()
    conn.close()


def etl_load_dataframe_to_database(df, table_name):
    conn = _create_connection()
    cur = conn.cursor()

    columns = df.columns.tolist()
    insert_query = f"""
        INSERT INTO {table_name} ({", ".join(columns)}) VALUES ({", ".join(["%s"] * len(columns))});
    """

    for index, row in df.iterrows():
        cur.execute(insert_query, row)

    conn.commit()

    cur.close()
    conn.close()


# conn = psycopg2.connect(
#    host="localhost",
#    database="postgres",
#    user=os.environ.get("POSTGRES_USER"),
#    password=os.environ.get("POSTGRES_PASSWORD"),
#    port=5432,
# )
# cur = conn.cursor()
# cur.execute(
#    """Create table if not exists stock_data(
#       id INT PRIMARY KEY,
#       name VARCHAR(255),
#       symbol VARCHAR(255),
#       exchange VARCHAR(255),
#       open FLOAT,
#       high FLOAT,
#       low FLOAT,
#       close FLOAT,
#       volume INT,
#       datetime TIMESTAMP
#   )
#   """
# )
# conn.commit()
# cur.close()
# conn.close()
