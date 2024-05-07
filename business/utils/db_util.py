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


def create_table(table_name, columns):
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

def insert_data_to_table(table_name, data):
    conn = _create_connection()
    cur = conn.cursor()

    columns = data.columns.tolist()
    insert_query = f"""
        INSERT INTO {table_name} ({", ".join(columns)}) VALUES ({", ".join(["%s"] * len(columns))});
    """

    for index, row in data.iterrows():
        cur.execute(insert_query, row)

    conn.commit()

    cur.close()
    conn.close()

def select_from_table(table_name, where_condition):
        conn = _create_connection()
        cursor = conn.cursor()

        select_query = f"""
        SELECT * 
        FROM {table_name}
        WHERE {where_condition}
        """

        cursor.execute(select_query)

        data_rows = [row for row in cursor]
        col_names = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data_rows, columns=col_names)
        conn.close()
        return df
    
    
def update_data_in_db(self, table_name, set_values, where_condition):
        conn = _create_connection()
        cursor = conn.cursor()
        update_query = f"""
        UPDATE {table_name}
        SET 
        WHERE {where_condition}
        """
        self.__update_data(conn, table_name, where_condition=where_condition)

        conn.close()

def delete_table(table_name):
    conn = _create_connection()
    cursor = conn.cursor()

    delete_query = f"""
    DROP TABLE {table_name}
    """

    cursor.execute(delete_query)
    conn.commit()
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
