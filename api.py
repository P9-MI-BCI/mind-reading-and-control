import sqlite3
import pandas as pd
from definitions import DB_PATH


def truncate_table(table_name: str):
    return str(f"""DELETE FROM {table_name}""")


def table_exist(table_name: str):
    return str(f"""SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{table_name}'""")


def sql_create_windows_table():
    return str(
    """create table Windows (
        id INTEGER not null
            constraint Windows_pk
                primary key autoincrement,
    '0' REAL,
    '1' REAL,
    '2' REAL,
    '3' REAL,
    '4' REAL,
    '5' REAL,
    '6' REAL,
    '7' REAL,
    '8' REAL
    );

    create unique index
    Windows_id_uindex on Windows(id);
    """)


def sql_create_dataset_table(dataset_name: str):
    return str(
        f"""CREATE TABLE IF NOT EXISTS {dataset_name} (
              id integer PRIMARY KEY,
              file_name text NOT NULL,
              file_blob text NOT NULL
            );"""
    )


def insert_into_database(file_path_name, file_blob):
  try:
    conn = sqlite3.connect('app.db')
    print("[INFO] : Successful connection!")
    cur = conn.cursor()
    sql_insert_file_query = """INSERT INTO uploads(file_name, file_blob)
      VALUES(?, ?)"""
    cur = conn.cursor()
    cur.execute(sql_insert_file_query, (file_path_name, file_blob, ))
    conn.commit()
    print("[INFO] : The blob for ", file_path_name, " is in the database.")
    last_updated_entry = cur.lastrowid
    return last_updated_entry
  except Error as e:
    print(e)
  finally:
    if conn:
      conn.close()
    else:
      error = "Oh shucks, something is wrong here."


def get_window_data(channel):
    con = sqlite3.connect(DB_PATH)
    statement = f"SELECT `{channel}` FROM Windows ORDER BY id DESC LIMIT 2400"
    df = pd.read_sql_query(statement, con)
    return df


def get_whole_data(channel):
    con = sqlite3.connect(DB_PATH)
    statement = f"SELECT `{channel}` FROM Windows ORDER BY id"
    df = pd.read_sql_query(statement, con)
    return df
