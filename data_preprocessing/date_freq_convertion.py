import pandas as pd
import datetime


# Used for time convertion from MATLAB dates to python datetime
def convert_mat_date_to_python_date(date_arr: [list]) -> pd.DataFrame:
    p_date_lst = []

    for date in date_arr:
        temp = []
        for num in date[:-1]:
            temp.append(int(num))

        time_str = str(date[-1]).split('.')
        temp.append(int(time_str[0]))
        temp.append(int(time_str[1][:3]) * 1000)

        timestamp = datetime.datetime(*temp)
        p_date_lst.append(timestamp)

    return pd.DataFrame(columns=['Date'], data=p_date_lst)


# converts an input freq to a timedelta
def convert_freq_to_datetime(input_freq: int, sample_freq: int) -> datetime.timedelta:
    time_in_seconds = (input_freq / sample_freq)
    time_str = str(time_in_seconds).split('.')

    # there might be unresolved edge-cases
    while len(time_str[1]) < 6:
        time_str[1] += '0'

    return datetime.timedelta(seconds=int(time_str[0]), microseconds=int(time_str[1][:6]))
