import pandas as pd
import datetime
from data_preprocessing.trigger_points import is_triggered


# Used for time convertion from MATLAB dates to python datetime
def convert_mat_date_to_python_date(date_lst):
    p_date_lst = []

    for date in date_lst:
        temp = []
        for num in date[:-1]:
            temp.append(int(num))

        time_str = str(date[-1]).split('.')
        temp.append(int(time_str[0]))
        temp.append(int(time_str[1][:3]) * 1000)

        timestamp = datetime.datetime(*temp)
        p_date_lst.append(timestamp)

    return pd.DataFrame(columns=['Date'], data=p_date_lst)


def convert_freq_to_datetime(freq, hz):
    time_in_seconds = (freq / hz)
    time_str = str(time_in_seconds).split('.')

    return datetime.timedelta(seconds=int(time_str[0]), microseconds=int(time_str[1][:3]) * 1000)


def aggregate_data(device_data_pd, freq_size, is_triggered_table):
    list_of_dataframes = []
    counter = 0

    for i in range(0, device_data_pd.shape[0], freq_size):
        list_of_dataframes.append(
            [device_data_pd.iloc[i:i + freq_size], is_triggered(i + freq_size / 2, is_triggered_table)])

        if i >= 40000:
            break

    return list_of_dataframes
