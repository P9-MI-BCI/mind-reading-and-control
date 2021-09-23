import pandas as pd
import datetime


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
