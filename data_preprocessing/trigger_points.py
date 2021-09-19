import pandas as pd
import datetime
from data_preprocessing.date_freq_convertion import convert_mat_date_to_python_date, convert_freq_to_datetime


def covert_trigger_points_to_pd(trigger_point_inp):
    trigger_point_lst = []

    for t_p in trigger_point_inp:
        temp = []
        for x in t_p[1:-1]:
            temp.append(int(x))

        # hacky fix using string split
        time_str = str(t_p[-1]).split('.')
        temp.append(int(time_str[0]))
        temp.append(int(time_str[1][:3])*1000)

        timestamp = datetime.datetime(*temp)
        trigger_point_lst.append([int(t_p[0]), timestamp])

    return pd.DataFrame(columns=['Trigger', 'Date'], data=trigger_point_lst)


def is_triggered(dataset, freq, is_triggered_table):
    sample_rate = dataset['sample_rate'][0][0]
    freq_in_sec = convert_freq_to_datetime(freq, sample_rate)

    for i, row in is_triggered_table.iterrows():
        if freq_in_sec > row['start'] and freq_in_sec < row['end']:
            return 1
    return 0


def trigger_time_table(trigger_points_pd, cue_set):
    time_start = convert_mat_date_to_python_date(cue_set['time_start_device1'])
    is_trigger_time_table = []
    prev_non_trigger_timestamp = 0

    tp_iter = trigger_points_pd.iterrows()
    for index, row in tp_iter:
        if row['Trigger'] == 1:
            _, tp = next(tp_iter)
            time_since_start = row['Date'] - time_start
            end_time = tp['Date'] - time_start

            # print(f'start: {time_since_start}, end: {end_time}')
            is_trigger_time_table.append([time_since_start.iloc[0][0], end_time.iloc[0][0]])

    return pd.DataFrame(columns=['start', 'end'],
                        data=is_trigger_time_table)  # in seconds from time start