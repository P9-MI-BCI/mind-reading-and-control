import pandas as pd
import datetime


# Converts an input freq to a timedelta
def convert_freq_to_datetime(input_freq: int, sample_freq: int) -> datetime.timedelta:
    time_in_seconds = (input_freq / sample_freq)
    time_str = str(time_in_seconds).split('.')

    # there might be unresolved edge-cases
    while len(time_str[1]) < 6:
        time_str[1] += '0'

    return datetime.timedelta(seconds=int(time_str[0]), microseconds=int(time_str[1][:6]))
