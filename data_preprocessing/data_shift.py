from classes import Dataset
from data_preprocessing.date_freq_convertion import convert_freq_to_datetime
from utility.logger import get_logger


# when given a frequency - shift all data by that much.
# this method will delete anything prior to the frequency.
# calculate time based on freq from time_start_device1
def shift_data(freq: int, dataset: Dataset) -> Dataset:
    time_shift = convert_freq_to_datetime(freq, dataset.sample_rate)

    dataset.time_cue_on = delete_timestamps(time_shift, dataset.time_start_device1, dataset.time_cue_on)
    dataset.time_cue_off = delete_timestamps(time_shift, dataset.time_start_device1, dataset.time_cue_off)
    dataset.TriggerPoint = delete_timestamps(time_shift, dataset.time_start_device1, dataset.TriggerPoint, is_tp=True)
    dataset.data_device1 = dataset.data_device1.iloc[freq:, :].reset_index(drop=True)
    dataset.time_axis_all_device1 = dataset.time_axis_all_device1.iloc[freq:, :].reset_index(drop=True)
    dataset.time_start_device1 = dataset.time_start_device1 + time_shift

    return dataset


def delete_timestamps(time_shift, time_start_device1, timestamp_arr, is_tp=False):
    delete_rows = []
    if is_tp:
        for i, row in timestamp_arr.iterrows():
            delete_rows.append((row['Date'] < time_start_device1 + time_shift).iloc[0]['Date'])
    else:
        for i, row in timestamp_arr.iterrows():
            delete_rows.append((row < time_start_device1 + time_shift).iloc[0]['Date'])

    for i in range(0, len(delete_rows)):
        if delete_rows[i]:
            timestamp_arr = timestamp_arr.drop(i)
    # checks if more than 1 TP is del. The first one is always del because the timestamp is before time_start_device1
    if any(delete_rows) and is_tp and sum(delete_rows) > 1:
        get_logger().warning("You are deleting TriggerPoints with shift_data. Consider shifting less frequency.")

    return timestamp_arr
