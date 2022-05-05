import copy

from classes.Dataset import Dataset

def downsample(datasets, config):
    # returns a new list of down-sampled datasets in order to differentiate datasets instead of overwriting
    down_sampled_datasets = []
    for dataset in datasets:
        temp_dataset = copy.deepcopy(dataset)
        temp_dataset.data = temp_dataset.data.iloc[::config.downsample_rate].reset_index(drop=True)
        temp_dataset.filtered_data = temp_dataset.filtered_data.iloc[::config.downsample_rate].reset_index(drop=True)
        for onsets in temp_dataset.clusters:
            onsets.data = onsets.data // config.downsample_rate
            onsets.peak = onsets.peak // config.downsample_rate
            onsets.create_info()
        temp_dataset.sample_rate = temp_dataset.sample_rate // config.downsample_rate

        down_sampled_datasets.append(temp_dataset)

    return down_sampled_datasets
