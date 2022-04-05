# TODO: Deprecated

def downsample(datasets, config):
    for dataset in datasets:
        dataset.data = dataset.data.iloc[::config.downsample_rate].reset_index(drop=True)
        dataset.filtered_data = dataset.filtered_data.iloc[::config.downsample_rate].reset_index(drop=True)

        for onsets in dataset.clusters:
            onsets.data = onsets.data // config.downsample_rate
            onsets.peak = onsets.peak // config.downsample_rate
            onsets.create_info()
        dataset.sample_rate = dataset.sample_rate // config.downsample_rate
