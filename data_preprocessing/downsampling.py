def downsample(datasets, config):
    for dataset in datasets:
        dataset.data = dataset.data.iloc[::config.downsample_rate].reset_index(drop=True)
        dataset.filtered_data = dataset.filtered_data.iloc[::config.downsample_rate].reset_index(drop=True)

        for onsets in dataset.onsets_index:
            onsets[0] = onsets[0] // config.downsample_rate
            onsets[1] = onsets[1] // config.downsample_rate
        dataset.sample_rate = dataset.sample_rate // config.downsample_rate
