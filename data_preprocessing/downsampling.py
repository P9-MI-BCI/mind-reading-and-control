def downsample(datasets, config):
    for dataset in datasets:
        dataset.filtered_data = dataset.filtered_data.iloc[::config.downsample_rate].reset_index(drop=True)

        for onsets in dataset.onsets_index:
            onsets[0] = onsets[0] // config.downsample_rate
            onsets[1] = onsets[1] // config.downsample_rate
        dataset.sample_rate = dataset.sample_rate // config.downsample_rate

        '''
        Downsamples EMG onset indicies in increments of nearest whole integer division of the downsample_rate, for use with true index of EEG signals (no reset_index)
        '''
        # for onsets in dataset.onsets_index:
        #     if (onsets[0] // config.downsample_rate) % config.downsample_rate > config.downsample_rate / 2:
        #         onsets[0] = (onsets[0] // config.downsample_rate) + (config.downsample_rate - (onsets[0] // config.downsample_rate % config.downsample_rate))
        #     else:
        #         onsets[0] = (onsets[0] // config.downsample_rate) - (onsets[0] // config.downsample_rate % config.downsample_rate)
        #     if (onsets[1] // config.downsample_rate) % config.downsample_rate > config.downsample_rate / 2:
        #         onsets[1] = (onsets[1] // config.downsample_rate) + (config.downsample_rate - (onsets[1] // config.downsample_rate % config.downsample_rate))
        #     else:
        #         onsets[1] = (onsets[1] // config.downsample_rate) - (onsets[1] // config.downsample_rate % config.downsample_rate)