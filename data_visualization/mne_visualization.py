import mne.viz
import numpy as np
import pandas as pd

from classes.Dataset import Dataset


def create_info(config):
    # ch_names = ''
    # if pattern == 'diamond':
    #     ch_names = ["F3", "FC1", "FC5", "Cz", "C3", "T7", "CP1", "CP5", "P3"]  # Diamond pattern
    # if pattern == 'line':
    #     ch_names = ['T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8']  # Line pattern
    ch_types = ['eeg'] * len(config.EEG_CHANNELS)

    info = mne.create_info(ch_names=config.EEG_CHANNELS, ch_types=ch_types, sfreq=1200)
    info.set_montage('standard_1020')

    return info


def create_raw_array(data: pd.DataFrame, config):
    info = create_info(config)

    raw_data = mne.io.RawArray(np.transpose(data[config.EEG_CHANNELS]), info)
    raw_proj_data = mne.compute_proj_raw(raw_data, n_eeg=len(config.EEG_CHANNELS))

    # Visualization of raw data
    raw_data.plot(n_channels=len(config.EEG_CHANNELS))  # plot of the signal(s) can be a single or all channels
    raw_data.plot_sensors(ch_type='eeg')  # plot sensor positions
    raw_data.plot_psd(average=False)  # plot spectral density of signals
    mne.viz.plot_projs_topomap(raw_proj_data, colorbar=True, vlim='joint', info=info)

    return raw_data


def create_epochs_array(dataset: Dataset, config, visualize=False):
    info = create_info(config)
    picks = mne.pick_types(info, eeg=True, misc=False)

    events = []
    epochs_data = []
    # TODO: Fix to use new Cluster class
    for onset in dataset.clusters:
        if onset[0] - config.window_padding * dataset.sample_rate < 0:
            continue
        temp = [onset[0], 0, dataset.label]
        events.append(temp)
        epochs_data.append(np.transpose(dataset.filtered_data[config.EEG_CHANNELS].iloc[
                                        onset[0] - config.window_padding * dataset.sample_rate:
                                        onset[0] + config.window_padding * dataset.sample_rate].to_numpy()))

    epochs_data = np.array(epochs_data)

    epochs = mne.EpochsArray(epochs_data, info=info, events=events)
    epoch_proj_data = mne.compute_proj_epochs(epochs, n_eeg=len(config.EEG_CHANNELS))

    # Visualization of epoched data
    if visualize:
        epochs.plot(scalings='auto', show=True, block=True)
        epochs.plot_image()
        mne.viz.plot_projs_topomap(epoch_proj_data, colorbar=True, vlim='joint', info=info)

    return epochs


def create_evoked_array(dataset, config, show_plots=False):
    epoched_data = create_epochs_array(dataset, config, visualize=show_plots)
    info = create_info(config)
    nave = len(epoched_data)
    evoked_data = np.mean(epoched_data, axis=0)

    evoked = mne.EvokedArray(evoked_data, info=info, comment='Arbitrary', nave=nave)
    # evoked_proj_data = mne.compute_proj_evoked(evoked, n_eeg=len(config.EEG_Channels))

    times = np.arange(0, config.window_padding*2, 0.1)
    evoked.plot_topomap(times, ch_type='eeg', time_unit='s', ncols=6, nrows='auto', extrapolate='head')
    evoked.plot_image(picks='eeg')

    # fig, anim = evoked.animate_topomap(
    #     times=times, ch_type='eeg', frame_rate=2, time_unit='s', blit=False)

    return evoked


def visualize_mne(datasets, config):
    for dataset in datasets:
        create_evoked_array(dataset, config, show_plots=True)
