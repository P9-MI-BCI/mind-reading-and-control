import matplotlib.pyplot

from data_preprocessing.emg_processing import onset_detection, multi_dataset_onset_detection
from data_preprocessing.filters import data_filtering, multi_dataset_filtering
from data_preprocessing.init_dataset import init, get_dataset_paths, create_dataset
from data_preprocessing.data_distribution import data_preparation, online_data_labeling, normalization
from data_preprocessing.recurrence_quantification_analysis import recurrence_quantification
from data_preprocessing.wavelet_transformation import wavelet_transformation
from data_training.EEGModels.training import EEGModels_training_hub
from data_training.SVM.training import svm_training_hub
from data_visualization.mne_visualization import visualize_mne
from data_visualization.brain_activity import visualize_brain_activity
from data_preprocessing.downsampling import downsample


def dispatch(subject_id, config):
    """
    Finds the paths for the datasets, and creates the initial dataset classes while loading them in.
    The training dataset will initially contain an array of datasets (Opening and closing data)
    """
    training_dataset_path, online_dataset_path, dwell_dataset_path = get_dataset_paths(subject_id, config)

    training_data = create_dataset(training_dataset_path, config)
    online_data = create_dataset(online_dataset_path, config)
    dwell_data = create_dataset(dwell_dataset_path, config)

    """
    Perform onset detection on the movement data and annotate the dataset with the indexes of the beginning
    of movement and end of movement. 
    """
    multi_dataset_onset_detection(training_data, config)
    multi_dataset_onset_detection(online_data, config, is_online=False)

    """
    Modules to filter the data, functions can take variety of default frequency bands annotated in the 
    json_config/default.json file. Method include possibility of handling multiple datasets at once. 
    """

    multi_dataset_filtering(config.BETA_BAND, config, training_data)
    multi_dataset_filtering(config.BASELINE, config, online_data)
    data_filtering(config.BASELINE, config, dwell_data)

    """
    Down sample testing. Visualizes the data with the MNE data structures.
    """
    # downsample(training_data, config)
    # visualize_mne(training_data, config)
    # visualize_brain_activity(training_data, config)

    """
    Feature extraction. 
    """
    # recurrence_quantification(training_data, config, plot=True)
    wavelet_transformation(training_data[0], config, wavelet_type='discrete', plot=True)

    """
    Prepare data for the models by combining the training datasets into a single vector. Each sample is cut
    into a sliding window defined by the config.window_padding parameter. The data is shuffled during creation.
    Also checks if test label file is available to overwrite the default labels for the online test dataset(s).
    """
    X, Y = data_preparation(training_data, config)
    X, scaler = normalization(X)
    online_X, online_Y = online_data_labeling(online_data, config, scaler, subject_id)

    # svm_training_hub(X, Y, online_X, online_Y)
    # EEGModels_training_hub(X, Y, online_X, online_Y)



