from data_preprocessing.data_distribution import normalization, online_data_labeling, data_preparation
from data_preprocessing.emg_processing import multi_dataset_onset_detection
from data_preprocessing.filters import multi_dataset_filtering, data_filtering
from data_preprocessing.init_dataset import get_dataset_paths, create_dataset


# Hypothesis three aims to test if training only on cross subject data is sufficient to achieve a 50%+ accuracy on
# an input subject.
def run(config):
    """
    A model can be trained to predict a new subject without any calibration.
    """
    subject_id = int(input("Choose subject to predict on 0-9\n"))
    config.transfer_learning = True
    config.rest_classification = True

    training_dataset_path, online_dataset_path, dwell_dataset_path = get_dataset_paths(subject_id, config)

    training_data = create_dataset(training_dataset_path, config)
    online_data = create_dataset(online_dataset_path, config)
    dwell_data = create_dataset(dwell_dataset_path, config)

    multi_dataset_onset_detection(training_data, config)
    multi_dataset_onset_detection(online_data, config, is_online=True)

    multi_dataset_filtering(config.BASELINE, config, training_data)
    multi_dataset_filtering(config.BASELINE, config, online_data)
    data_filtering(config.BASELINE, config, dwell_data)

    """
    Prepare data for the models by combining the training datasets into a single vector. Each sample is cut
    into a sliding window defined by the config.window_padding parameter. The data is shuffled during creation.
    Also checks if test label file is available to overwrite the default labels for the online test dataset(s).
    """
    # TODO implement our deep learning method
    X, Y = data_preparation(training_data, config)
    X, scaler = normalization(X)
    online_X, online_Y = online_data_labeling(online_data, config, scaler, subject_id)
    # EEGModels_training_hub(X, Y, online_X, online_Y)
