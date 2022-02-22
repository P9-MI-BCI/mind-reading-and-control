from data_preprocessing.emg_processing import onset_detection, multi_dataset_onset_detection
from data_preprocessing.filters import data_filtering, multi_dataset_filtering
from data_preprocessing.init_dataset import init, get_dataset_paths, create_dataset


def dispatch(subject_id, config):

    training_dataset_path, online_dataset_path, dwell_dataset_path = get_dataset_paths(subject_id)

    # consists of list of training dataset
    training_data = create_dataset(training_dataset_path, config)

    online_data = create_dataset(online_dataset_path, config)
    dwell_data = create_dataset(dwell_dataset_path, config)

    multi_dataset_onset_detection(training_data, config)

    filtered_training_data = multi_dataset_filtering(config.DELTA_BAND, config, training_data)
    filtered_online_data = data_filtering(config.DELTA_BAND, config, online_data)
    filtered_dwell_data = data_filtering(config.DELTA_BAND, config, dwell_data)

