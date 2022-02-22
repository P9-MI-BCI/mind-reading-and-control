from data_preprocessing.emg_processing import onset_detection, multi_dataset_onset_detection
from data_preprocessing.filters import data_filtering, multi_dataset_filtering
from data_preprocessing.init_dataset import init, get_dataset_paths, create_dataset


def dispatch(subject_id, config):
    """
    Finds the paths for the datasets, and creates the initial dataset classes while loading them in.
    The training dataset will initially contain an array of datasets (Opening and closing data)
    """
    training_dataset_path, online_dataset_path, dwell_dataset_path = get_dataset_paths(subject_id)

    training_data = create_dataset(training_dataset_path, config)
    online_data = create_dataset(online_dataset_path, config)
    dwell_data = create_dataset(dwell_dataset_path, config)

    """
    Perform onset detection on the movement data and annotate the dataset with the indexes of the beginning
    of movement and end of movement. 
    """
    multi_dataset_onset_detection(training_data, config)

    """
    Modules to filter the data, functions can take variety of default frequency bands annotated in the 
    json_config/default.json file. Method include possibility of handling multiple datasets at once. 
    """
    multi_dataset_filtering(config.DELTA_BAND, config, training_data)
    data_filtering(config.DELTA_BAND, config, online_data)
    data_filtering(config.DELTA_BAND, config, dwell_data)

    
