import matplotlib.pyplot

from data_preprocessing.emg_processing import onset_detection, multi_dataset_onset_detection
from data_preprocessing.filters import data_filtering, multi_dataset_filtering
from data_preprocessing.hypothesis import hypothesis_one, hypothesis_two, hypothesis_three, hypothesis_four, \
    hypothesis_five, hypothesis_six
from data_preprocessing.init_dataset import init, get_dataset_paths, create_dataset, format_input
from data_preprocessing.data_distribution import data_preparation, online_data_labeling, normalization
from data_training.EEGModels.training import EEGModels_training_hub
from data_visualization.mne_visualization import visualize_mne
from data_preprocessing.downsampling import downsample
from utility.logger import get_logger


def dispatch(config):

    if config.hypothesis_choice == 1:
        hypothesis_one(config)
    elif config.hypothesis_choice == 2:
        hypothesis_two(config)
    elif config.hypothesis_choice == 3:
        hypothesis_three(config)
    elif config.hypothesis_choice == 4:
        hypothesis_four(config)
    elif config.hypothesis_choice == 5:
        hypothesis_five(config)
    elif config.hypothesis_choice == 6:
        hypothesis_six(config)
    else:
        get_logger().info('No valid hypothesis selected - exiting')
        exit()



