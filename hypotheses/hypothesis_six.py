# Hypothesis six aims to test if recording more data helped improving the accuracy of the model, by adding increments of
# data to the training.
from data_preprocessing.data_distribution import data_preparation, normalization, online_data_labeling
from data_preprocessing.emg_processing import multi_dataset_onset_detection
from data_preprocessing.filters import multi_dataset_filtering, data_filtering
from data_preprocessing.init_dataset import get_dataset_paths, create_dataset
# from data_training.EEGModels.training import EEGModels_training_hub
from utility import logger

def run(config):
    """
    A high number of recorded movements from each subject will help improve classification of our models.
    """

    subject_id = int(input("Choose subject to predict on 0-9\n"))
    config.transfer_learning = False
    config.rest_classification = True

    training_dataset_path, online_dataset_path, dwell_dataset_path = get_dataset_paths(subject_id, config)

    training_data = create_dataset(training_dataset_path, config)
    online_data = create_dataset(online_dataset_path, config)
    dwell_data = create_dataset(dwell_dataset_path, config)

    multi_dataset_onset_detection(training_data, config)
    multi_dataset_onset_detection(online_data, config)

    multi_dataset_filtering(config.DELTA_BAND, config, training_data)
    multi_dataset_filtering(config.DELTA_BAND, config, online_data)
    data_filtering(config.DELTA_BAND, config, dwell_data)

    X, Y = data_preparation(training_data, config)
    X, scaler = normalization(X)
    online_X, online_Y = online_data_labeling(online_data, config, scaler, subject_id)

    # Hypothesis data divide
    logger.get_logger().info('Training using 25% of the dataset')
    X_25, Y_25 = select_dataset_subset(X, Y, 25)
    # EEGModels_training_hub(X_25, Y_25, online_X, online_Y)

    logger.get_logger().info('Training using 50% of the dataset')
    X_50, Y_50 = select_dataset_subset(X, Y, 50)
    # EEGModels_training_hub(X_50, Y_50, online_X, online_Y)

    logger.get_logger().info('Training using 75% of the dataset')
    X_75, Y_75 = select_dataset_subset(X, Y, 75)
    # EEGModels_training_hub(X_75, Y_75, online_X, online_Y)

    logger.get_logger().info('Training using 100% of the dataset')
    # EEGModels_training_hub(X, Y, online_X, online_Y)


def select_dataset_subset(X, Y, percentage_data):
    return X[:int(len(X) * percentage_data / 100)], \
           Y[:int(len(X) * percentage_data / 100)]
