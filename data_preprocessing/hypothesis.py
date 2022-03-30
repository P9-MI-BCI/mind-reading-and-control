"""
Holds information for all the hypotheses. Each function will run the necessary data division for the hypothesis chosen.
"""
from data_preprocessing.emg_processing import multi_dataset_onset_detection
from data_preprocessing.filters import multi_dataset_filtering, data_filtering
from data_preprocessing.init_dataset import format_input, get_dataset_paths, create_dataset
from data_preprocessing.data_distribution import data_preparation, online_data_labeling, normalization
from data_preprocessing.downsampling import downsample
from data_training.EEGModels.training import EEGModels_training_hub


# Hypothesis one aims to test the difference between spatial and temporal feature extraction. It will test if combining
# the feature extraction improves the results
def hypothesis_one(config):
    """
    A combination of features extracted from different deep learning algorithms will improve the classification
    """
    pass


# Hypothesis two aims to test if one half of the brain is more active during a movement. Looking in the possibility of
# using spatial filtering across all the bands or separating them bands into each half.
def hypothesis_two(config):
    """
    The electrode layout will allow for differentiation between rest and movement in the EEG signals
    """
    pass


# Hypothesis three aims to test if training only on cross subject data is sufficient to achieve a 50%+ accuracy on
# an input subject.
def hypothesis_three(config):
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

    downsample(training_data, config)
    downsample(online_data, config)
    """
    Prepare data for the models by combining the training datasets into a single vector. Each sample is cut
    into a sliding window defined by the config.window_padding parameter. The data is shuffled during creation.
    Also checks if test label file is available to overwrite the default labels for the online test dataset(s).
    """
    # TODO implement our deep learning method
    X, Y = data_preparation(training_data, config)
    X, scaler = normalization(X)
    online_X, online_Y = online_data_labeling(online_data, config, scaler, subject_id)
    EEGModels_training_hub(X, Y, online_X, online_Y)


# Hypothesis four aims to test if training cross subject with subject specific calibration can improve the accuracy, and
# achieve a higher accuracy than the other hypothesis.
def hypothesis_four(config):
    """
    Calibration will improve the accuracy of the model when predicting on a new subject.
    """
    subject_id = int(input("Choose subject to predict on 0-9\n"))
    config.transfer_learning = True
    config.rest_classification = True

    training_dataset_path, online_dataset_path, dwell_dataset_path = get_dataset_paths(subject_id, config)
    pass


# Hypothesis five aims to test whether deep feature extraction models will perform better than handcrafted ones. It
# should compare the hand crafted feature extraction from last semester with the new one.
def hypothesis_five(config):
    """
    Deep feature extraction will generalize better to cross-session datasets compared to handcrafted features.
    """
    pass


# Hypothesis six aims to test if recording more data helped improving the accuracy of the model, by adding increments of
# data to the training.
def hypothesis_six(config):
    """
    A high number of recorded movements from each subject will help improve classification of our models.
    """
    pass



