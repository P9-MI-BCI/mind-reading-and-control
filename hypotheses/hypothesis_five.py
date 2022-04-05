from classes.Simulation import Simulation
from data_preprocessing.emg_processing import multi_dataset_onset_detection
from data_preprocessing.filters import multi_dataset_filtering, data_filtering
from data_preprocessing.handcrafted_feature_extraction import extract_features
from data_preprocessing.init_dataset import get_dataset_paths, create_dataset
from data_preprocessing.data_distribution import data_preparation, normalization, online_data_labeling
from data_training.XGBoost.xgboost_hub import xgboost_training

"""
Hypothesis five aims to test whether deep feature extraction models will perform better than handcrafted ones. It
should compare the hand crafted feature extraction from last semester with the new one.
"""


def run(config):
    """
    Deep feature extraction will generalize better to cross-session datasets compared to handcrafted features.
    """
    subject_id = int(input("Choose subject to predict on 0-9\n"))
    config.transfer_learning = False
    config.rest_classification = True

    training_dataset_path, online_dataset_path, dwell_dataset_path = get_dataset_paths(subject_id, config)

    training_data = create_dataset(training_dataset_path, config)
    online_data = create_dataset(online_dataset_path, config)
    dwell_data = create_dataset(dwell_dataset_path, config)

    multi_dataset_onset_detection(training_data, config)
    multi_dataset_onset_detection(online_data, config, is_online=True)
    dwell_data.onsets_index = []

    multi_dataset_filtering(config.DELTA_BAND, config, training_data)
    multi_dataset_filtering(config.DELTA_BAND, config, online_data)
    data_filtering(config.DELTA_BAND, config, dwell_data)

    X, Y = data_preparation(training_data, config)
    X, scaler = normalization(X)

    # extract hand crafted features
    X = extract_features(X)

    model = xgboost_training(X, Y)

    simulation = Simulation(config)
    simulation.set_normalizer(scaler)
    simulation.set_filter(config.DELTA_BAND)
    simulation.set_feature_extraction(True)
    simulation.set_evaluation_metrics()
    simulation.load_models(model)

    # simulation.tune_dwell(dwell_data)

    # test the first dataset
    simulation.mount_dataset(online_data[0])
    simulation.simulate(real_time=False)

    simulation.reset()

    # test the second dataset
    simulation.mount_dataset(online_data[1])
    simulation.simulate(real_time=False)

    #TODO implement our deep learning method


