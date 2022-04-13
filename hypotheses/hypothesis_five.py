from classes.Simulation import Simulation
from data_preprocessing.emg_processing import multi_dataset_onset_detection
from data_preprocessing.filters import multi_dataset_filtering, data_filtering
from data_preprocessing.handcrafted_feature_extraction import extract_features
from data_preprocessing.init_dataset import get_dataset_paths, create_dataset
from data_preprocessing.data_distribution import data_preparation, normalization, online_data_labeling, \
    data_preparation_with_filtering, load_data_from_temp, shuffle, features_to_file, load_features_from_file, \
    load_scaler
from data_training.XGBoost.xgboost_hub import xgboost_training, optimized_xgboost

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
    dwell_data.clusters = []

    data_preparation_with_filtering(training_data, config)
    X, Y = load_data_from_temp()
    X, Y = shuffle(X, Y)
    X, scaler = normalization(X)

    # extract hand crafted features
    X = extract_features(X)
    features_to_file(X, Y, scaler)
    X, Y = load_features_from_file()
    scaler = load_scaler()

    model = xgboost_training(X, Y)
    # optimized_xgboost(X, Y)
    simulation = Simulation(config)

    simulation.set_normalizer(scaler)
    simulation.set_filter(config.DELTA_BAND)
    simulation.set_feature_extraction(True)
    simulation.set_evaluation_metrics()
    simulation.load_models(model)

    simulation.tune_dwell(online_data[0])

    # test the first dataset
    simulation.mount_dataset(online_data[0])
    simulation.simulate(real_time=False)

    simulation.reset()

    # test the second dataset
    simulation.mount_dataset(online_data[1])
    simulation.simulate(real_time=False)

    #TODO implement our deep learning method


