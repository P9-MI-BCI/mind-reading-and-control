from datetime import datetime

from classes.Simulation import Simulation
from data_preprocessing.downsampling import downsample
from data_preprocessing.emg_processing import multi_dataset_onset_detection
from data_preprocessing.filters import multi_dataset_filtering, data_filtering
from data_preprocessing.handcrafted_feature_extraction import extract_features
from data_preprocessing.init_dataset import get_dataset_paths, create_dataset
from data_preprocessing.data_distribution import data_preparation, normalization, online_data_labeling, \
    data_preparation_with_filtering, load_data_from_temp, shuffle, features_to_file, load_features_from_file, \
    load_scaler
from data_training.XGBoost.xgboost_hub import xgboost_training, optimized_xgboost
from data_training.transformer.transformer import transformer
from simulation_scripts.cspsvm_script import cspsvm_simulation
from simulation_scripts.deepconvnet_script import deepconvnet_simulation
from simulation_scripts.eegnet_script import eegnet_simulation
from simulation_scripts.shallowconvnet_script import shallowconvnet_simulation
from simulation_scripts.transformer_script import transformer_simulation
from simulation_scripts.xgboost_script import xgboost_simulation
from utility.logger import result_logger

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

    now = datetime.now()
    hypothesis_logger_location = 'hypothesis_five.txt'
    result_logger(hypothesis_logger_location, f'----- Initializing Hypothesis Five -- start time: {now} \n')
    result_logger(hypothesis_logger_location, f'Logger ID: {config.logger_id} \n')
    result_logger(hypothesis_logger_location, f'Subject ID: {subject_id} \n')
    result_logger(hypothesis_logger_location, f'Transfer learning: {config.transfer_learning} \n')
    result_logger(hypothesis_logger_location, f'Rest Classification: {config.rest_classification} \n')
    training_dataset_path, online_dataset_path, dwell_dataset_path = get_dataset_paths(subject_id, config)

    training_data = create_dataset(training_dataset_path, config)
    online_data = create_dataset(online_dataset_path, config)
    dwell_data = create_dataset(dwell_dataset_path, config)

    multi_dataset_onset_detection(training_data, config)
    multi_dataset_onset_detection(online_data, config)
    multi_dataset_onset_detection(dwell_data, config)
    # dwell_data.clusters = []

    data_preparation_with_filtering(training_data, config, config.DELTA_BAND)
    X, Y = load_data_from_temp()
    X, Y = shuffle(X, Y)
    X, scaler = normalization(X)
    xgboost_simulation(X, Y, scaler, config, online_data, dwell_data, hypothesis_logger_location)
    transformer_simulation(X, Y, scaler, config, online_data, dwell_data, hypothesis_logger_location)

    training_down_sampled = downsample(training_data, config)
    online_down_sampled = downsample(online_data, config)
    dwell_down_sampled = downsample([dwell_data], config) # [0]

    data_preparation_with_filtering(training_down_sampled, config, config.EEGNET_BAND)
    X, Y = load_data_from_temp()
    X, Y = shuffle(X, Y)
    X, scaler = normalization(X)

    cspsvm_simulation(X, Y, scaler, config, online_down_sampled, dwell_down_sampled, hypothesis_logger_location)
    eegnet_simulation(X, Y, scaler, config, online_down_sampled, dwell_down_sampled, hypothesis_logger_location)
    deepconvnet_simulation(X, Y, scaler, config, online_down_sampled, dwell_down_sampled, hypothesis_logger_location)
    shallowconvnet_simulation(X, Y, scaler, config, online_down_sampled, dwell_down_sampled, hypothesis_logger_location)

    now = datetime.now()
    result_logger(hypothesis_logger_location, f'----- Hypothesis five has been completed: {now} \n')

