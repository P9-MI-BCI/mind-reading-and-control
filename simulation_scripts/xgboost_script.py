from datetime import datetime

from classes.Simulation import Simulation
from data_preprocessing.data_distribution import features_to_file, load_features_from_file
from data_preprocessing.handcrafted_feature_extraction import extract_features
from data_training.XGBoost.xgboost_hub import xgboost_training
import time

from utility.logger import result_logger


def xgboost_simulation(X, Y, scaler, config, online_data, dwell_data, hypothesis_logger_location):
    now = datetime.now()

    now = now.strftime("%H:%M:%S")
    xgboost_logger_location = 'xgboost.txt'
    result_logger(xgboost_logger_location, f'XGBOOST SIMULATION MODULE: {config.logger_id} ----- {now}\n')

    # Simulation
    simulation = Simulation(config)

    simulation.set_normalizer(scaler)
    simulation.set_filter(config.DELTA_BAND)
    simulation.set_evaluation_metrics()
    simulation.feature_extraction(True, extract_features)
    simulation.set_logger_location(xgboost_logger_location)

    # simulate for xgboost
    X = extract_features(X)
    features_to_file(X, Y, scaler)
    X, Y = load_features_from_file()
    model = xgboost_training(X, Y, logger_location=xgboost_logger_location)
    # optimized_xgboost(X, Y)

    simulation.load_models(model)
    simulation.tune_dwell(dwell_data)

    # test the first dataset
    result_logger(xgboost_logger_location, f'-- Simulating Test 1 \n')
    simulation.mount_dataset(online_data[0])
    simulation.simulate(real_time=False)

    simulation.reset()

    # test the second dataset
    result_logger(xgboost_logger_location, f'-- Simulating Test 1 \n')
    simulation.mount_dataset(online_data[1])
    simulation.simulate(real_time=False)

    now = datetime.now()
    result_logger(hypothesis_logger_location, f'XGBoost simulation finished: {now} \n')