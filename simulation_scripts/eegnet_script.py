from datetime import datetime
from classes.Simulation import Simulation
from data_training.EEGModels.training import get_EEGNet, stratified_kfold_cv
import time

from utility.logger import result_logger


def eegnet_simulation(X, Y, scaler, config, online_data, dwell_data, hypothesis_logger_location):
    now = datetime.now()

    now = now.strftime("%H:%M:%S")
    eeg_logger_location = 'eeg_net.txt'
    result_logger(eeg_logger_location, f'EEGNET SIMULATION MODULE: {config.logger_id} ----- {now}\n')
    eeg_net = get_EEGNet(X)

    model = stratified_kfold_cv(X, Y, eeg_net, logger_location=eeg_logger_location)

    # Simulation
    simulation = Simulation(config)

    simulation.set_normalizer(scaler)
    simulation.set_filter(config.EEGNET_BAND)
    simulation.set_evaluation_metrics()
    simulation.set_logger_location(eeg_logger_location)

    # First model
    simulation.load_models(model)
    simulation.tune_dwell(dwell_data)

    # test the first dataset
    result_logger(eeg_logger_location, f'-- Simulating Test 1 \n')
    simulation.mount_dataset(online_data[0])
    simulation.simulate(real_time=False)

    simulation.reset()

    # test the second dataset
    result_logger(eeg_logger_location, f'-- Simulating Test 2 \n')
    simulation.mount_dataset(online_data[1])
    simulation.simulate(real_time=False)

    now = datetime.now()
    result_logger(hypothesis_logger_location, f'EEGNET simulation finished: {now} \n')