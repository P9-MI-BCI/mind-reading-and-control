from datetime import datetime

from classes.Simulation import Simulation
from data_training.EEGModels.training import get_DeepConvNet, stratified_kfold_cv
from utility.logger import result_logger
import time


def deepconvnet_simulation(X, Y, scaler, config, online_data, dwell_data, hypothesis_logger_location):
    now = datetime.now()

    now = now.strftime("%H:%M:%S")
    deepconvnet_logger_location = 'deepconvnet.txt'
    result_logger(deepconvnet_logger_location, f'DEEP CONV NET SIMULATION MODULE: {config.logger_id} ----------- {now}\n')
    shallow_deep_net = get_DeepConvNet(X)

    model = stratified_kfold_cv(X, Y, shallow_deep_net, logger_location=deepconvnet_logger_location)

    # Simulation
    simulation = Simulation(config)

    simulation.set_normalizer(scaler)
    simulation.set_filter(config.BASELINE)
    simulation.set_evaluation_metrics()
    simulation.set_logger_location(deepconvnet_logger_location)

    # First model
    simulation.load_models(model)
    simulation.tune_dwell(dwell_data)

    # test the first dataset
    result_logger(deepconvnet_logger_location, f'-- Simulating Test 1 \n')
    simulation.mount_dataset(online_data[0])
    simulation.simulate(real_time=False)

    simulation.reset()

    # test the second dataset
    result_logger(deepconvnet_logger_location, f'-- Simulating Test 2 \n')
    simulation.mount_dataset(online_data[1])
    simulation.simulate(real_time=False)

    now = datetime.now()
    result_logger(hypothesis_logger_location, f'DEEPCONVNET simulation finished: {now} \n')
