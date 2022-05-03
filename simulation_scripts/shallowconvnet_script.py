from datetime import datetime

from classes.Simulation import Simulation
from data_training.EEGModels.training import get_ShallowConvNet, stratified_kfold_cv
import time

from utility.logger import result_logger


def shallowconvnet_simulation(X, Y, scaler, config, online_data, dwell_data):
    now = datetime.now()

    now = now.strftime("%H:%M:%S")
    shallow_logger_location = 'shallowconvnet.txt'
    result_logger(shallow_logger_location, f'SHALLOW CONV NET SIMULATION MODULE ----------- {now}\n')
    shallow_deep_net = get_ShallowConvNet(X)

    model = stratified_kfold_cv(X, Y, shallow_deep_net, logger_location=shallow_logger_location)

    # Simulation
    simulation = Simulation(config)

    simulation.set_normalizer(scaler)
    simulation.set_filter(config.BASELINE)
    simulation.set_evaluation_metrics()
    simulation.set_logger_location(shallow_logger_location)

    # First model
    simulation.load_models(model)
    simulation.tune_dwell(dwell_data)

    # test the first dataset
    simulation.mount_dataset(online_data[0])
    simulation.simulate(real_time=False)

    simulation.reset()

    # test the second dataset
    simulation.mount_dataset(online_data[1])
    simulation.simulate(real_time=False)
