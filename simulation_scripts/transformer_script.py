from datetime import datetime

from classes.Simulation import Simulation
from data_training.transformer.transformer import transformer
import time

from utility.logger import result_logger


def transformer_simulation(X, Y, scaler, config, online_data, dwell_data):
    now = datetime.now()

    now = now.strftime("%H:%M:%S")
    transformer_logger_location = 'transformer.txt'
    result_logger(transformer_logger_location, f'TRANSFORMER SIMULATION MODULE ----------- {now}\n')
    model = transformer(X, Y, logger_location=transformer_logger_location)

    # Simulation
    simulation = Simulation(config)

    simulation.set_normalizer(scaler)
    simulation.set_filter(config.DELTA_BAND)
    simulation.set_evaluation_metrics()
    simulation.set_logger_location(transformer_logger_location)

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