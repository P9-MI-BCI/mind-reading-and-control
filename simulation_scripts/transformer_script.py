from classes.Simulation import Simulation
from data_training.transformer.transformer import transformer


def transformer_simulation(X, Y, scaler, config, online_data):

    model = transformer(X, Y)

    # Simulation
    simulation = Simulation(config)

    simulation.set_normalizer(scaler)
    simulation.set_filter(config.DELTA_BAND)
    simulation.set_evaluation_metrics()

    # First model
    simulation.load_models(model)
    simulation.tune_dwell(online_data[0])

    # test the first dataset
    simulation.mount_dataset(online_data[0])
    simulation.simulate(real_time=False)

    simulation.reset()

    # test the second dataset
    simulation.mount_dataset(online_data[1])
    simulation.simulate(real_time=False)