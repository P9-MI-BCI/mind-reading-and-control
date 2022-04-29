from classes.Simulation import Simulation
from data_training.EEGModels.training import get_EEGNet, stratified_kfold_cv


def eegnet_simulation(X, Y, scaler, config, online_data):
    eeg_net = get_EEGNet(X)

    model = stratified_kfold_cv(X, Y, eeg_net)

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
