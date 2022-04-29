from classes.Simulation import Simulation
from data_training.EEGModels.training import get_DeepConvNet, stratified_kfold_cv


def deepconvnet_simulation(X, Y, scaler, config, online_data, dwell_data):
    shallow_deep_net = get_DeepConvNet(X)

    model = stratified_kfold_cv(X, Y, shallow_deep_net)

    # Simulation
    simulation = Simulation(config)

    simulation.set_normalizer(scaler)
    simulation.set_filter(config.BASELINE)
    simulation.set_evaluation_metrics()

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
