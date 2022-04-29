from classes.Simulation import Simulation
from data_preprocessing.data_distribution import features_to_file, load_features_from_file
from data_preprocessing.handcrafted_feature_extraction import extract_features
from data_training.XGBoost.xgboost_hub import xgboost_training


def xgboost_simulation(X, Y, scaler, config, online_data):
    # Simulation
    simulation = Simulation(config)

    simulation.set_normalizer(scaler)
    simulation.set_filter(config.DELTA_BAND)
    simulation.set_evaluation_metrics()
    simulation.feature_extraction(True, extract_features)

    # simulate for xgboost
    X = extract_features(X)
    features_to_file(X, Y, scaler)
    X, Y = load_features_from_file()
    model = xgboost_training(X, Y)
    # optimized_xgboost(X, Y)

    simulation.load_models(model)
    simulation.tune_dwell(online_data[0])

    # test the first dataset
    simulation.mount_dataset(online_data[0])
    simulation.simulate(real_time=False)

    simulation.reset()

    # test the second dataset
    simulation.mount_dataset(online_data[1])
    simulation.simulate(real_time=False)