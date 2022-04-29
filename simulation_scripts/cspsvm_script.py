from mne.decoding import CSP

from classes.Simulation import Simulation
from data_training.SVM.svm_prediction import svm_cv


def cspsvm_simulation(X, Y, scaler, config, online_data):
    csp = CSP(n_components=3, reg=None, log=True, norm_trace=False)
    X = csp.fit_transform(X, Y)

    model = svm_cv(X, Y)

    simulation = Simulation(config)

    simulation.set_normalizer(scaler)
    simulation.set_filter(config.DELTA_BAND)
    simulation.set_evaluation_metrics()
    simulation.feature_extraction(True, csp)

    simulation.load_models(model)
    simulation.tune_dwell(online_data[0])

    # test the first dataset
    simulation.mount_dataset(online_data[0])
    simulation.simulate(real_time=False)

    simulation.reset()

    # test the second dataset
    simulation.mount_dataset(online_data[1])
    simulation.simulate(real_time=False)

    pass