import json

from classes.Simulation import Simulation
from data_training.scikit_classifiers import load_scikit_classifiers
from data_preprocessing.init_dataset import init


def online(config, dataset):
    calibration_dataset = init(selected_cue_set=1)

    # Create Simulation
    simulation = Simulation(config)
    simulation.mount_calibration_dataset(calibration_dataset)
    simulation.calibrate(centering=True)

    simulation.mount_dataset(dataset)

    # models = load_scikit_classifiers('lda')
    # simulation.load_models(models)
    simulation.evaluation_metrics()
    simulation.simulate(real_time=False)
