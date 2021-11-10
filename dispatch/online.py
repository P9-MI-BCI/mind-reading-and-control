from classes.Simulation import Simulation
from data_training.scikit_classifiers import load_scikit_classifiers


def online(config, dataset):
    # Load models
    models = load_scikit_classifiers('knn')
    # Create Simulation
    simulation = Simulation(config)
    simulation.mount_dataset(dataset)
    simulation.load_models(models)
    simulation.evaluation_metrics()
    simulation.simulate(real_time=False)
