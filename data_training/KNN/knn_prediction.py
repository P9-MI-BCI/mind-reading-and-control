from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

from data_training.scikit_classifiers import scikit_classifier

pd.options.mode.chained_assignment = None


def knn_classifier(train_data, test_data, channels=None, features='raw'):
    model = KNeighborsClassifier(n_neighbors=3)

    result = scikit_classifier(model, train_data, test_data, channels, features)

    return result


