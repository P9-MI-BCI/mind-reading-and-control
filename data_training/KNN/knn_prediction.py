import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from data_training.scikit_classifiers import scikit_classifier, scikit_classifier_loocv, scikit_classifier_loocv_csp

pd.options.mode.chained_assignment = None


def knn_classifier(train_data, test_data, channels=None, features='raw'):
    model = KNeighborsClassifier(n_neighbors=3)

    result = scikit_classifier(model, train_data, test_data, channels, features, dir_name='knn')

    return result


def knn_classifier_loocv(data, channels=None, features='raw', prediction='whole'):

    model = KNeighborsClassifier(n_neighbors=3)

    if features == 'csp':
        result = scikit_classifier_loocv_csp(model, data, channels, features, dir_name='knn', prediction=prediction)
    else:
        result, model = scikit_classifier_loocv(model, data, channels, features, dir_name='knn', prediction=prediction)

    return result, model
