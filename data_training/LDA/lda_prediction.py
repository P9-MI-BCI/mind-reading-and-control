from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

from data_training.scikit_classifiers import scikit_classifier, scikit_classifier_loocv, scikit_classifier_loocv_csp

pd.options.mode.chained_assignment = None


def lda_classifier(train_data, test_data, channels=None, features='raw'):
    model = LinearDiscriminantAnalysis()

    result = scikit_classifier(model, train_data, test_data, channels, features, dir_name='lda')

    return result


def lda_classifier_loocv(data, channels=None, features='raw', prediction='whole'):

    model = LinearDiscriminantAnalysis()

    if features == 'csp':
        result = scikit_classifier_loocv_csp(model, data, channels, features, dir_name='lda', prediction=prediction)
    else:
        result, model = scikit_classifier_loocv(model, data, channels, features, dir_name='lda', prediction=prediction)

    return result, model