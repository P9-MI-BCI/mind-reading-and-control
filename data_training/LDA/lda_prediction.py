from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

from data_training.scikit_classifiers import scikit_classifier

pd.options.mode.chained_assignment = None


def lda_classifier(train_data, test_data, channels=None, features='raw'):
    model = LinearDiscriminantAnalysis()

    result = scikit_classifier(model, train_data, test_data, channels, features)

    return result


