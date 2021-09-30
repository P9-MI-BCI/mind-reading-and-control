from data_training.scikit_classifiers import scikit_classifier
from sklearn import svm


def svm_classifier(train_data, test_data, channels=None):
    model = svm.SVC()

    result = scikit_classifier(model, train_data, test_data, channels)

    return result