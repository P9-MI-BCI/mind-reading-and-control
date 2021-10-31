from data_training.scikit_classifiers import scikit_classifier, scikit_classifier_loocv
from sklearn import svm


def svm_classifier(train_data, test_data, channels=None, features='raw'):
    model = svm.SVC()

    result = scikit_classifier(model, train_data, test_data, channels, features, dir_name='svm')

    return result


def svm_classifier_loocv(data, channels=None, features='raw', prediction='whole'):

    model = svm.SVC()

    result = scikit_classifier_loocv(model, data, channels, features, dir_name='svm', prediction=prediction)

    return result