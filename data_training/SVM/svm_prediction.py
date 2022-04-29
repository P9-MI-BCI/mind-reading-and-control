from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from data_training.scikit_classifiers import scikit_classifier, scikit_classifier_loocv, scikit_classifier_loocv_csp
from sklearn import svm
import pandas as pd

def svm_classifier(train_data, test_data, channels=None, features='raw'):
    model = svm.SVC()

    result = scikit_classifier(model, train_data, test_data, channels, features, dir_name='svm')

    return result


def svm_classifier_loocv(data, channels=None, features='raw', prediction='whole'):

    model = svm.SVC()
    if features == 'csp':
        result = scikit_classifier_loocv_csp(model, data, channels, features, dir_name='svm', prediction=prediction)
    else:
        result, model = scikit_classifier_loocv(model, data, channels, features, dir_name='svm', prediction=prediction)

    return result, model


def svm_cv(X, Y):
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    cv_scores = {}
    split = 0
    for train_index, val_index in skf.split(X, Y):
        model = svm.SVC()

        model.fit(X[train_index], Y[train_index])

        validation_prediction = model.predict(X[val_index])

        accuracy = accuracy_score(validation_prediction, Y[val_index])

        cv_scores[f'split_{split}'] = accuracy
        split += 1

    cv_scores = pd.DataFrame(cv_scores, index=[0])
    cv_scores['mean'] = cv_scores.mean(axis=1)
    cv_scores['std'] = cv_scores.std(axis=1)
    print(f'{cv_scores}')

    return model