# this file functions as a default method that takes in a scikit model and runs training and prediction
# the current working models are KNN and SVM

from sklearn.neighbors import KNeighborsClassifier
from data_preprocessing.train_test_split import format_dataset
from data_training.measurements import combine_predictions, get_precision, get_recall, get_f1_score, \
    get_confusion_matrix
from data_training.measurements import get_accuracy
import pandas as pd


def scikit_classifier(model, train_data, test_data, channels=None, features='raw'):
    if channels is None:
        channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    score_dict = {}
    ensemble_predictions_train = []
    ensemble_predictions_test = []

    _, y_train = format_dataset(train_data, channel=0, features=features)
    _, y_test = format_dataset(test_data, channel=0, features=features)

    for channel in channels:
        x_train, _ = format_dataset(train_data, channel=channel, features=features)
        x_test, _ = format_dataset(test_data, channel=channel, features=features)

        model.fit(x_train, y_train)

        train_preds = model.predict(x_train)
        test_preds = model.predict(x_test)

        ensemble_predictions_train.append(train_preds)
        ensemble_predictions_test.append(test_preds)

        score_dict[channel] = {'accuracy_train': get_accuracy(train_preds, y_train),
                               'accuracy_test': get_accuracy(test_preds, y_test),
                               'precision_train': get_precision(train_preds, y_train),
                               'precision_test': get_precision(test_preds, y_test),
                               'recall_train': get_recall(train_preds, y_train),
                               'recall_test': get_recall(test_preds, y_test),
                               'f1_train': get_f1_score(train_preds, y_train),
                               'f1_test': get_f1_score(test_preds, y_test),
                               }

    ensemble_preds_train = combine_predictions(ensemble_predictions_train).astype(int)
    ensemble_preds_test = combine_predictions(ensemble_predictions_test).astype(int)

    score_dict['ensemble'] = {'accuracy_train': get_accuracy(ensemble_preds_train, y_train),
                              'accuracy_test': get_accuracy(ensemble_preds_test, y_test),
                              'precision_train': get_precision(ensemble_preds_train, y_train),
                              'precision_test': get_precision(ensemble_preds_test, y_test),
                              'recall_train': get_recall(ensemble_preds_train, y_train),
                              'recall_test': get_recall(ensemble_preds_test, y_test),
                              'f1_train': get_f1_score(ensemble_preds_train, y_train),
                              'f1_test': get_f1_score(ensemble_preds_test, y_test),
                              }

    return pd.DataFrame(score_dict)
