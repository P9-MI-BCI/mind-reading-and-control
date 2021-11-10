# this file functions as a default method that takes in a scikit model and runs training and prediction
# the current tested models are KNN, SVM, LDA, LGBM
import glob
import pickle
import os
import pandas as pd
from data_preprocessing.train_test_split import format_dataset
from data_training.measurements import combine_predictions, precision, recall, f1, \
    get_confusion_matrix, combine_loocv_predictions
from data_training.measurements import accuracy
from utility.file_util import create_dir
from definitions import OUTPUT_PATH
import statistics
import numpy as np
from classes.Window import Window


# This function is able to accept any scikit model
def scikit_classifier(model, train_data, test_data, channels=None, features='raw', save_model=True, dir_name='None'):
    if channels is None:
        channels = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    score_df = pd.DataFrame()
    score_df.index = ['accuracy_train',
                      'accuracy_test',
                      'precision_train',
                      'precision_test',
                      'recall_train',
                      'recall_test',
                      'f1_train',
                      'f1_test']
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

        score_df[channel] = [accuracy(y_train, train_preds),
                             accuracy(y_test, test_preds),
                             precision(y_train, train_preds),
                             precision(y_test, test_preds),
                             recall(y_train, train_preds),
                             recall(y_test, test_preds),
                             f1(y_train, train_preds),
                             f1(y_test, test_preds),
                             ]

        if save_model:
            path = os.path.join(OUTPUT_PATH, 'models', dir_name)
            create_dir(path, recursive=True)
            filename = os.path.join(path, dir_name + str(channel))
            pickle.dump(model, open(filename, 'wb'))

    score_df['average'] = score_df.mean(numeric_only=True, axis=1)

    ensemble_preds_train = combine_predictions(ensemble_predictions_train).astype(int)
    ensemble_preds_test = combine_predictions(ensemble_predictions_test).astype(int)

    score_df['ensemble'] = [accuracy(y_train, ensemble_preds_train),
                            accuracy(y_test, ensemble_preds_test),
                            precision(y_train, ensemble_preds_train),
                            precision(y_test, ensemble_preds_test),
                            recall(y_train, ensemble_preds_train),
                            recall(y_test, ensemble_preds_test),
                            f1(y_train, ensemble_preds_train),
                            f1(y_test, ensemble_preds_test),
                            ]

    return score_df


# used to load weights from previously trained models.
def load_scikit_classifiers(dir_name: str):
    path = os.path.join(OUTPUT_PATH, 'models', dir_name)

    models = []
    for file in glob.glob(path + '/*', recursive=True):
        models.append(pickle.load(open(file, 'rb')))

    return models


def scikit_classifier_loocv(model, data: [Window], channels=None, features='raw', save_model=True, dir_name='None',
                            prediction='whole'):
    if channels is None:
        channels = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    score_df = pd.DataFrame()
    score_df.index = ['accuracy_train',
                      'accuracy_test',
                      'precision_train',
                      'precision_test',
                      'recall_train',
                      'recall_test',
                      'f1_train',
                      'f1_test']
    ensemble_predictions_train = []
    ensemble_predictions_test = []

    pred_data = []
    if prediction == 'whole' or prediction == 'w':
        for window in data:
            if not window.is_sub_window:
                pred_data.append(window)
    elif prediction == 'sub' or prediction == 's':
        for window in data:
            if window.is_sub_window:
                pred_data.append(window)
    elif prediction == 'majority' or prediction == 'm':
        sub_data = []
        for window in data:
            if not window.is_sub_window:
                pred_data.append(window)
                temp = []
                for sub in window.sub_windows:
                    for sw in data:
                        if sw.num_id == sub:
                            temp.append(sw)
                        if len(temp) == len(window.sub_windows):
                            break
                sub_data.append(temp)

    for channel in channels:
        if prediction == 'majority' or prediction == 'm':
            sub_feats = []
            sub_labels = []
            feats, labels = format_dataset(pred_data, channel=channel, features=features)
            for sd in sub_data:
                sf, sf_lb = format_dataset(sd, channel=channel, features=features)
                sub_feats.append(sf)
                sub_labels.append(sf_lb)
        else:
            feats, labels = format_dataset(pred_data, channel=channel, features=features)

        train_predictions = []
        train_labels = []
        test_predictions = []
        test_labels = []

        temp_scores = np.zeros((len(pred_data), 4))

        for sample in range(0, len(pred_data)):
            if prediction == 'majority' or prediction == 'm':
                x_test = sub_feats[sample]
                y_test = labels[sample]
                x_train_full_window = [x for i, x in enumerate(feats) if i != sample]
                x_train_temp = [x for i, x in enumerate(sub_feats) if i != sample]
                x_train = []
                for x in x_train_temp:
                    x_train.extend(x)
                y_train_full_window = [x for i, x in enumerate(labels) if i != sample]
                y_train_temp = [x for i, x in enumerate(sub_labels) if i != sample]
                y_train = []
                for y in y_train_temp:
                    y_train.extend(y)

                model.fit(x_train, y_train)

                temp_pred = []
                for x_t in x_test:
                    temp_pred.append(model.predict([x_t]).tolist()[0])
                test_predictions.append(max(set(temp_pred), key=temp_pred.count))


                counter = 0
                train_preds = []
                for x_tr in range(0, len(x_train_full_window)):
                    temp_pred_train = []
                    for x in range(0, len(x_train) // len(x_train_full_window)): # 7 sub windows for  each windows
                        temp_pred_train.append(model.predict([x_train[counter]]).tolist()[0])
                        counter += 1
                    train_preds.append(max(set(temp_pred_train), key=temp_pred.count))
                train_predictions.append(train_preds)

                y_train = y_train_full_window
            else:
                x_test = feats[sample]
                y_test = labels[sample]
                x_train = [x for i, x in enumerate(feats) if i != sample]
                y_train = [x for i, x in enumerate(labels) if i != sample]

                model.fit(x_train, y_train)
                test_predictions.append(model.predict([x_test]).tolist()[0])

                train_preds = model.predict(x_train)
                train_predictions.append(train_preds.tolist())

            train_labels.append(y_train)
            test_labels.append(y_test)
            temp_scores[sample, 0] = accuracy(y_train, train_preds)
            temp_scores[sample, 1] = precision(y_train, train_preds)
            temp_scores[sample, 2] = recall(y_train, train_preds)
            temp_scores[sample, 3] = f1(y_train, train_preds)

        ensemble_predictions_train.append(train_predictions)
        ensemble_predictions_test.append(test_predictions)

        score_df[channel] = [np.mean(temp_scores[:, 0]),
                             accuracy(test_labels, test_predictions),
                             np.mean(temp_scores[:, 1]),
                             precision(test_labels, test_predictions),
                             np.mean(temp_scores[:, 2]),
                             recall(test_labels, test_predictions),
                             np.mean(temp_scores[:, 3]),
                             f1(test_labels, test_predictions),
                             ]

        if save_model:
            path = os.path.join(OUTPUT_PATH, 'models', dir_name)
            create_dir(path, recursive=True)
            filename = os.path.join(path, dir_name + str(channel))
            pickle.dump(model, open(filename, 'wb'))

    score_df['average'] = score_df.mean(numeric_only=True, axis=1)

    ensemble_preds_test = combine_predictions(ensemble_predictions_test).astype(int)
    ensemble_acc, ensemble_precision, ensemble_recall, ensemble_f1 = combine_loocv_predictions(train_labels,
                                                                                               ensemble_predictions_train)

    score_df['ensemble'] = [ensemble_acc,
                            accuracy(test_labels, ensemble_preds_test),
                            ensemble_precision,
                            precision(test_labels, ensemble_preds_test),
                            ensemble_recall,
                            recall(test_labels, ensemble_preds_test),
                            ensemble_f1,
                            f1(test_labels, ensemble_preds_test),
                            ]

    return score_df
