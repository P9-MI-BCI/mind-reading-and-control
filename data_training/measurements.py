import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


def accuracy(targets, preds):
    counter = 0
    for pred, target in zip(preds, targets):
        if pred == target:
            counter += 1

    return counter / len(preds)


def precision(targets, preds):
    return precision_score(targets, preds, zero_division=1)


def recall(targets, preds):
    return recall_score(targets, preds, zero_division=1)


def f1(targets, preds):
    return f1_score(targets, preds, zero_division=1)


def get_confusion_matrix(targets, preds):
    return confusion_matrix(targets, preds)


# combines predictions for all channels based on the most frequent prediction
def combine_predictions(all_channel_predictions):
    c_df = pd.DataFrame()
    counter = 0

    for channel in all_channel_predictions:
        c_df[counter] = channel
        counter += 1

    most_frequent_pred = c_df.mode(axis=1)
    return most_frequent_pred[0].to_numpy()


def combine_loocv_predictions(labels, all_channel_predictions):

    all_preds = np.array(all_channel_predictions)

    accs = []
    prec = []
    recall_score = []
    f1_score = []
    for sample in range(0, all_preds.shape[1]):
        new_arr = []
        for channel in range(0, all_preds.shape[0]):
            new_arr.append(all_preds[channel][sample])

        combined = combine_predictions(new_arr)

        accs.append(accuracy(labels[sample], combined))
        prec.append(precision(labels[sample], combined))
        recall_score.append(recall(labels[sample], combined))
        f1_score.append(f1(labels[sample], combined))

    return np.mean(accs), np.mean(prec), np.mean(recall_score), np.mean(f1_score)