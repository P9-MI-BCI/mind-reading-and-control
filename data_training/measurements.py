import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def get_accuracy(targets, preds):
    counter = 0
    for pred, target in zip(preds, targets):
        if pred == target:
            counter += 1

    return counter / len(preds)


def get_precision(targets, preds):
    return precision_score(targets, preds)


def get_recall(targets, preds):
    return recall_score(targets, preds)


def get_f1_score(targets, preds):
    return f1_score(targets, preds)


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


def combine_loocv_predictions(all_channel_predictions):

    all_preds = np.array(all_channel_predictions)

    pass