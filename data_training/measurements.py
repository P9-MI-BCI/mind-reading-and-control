import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def get_accuracy(preds, targets):
    counter = 0
    for pred, target in zip(preds, targets):
        if pred == target:
            counter += 1

    return (counter / len(preds))


def get_precision(preds, targets):
    return precision_score(preds,targets)


def get_recall(preds, targets):
    return recall_score(preds, targets)


def get_f1_score(preds, targets):
    return f1_score(preds, targets)


def get_confusion_matrix(preds, targets):
    return confusion_matrix(preds, targets)


def combine_predictions(all_channel_predictions):
    c_df = pd.DataFrame()
    counter = 0

    for channel in all_channel_predictions:
        c_df[counter] = channel
        counter += 1

    most_frequent_pred = c_df.mode(axis=1)
    return most_frequent_pred[0].to_numpy()
