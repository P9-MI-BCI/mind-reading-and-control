import pandas as pd


def get_accuracy(preds, targets):
    counter = 0
    for pred, target in zip(preds, targets):
        if pred == target:
            counter += 1

    return (counter / len(preds)) * 100


def combine_predictions(all_channel_predictions):
    c_df = pd.DataFrame()
    counter = 0

    for channel in all_channel_predictions:
        c_df[counter] = channel
        counter += 1

    most_frequent_pred = c_df.mode(axis=1)
    return most_frequent_pred[0].to_numpy()
