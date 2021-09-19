from sklearn.neighbors import KNeighborsClassifier
from data_preprocessing.train_test_split import format_dataset


def channel_accuracy(train_data, test_data):
    channels = len(train_data[0][1].columns)
    score_dict = {}
    ensemble_predictions_train = []
    ensemble_predictions_test = []

    neigh = KNeighborsClassifier(n_neighbors=2)
    _, y_train = format_dataset(train_data, channel=0)
    _, y_test = format_dataset(test_data, channel=0)

    for channel in range(channels):
        x_train, _ = format_dataset(train_data, channel=channel)
        x_test, _ = format_dataset(test_data, channel=channel)

        neigh.fit(x_train, y_train)

        train_preds = neigh.predict(x_train)
        test_preds = neigh.predict(x_test)

        ensemble_predictions_train.append(train_preds)
        ensemble_predictions_test.append(test_preds)

        train_score = neigh.score(x_train, y_train)
        test_score = neigh.score(x_test, y_test)

        score_dict[channel] = {'train_score': train_score, 'test_score': test_score}

    ensemble_preds_train = combine_predictions(ensemble_predictions_train).astype(int)
    ensemble_preds_test = combine_predictions(ensemble_predictions_test).astype(int)

    score_dict['ensemble'] = {'train_score': get_accuracy(ensemble_preds_train, y_train),
                              'test_score': get_accuracy(ensemble_preds_test, y_test)}
    return score_dict


def get_accuracy(preds, targets):
    counter = 0
    for pred, target in zip(preds, targets):
        if pred == target:
            counter += 1

    return (counter / len(preds)) * 100


def combine_predictions(all_channel_predictions):
    channel_lst = []
    c_df = pd.DataFrame()
    counter = 0

    for channel in all_channel_predictions:
        c_df[counter] = channel
        counter += 1

    most_frequent_pred = c_df.mode(axis=1)
    return most_frequent_pred[0].to_numpy()