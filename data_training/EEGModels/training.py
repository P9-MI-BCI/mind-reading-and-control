"""
Prepare data format for using the EEG models and set up dispatch functions for the three different preloadeded EEG models.
"""
import numpy as np
from utility.logger import get_logger
from data_training.EEGModels.EEG_Models import EEGNet, DeepConvNet, ShallowConvNet

kernels = 1


def EEGModels_training_hub(X, Y):
    run_EEGNet(X, Y)
    run_DeepConvNet(X, Y)
    run_ShallowConvNet(X, Y)


"""
Given shuffled data and labels split the data int
70% training data
15% validation data
15% test data
while retaining same label distribution.
"""


def train_val_test_split(X, Y):
    try:
        X_train, Y_train = [], []
        X_val, Y_val = [], []
        X_test, Y_test = [], []

        for x, y in zip(X, Y):
            if len(X_train) < len(X) * 0.7:
                X_train.append(x)
                Y_train.append(y)
            elif len(X_val) < len(X) * 0.15:
                X_val.append(x)
                Y_val.append(y)
            else:
                X_test.append(x)
                Y_test.append(y)

        return np.array(X_train), np.array(Y_train), np.array(X_val), np.array(Y_val), np.array(X_test), np.array(
            Y_test)
    except AssertionError:
        get_logger().error('The dataset does not contain an even distribution. Unable to split the dataset evenly')


def run_EEGNet(X, Y):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = train_val_test_split(X, Y)

    # Reshape data into (Num_samples, Num_channels, Num_data_points, kernel=1)
    X_train = X_train.reshape((X_train.shape[0], X_train[0].shape[1], X_train[0].shape[0], kernels))
    X_val = X_val.reshape((X_val.shape[0], X_val[0].shape[1], X_val[0].shape[0], kernels))
    X_test = X_test.reshape((X_test.shape[0], X_test[0].shape[1], X_test[0].shape[0], kernels))

    model = EEGNet(nb_classes=1, Chans=X_train[0].shape[0], Samples=X_train[0].shape[1], dropoutRate=0.5, kernLength=32,
                   F1=8, D=2, F2=16, dropoutType='Dropout')

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    numParams = model.count_params()

    model.fit(X_train, Y_train, batch_size=16, epochs=300,
              verbose=2, validation_data=(X_val, Y_val),
              )

    print(model.predict(X_test))
    print(Y_test)


def run_DeepConvNet(X, Y):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = train_val_test_split(X, Y)

    # Reshape data into (Num_samples, Num_channels, Num_data_points, kernel=1)
    X_train = X_train.reshape((X_train.shape[0], X_train[0].shape[1], X_train[0].shape[0], kernels))
    X_val = X_val.reshape((X_val.shape[0], X_val[0].shape[1], X_val[0].shape[0], kernels))
    X_test = X_test.reshape((X_test.shape[0], X_test[0].shape[1], X_test[0].shape[0], kernels))

    model = DeepConvNet(nb_classes=1, Chans=X_train[0].shape[0], Samples=X_train[0].shape[1], dropoutRate=0.5)

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    numParams = model.count_params()

    model.fit(X_train, Y_train, batch_size=16, epochs=300,
              verbose=2, validation_data=(X_val, Y_val),
              )

    print(model.predict(X_test))
    print(Y_test)


def run_ShallowConvNet(X, Y):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = train_val_test_split(X, Y)

    # Reshape data into (Num_samples, Num_channels, Num_data_points, kernel=1)
    X_train = X_train.reshape((X_train.shape[0], X_train[0].shape[1], X_train[0].shape[0], kernels))
    X_val = X_val.reshape((X_val.shape[0], X_val[0].shape[1], X_val[0].shape[0], kernels))
    X_test = X_test.reshape((X_test.shape[0], X_test[0].shape[1], X_test[0].shape[0], kernels))

    model = ShallowConvNet(nb_classes=1, Chans=X_train[0].shape[0], Samples=X_train[0].shape[1], dropoutRate=0.5)

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    numParams = model.count_params()

    model.fit(X_train, Y_train, batch_size=16, epochs=300,
              verbose=2, validation_data=(X_val, Y_val),
              )

    print(model.predict(X_test))
    print(Y_test)

