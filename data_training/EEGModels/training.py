"""
Prepare data format for using the EEG models and set up dispatch functions for the three different preloadeded EEG models.
"""
import numpy as np
from utility.logger import get_logger
from data_training.EEGModels.EEG_Models import EEGNet, DeepConvNet, ShallowConvNet
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from tqdm.keras import TqdmCallback
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn

kernels = 1


def EEGModels_training_hub(X, Y, online_X, online_Y):
    eeg_net = get_EEGNet(X)
    deep_conv_net = get_DeepConvNet(X)
    shallow_conv_net = get_ShallowConvNet(X)

    stratified_kfold_cv(X, Y, eeg_net, online_X, online_Y)
    stratified_kfold_cv(X, Y, deep_conv_net, online_X, online_Y)
    stratified_kfold_cv(X, Y, shallow_conv_net, online_X, online_Y)


# Creates a stratified-k-fold cross validation object that splits the data and shuffles it.
def stratified_kfold_cv(X, Y, model, online_X, online_Y):
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    # The initial weights are saved in order to reset the model between k-fold cv iterations
    model.save_weights('initial_weights.h5')
    for train_index, val_index in skf.split(X, Y):
        model.load_weights('initial_weights.h5')
        # Reshape data into (Num_samples, Num_channels, Num_data_points, kernel=1)
        X_reshaped = X[train_index].reshape((X[train_index].shape[0], X[0].shape[1], X[0].shape[0], kernels))
        X_val_reshaped = X[val_index].reshape((X[val_index].shape[0], X[0].shape[1], X[0].shape[0], kernels))
        history = model.fit(X_reshaped, Y[train_index], batch_size=8, epochs=300,
                            verbose=0, validation_data=(X_val_reshaped, Y[val_index]),
                            callbacks=[TqdmCallback(verbose=0),
                                       EarlyStopping(monitor='val_loss', patience=30)])

    X_reshaped = X.reshape((X.shape[0], X[0].shape[1], X[0].shape[0], kernels))
    X_online_reshaped = online_X.reshape((online_X.shape[0], online_X[0].shape[1], online_X[0].shape[0], kernels))

    get_logger().info(f'Cross Validation Finished -- Training using entire dataset')
    history = model.fit(X_reshaped, Y, batch_size=8, epochs=300,verbose=0,
                        validation_data=(X_online_reshaped, online_Y),
                        callbacks=[TqdmCallback(verbose=0),
                        EarlyStopping(monitor='val_loss', patience=50)])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()

    Y_hat = model.predict(X_online_reshaped)
    Y_hat = Y_hat.reshape((len(Y_hat),))
    plot_confusion_matrix(online_Y, Y_hat)


def get_EEGNet(X):
    model = EEGNet(nb_classes=1, Chans=X[0].shape[1], Samples=X[0].shape[0], dropoutRate=0.5, kernLength=32,
                   F1=8, D=2, F2=16, dropoutType='Dropout')

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    return model


def get_DeepConvNet(X):
    model = DeepConvNet(nb_classes=1, Chans=X[0].shape[1], Samples=X[0].shape[0], dropoutRate=0.5)

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    return model

def get_ShallowConvNet(X):
    model = ShallowConvNet(nb_classes=1, Chans=X[0].shape[1], Samples=X[0].shape[0], dropoutRate=0.5)

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    return model


def plot_confusion_matrix(Y, Y_hat):
    Y_hat = [round(y_hat) for y_hat in Y_hat]
    labels = ['close', 'open']
    conf_matrix = confusion_matrix(Y, Y_hat)

    df_cm = pd.DataFrame(conf_matrix)

    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size

    plt.show()