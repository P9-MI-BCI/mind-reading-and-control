from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from wandb.integration.keras import WandbCallback

from classes.Dataset import Dataset
from data_preprocessing.filters import butter_filter
from data_preprocessing.mrcp_detection import load_index_list, pair_index_list
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from tcn import TCN, tcn_full_summary
from tensorflow.keras.models import Sequential
import tensorflow as tf

import wandb

wandb.init(project="my-test-project", entity="cryppen")


wandb.config = {
  "epochs": 50,
  "batch_size": 16
}

def data_preparation(dataset: Dataset, config):
    # given dataset use the index list to label each freq.
    # use the same datashift as was used for the index
    index_location = "C:\\p9_data\\output\\cue_set1\\index"
    raw_index = load_index_list(index_location)
    new_index = []
    i = 0
    while i < len(raw_index):
        if i % dataset.sample_rate == 0:
            i+= dataset.sample_rate
        new_index.append(raw_index[i])
        i += 1

    new_index = np.array(new_index)
    # new_index -= 600

    pair_indexes = pair_index_list(raw_index)
    center_index_list = []
    for pair in pair_indexes:
        center_index_list.append(int(sum(pair)/len(pair)))

    seed = 3407  # ref https://arxiv.org/abs/2109.08203
    tf.random.set_seed(seed)

    data = dataset.data_device1.iloc[:, config.EEG_Channels]

    data['label'] = np.zeros(len(data), dtype=np.int)

    data['label'].iloc[new_index] = 1

    # data_centers = data[data['label'] == 1].iloc[:,config.EEG_Channels].values

    step_size = int(dataset.sample_rate / 10)

    window_size = dataset.sample_rate

    labels = data['label'].values
    # labels_one_hot = to_categorical(labels, num_classes=2)

    for i in config.EEG_Channels:
        data[i] = butter_filter(data=data.iloc[:,i],
                                order=config.eeg_order,
                                cutoff=config.eeg_cutoff,
                                btype=config.eeg_btype
                                )

    data = data.drop(columns=['label'], axis=1).values

    num_training_samples = int(len(data) * 0.8)
    num_validation_samples = int(len(data) * 0.1)
    num_test_samples = len(data) - num_validation_samples - num_training_samples

    print(num_training_samples)
    print(num_validation_samples)
    print(num_test_samples)

    # normalization
    # mean = data[:num_training_samples].mean(axis=0)
    # data -= mean
    # std = data[:num_training_samples].std(axis=0)
    # data /= std

    scaler = StandardScaler()
    scaler.fit(data[:num_training_samples])

    data = scaler.transform(data)

    sampling_rate = 1
    sequence_length = window_size
    batch_size = 16

    num_labels = np.bincount(labels[:num_training_samples])
    num_val_labels = np.bincount(labels[num_training_samples:-num_test_samples])
    num_test_labels = np.bincount(labels[-num_test_samples:])
    print(f'training distribution labels: {num_labels[1]/sum(num_labels)}')
    print(f'validation distribution: {num_val_labels[1]/sum(num_val_labels)}')
    print(f'test distribution: {num_test_labels[1]/sum(num_test_labels)}')

    def create_dataset(in_data, in_labels, s_size):
        vecs = []
        labls = []
        for d in range(0, len(in_data), s_size):
            vecs.append(in_data[d:d+sequence_length])
            labls.append(in_labels[d])

        np.random.shuffle(vecs)
        np.random.shuffle(labls)

        dis_vecs = []
        dis_labels = []
        for y in range(len(labls)):
            if labls[y] == 1:
                dis_vecs.append(vecs[y])
                dis_labels.append(labls[y])

        for i in range(len(dis_labels)):
            for x in range(len(labls)):
                if labls[x] == 0:
                    dis_vecs.append(vecs[x])
                    dis_labels.append(labls[x])

                    del vecs[x]
                    del labls[x]
                    break

        return dis_vecs, dis_labels

    x, y = create_dataset(data, labels, step_size)

    np.random.shuffle(x)
    np.random.shuffle(y)
    x = np.array(x)
    y = np.array(y)
    num_training_samples = int(len(x) * 0.8)
    num_validation_samples = int(len(x) * 0.1)
    num_test_samples = len(x) - num_validation_samples - num_training_samples

    # train_dataset = keras.utils.timeseries_dataset_from_array(
    #     data[:],
    #     targets=labels[:],
    #     sampling_rate=sampling_rate,
    #     sequence_stride=step_size,
    #     sequence_length=sequence_length,
    #     shuffle=True,
    #     batch_size=batch_size,
    #     start_index=0,
    #     seed=seed,
    #     end_index=num_training_samples)
    #
    # val_dataset = keras.utils.timeseries_dataset_from_array(
    #     data[:],
    #     targets=labels[:],
    #     sampling_rate=sampling_rate,
    #     sequence_stride=step_size,
    #     sequence_length=sequence_length,
    #     shuffle=True,
    #     batch_size=batch_size,
    #     seed=seed,
    #     start_index=num_training_samples,
    #     end_index=num_training_samples + num_validation_samples)
    #
    # test_dataset = keras.utils.timeseries_dataset_from_array(
    #     data[:],
    #     targets=labels[:],
    #     sampling_rate=sampling_rate,
    #     sequence_stride=step_size,
    #     sequence_length=sequence_length,
    #     shuffle=True,
    #     batch_size=batch_size,
    #     seed=seed,
    #     start_index=num_training_samples + num_validation_samples)
    #
    # for samples, targets in train_dataset:
    #     print(targets)
    #     print("samples shape:", samples.shape)
    #     print("targets shape:", targets.shape)
    #     break

    tcn_layer1 = TCN(nb_filters=128,
                     kernel_size=9,
                     nb_stacks=1,
                     dilations=(1, 2, 4, 8, 16, 32),
                     padding='causal',
                     use_skip_connections=True,
                     dropout_rate=0.05,
                     return_sequences=False,
                     activation='relu',
                     kernel_initializer='glorot_uniform',
                     use_batch_norm=False,
                     use_layer_norm=False,
                     use_weight_norm=False,
                     input_shape=(sequence_length, data.shape[-1])
                     )

    # The receptive field tells you how far the model can see in terms of timesteps.
    print('Receptive field size =', tcn_layer1.receptive_field)

    callbacks = [
        keras.callbacks.ModelCheckpoint("jena_dense.keras",
                                        save_best_only=True),
        WandbCallback()
    ]


    model = Sequential([
        tcn_layer1,
        # layers.Dense(64, activation='relu'),
        # layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=["accuracy"])
    history = model.fit(x,
                        y,
                        batch_size=16,
                        epochs=50,
                        validation_split=0.2,
                        shuffle=True,
                        callbacks=callbacks)

    # model = keras.models.load_model("jena_dense.keras")
    print(f"Test accuracy: {model.evaluate(x[num_test_samples:])[1]:.2f}")

    print(model.predict(x[num_test_samples:])[0])
    loss = history.history["accuracy"]
    val_loss = history.history["val_accuracy"]
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training accuracy")
    plt.plot(epochs, val_loss, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.show()
