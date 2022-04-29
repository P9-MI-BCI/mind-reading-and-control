import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping
import pandas as pd

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 3
image_length = 100
image_height = 3
num_patches = 168
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]

data_augmentation = keras.Sequential(
    [
        layers.Resizing(9, 2400),
    ],
    name="data_augmentation",
)


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, height, length):
        super(Patches, self).__init__()
        self.height = height
        self.length = length

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.height, self.length, 1],
            strides=[1, 1, self.length, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def create_vit_classifier(input_shape):
    inputs = layers.Input(shape=input_shape)

    augmented = data_augmentation(inputs)

    patches = Patches(image_height, image_length)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(1)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


def run_experiment(model, x_train, y_train, x_test, y_test):

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate
                                        ),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    checkpoint_filepath = "../checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    x_train = x_train.reshape((x_train.shape[0], x_train[0].shape[1], x_train[0].shape[0], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test[0].shape[1], x_test[0].shape[0], 1))
    model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        batch_size=batch_size,
        epochs=num_epochs,
        callbacks=[checkpoint_callback,
                   EarlyStopping(monitor='val_loss', patience=10)],
    )

    model.load_weights(checkpoint_filepath)
    accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {accuracy}%")

    return model, accuracy


def transformer(X, Y):
    Y = np.array(Y)
    input_shape = (9, 2400, 1)
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    cv_scores = {}
    split = 0
    for train_index, val_index in skf.split(X, Y):
        model = create_vit_classifier(input_shape)
        model, accuracy = run_experiment(model, X[train_index], Y[train_index], X[val_index], Y[val_index])
        cv_scores[f'split_{split}'] = accuracy
        split += 1

    cv_scores = pd.DataFrame(cv_scores, index=[0])
    cv_scores['mean'] = cv_scores.mean(axis=1)
    print(f'{cv_scores}')
    return model
