import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import (
    Activation,
    Add,
    Conv1D,
    Dense,
    Flatten,
    Input,
    MaxPooling1D,
    Softmax,
)
from tensorflow.keras.models import Model


def create_cnn_model() -> tf.keras.Sequential:
    """Creates 1D CNN model.

    Returns:
        tf.keras.Sequential: A 1D CNN model.
    """
    model = Sequential(
        [
            layers.Convolution1D(filters=16, kernel_size=7, input_shape=[186, 1]),
            layers.LeakyReLU(),
            layers.MaxPool1D(pool_size=2),
            layers.Convolution1D(filters=16, kernel_size=5),
            layers.LeakyReLU(),
            layers.Convolution1D(filters=16, kernel_size=5),
            layers.LeakyReLU(),
            layers.Convolution1D(filters=16, kernel_size=5),
            layers.LeakyReLU(),
            layers.MaxPool1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(128),
            layers.LeakyReLU(),
            layers.Dense(5, activation="softmax"),
        ]
    )
    return model


def create_dense_model() -> tf.keras.Sequential:
    """Creates ANN model.

    Returns:
        tf.keras.Sequential: An ANN model.
    """
    model = Sequential(
        [
            layers.InputLayer(input_shape=[186, 1]),
            layers.Flatten(),
            layers.Dense(
                50,
                activation="relu",
            ),
            layers.Dense(50, activation="relu"),
            layers.Dense(50, activation="relu"),
            layers.Dense(5, activation="softmax"),
        ]
    )

    return model


def create_softmax_model() -> tf.keras.Sequential:
    """Creates a softmax regression model.

    Returns:
        tf.keras.Sequential: A softmax regresion model.
    """
    model = Sequential(
        [
            layers.InputLayer(input_shape=[186, 1]),
            layers.Flatten(),
            layers.Dense(5),
        ]
    )
    return model
