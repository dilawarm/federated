import collections
import functools

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
from keras.utils.np_utils import to_categorical
from sklearn.utils import resample

SAMPLES = 20_000
NUM_OF_CLIENTS = 10

transform_data = lambda wave: wave + np.random.normal(0, 0.5, 186)  # Data augmentation

"""
Function seperates label column from datapoints columns.
Returns tuple: dataframe without label, labels
"""
split_dataframe = lambda df: (
    df.iloc[:, :186].values,
    to_categorical(df[187]).astype(int),
)


def create_dataset(X, y):
    """
    Function converts pandas dataframe to tensorflow federated.
    Returns dataset of type tff.simulation.ClientData
    """
    num_of_clients = NUM_OF_CLIENTS
    total_ecg_count = len(X)
    ecgs_per_set = int(np.floor(total_ecg_count / num_of_clients))

    client_dataset = collections.OrderedDict()
    for i in range(1, num_of_clients + 1):
        name_of_client = f"client_{i}"
        start = ecgs_per_set * (i - 1)
        end = ecgs_per_set * i
        data = collections.OrderedDict(
            (("label", y[start:end]), ("datapoints", X[start:end]))
        )
        client_dataset[name_of_client] = data

    return tff.simulation.FromTensorSlicesClientData(client_dataset)


def create_class_distributed_dataset(X, y):
    n = len(X)
    clients_data = dict(
        [
            ("client_1", [[], []]),
            ("client_2", [[], []]),
            ("client_3", [[], []]),
            ("client_4", [[], []]),
            ("client_5", [[], []]),
        ]
    )

    for i in range(n):
        index = np.where(y[i] == 1)[0][0]
        clients_data[f"client_{index+1}"][0].append(X[i])
        clients_data[f"client_{index+1}"][1].append(y[i])

    client_dataset = collections.OrderedDict()

    for client in clients_data:
        data = collections.OrderedDict(
            (
                ("label", np.array(clients_data[client][1], dtype=np.int64)),
                ("datapoints", np.array(clients_data[client][0], dtype=np.float64)),
            )
        )
        client_dataset[client] = data

    return tff.simulation.FromTensorSlicesClientData(client_dataset)


def load_data(
    normalized=False, data_analysis=False, transform=False, data_selector=None
):
    """
    Function loads data from csv-file
    and preprocesses the training and test data seperately.
    Returns a tuple of tff.simulation.ClientData
    """
    train_file = "data/mitbih/mitbih_train.csv"
    test_file = "data/mitbih/mitbih_test.csv"

    if data_analysis:
        train_file = "../" + train_file
        test_file = "../" + test_file

    train_df = pd.read_csv(train_file, header=None)
    test_df = pd.read_csv(test_file, header=None)

    train_df[187], test_df[187] = (
        train_df[187].astype(int),
        test_df[187].astype(int),
    )

    if normalized:
        # From the data analysis, we can see that the normal heartbeats are overrepresented in the dataset
        df_0 = (train_df[train_df[187] == 0]).sample(n=SAMPLES, random_state=42)
        train_df = pd.concat(
            [df_0]
            + [
                resample(
                    train_df[train_df[187] == i],
                    replace=True,
                    n_samples=SAMPLES,
                    random_state=int(f"12{i+2}"),
                )
                for i in range(1, 5)
            ]
        )

    train_X, train_y = split_dataframe(train_df)
    test_X, test_y = split_dataframe(test_df)

    if data_analysis:
        return test_X, test_y

    if transform:
        for i in range(len(train_X)):
            train_X[i, :186] = transform_data(train_X[i, :186])

    return (
        data_selector(train_X, train_y),
        data_selector(test_X, test_y),
    )


def preprocess_dataset(epochs, batch_size, shuffle_buffer_size):
    """
    Function returns a function for preprocessing of a dataset
    """

    def _reshape(element):
        """
        Function returns reshaped tensors
        """

        return (tf.expand_dims(element["datapoints"], axis=-1), element["label"])

    @tff.tf_computation(
        tff.SequenceType(
            collections.OrderedDict(
                label=tff.TensorType(tf.int64, shape=(5,)),
                datapoints=tff.TensorType(tf.float64, shape=(186,)),
            )
        )
    )
    def preprocess(dataset):
        """
        Function returns shuffled dataset
        """
        return (
            dataset.shuffle(shuffle_buffer_size)
            .repeat(epochs)
            .batch(batch_size, drop_remainder=False)
            .map(_reshape, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        )

    return preprocess


def get_datasets(
    train_batch_size=32,
    test_batch_size=32,
    train_shuffle_buffer_size=10000,
    test_shuffle_buffer_size=10000,
    train_epochs=5,
    test_epochs=5,
    transform=False,
    centralized=False,
    normalized=True,
    data_selector=create_dataset,
):

    """
    Function preprocesses datasets.
    Return input-ready datasets
    """
    train_dataset, test_dataset = load_data(
        normalized=normalized, transform=transform, data_selector=data_selector
    )

    if centralized:
        train_dataset, test_dataset = (
            train_dataset.create_tf_dataset_from_all_clients(),
            test_dataset.create_tf_dataset_from_all_clients(),
        )

    train_preprocess = preprocess_dataset(
        epochs=train_epochs,
        batch_size=train_batch_size,
        shuffle_buffer_size=train_shuffle_buffer_size,
    )

    test_preprocess = preprocess_dataset(
        epochs=test_epochs,
        batch_size=test_batch_size,
        shuffle_buffer_size=test_shuffle_buffer_size,
    )

    if centralized:
        train_dataset = train_preprocess(train_dataset)
        test_dataset = test_preprocess(test_dataset)
    else:
        train_dataset = train_dataset.preprocess(train_preprocess)
        test_dataset = test_dataset.preprocess(test_preprocess)

    return train_dataset, test_dataset


def randomly_select_clients_for_round(
    population, num_of_clients, replace=False, seed=None
):
    """
    This function creates a partial function for sampling random clients.
    Returns a partial object.
    """

    def select(round_number, seed, replace):
        """
        Function for selecting random clients.
        Returns a random sample from the client id's.
        """
        return np.random.RandomState().choice(
            population, num_of_clients, replace=replace
        )

    return functools.partial(select, seed=seed, replace=False)


def get_client_dataset_fn(
    dataset,
    number_of_clients_per_round,
    seed=None,
):
    """
    This function generates a function for selecting client-datasets for each round number.
    Returns a function for choosing clients while training.
    """
    sample_clients = randomly_select_clients_for_round(
        dataset.client_ids,
        num_of_clients=number_of_clients_per_round,
        replace=False,
        seed=seed,
    )

    def get_dataset_for_client(round_number):
        """
        This function chooses the client datasets.
        Returns a list of client datasets.
        """
        clients = sample_clients(round_number)
        return [dataset.create_tf_dataset_for_client(client) for client in clients]

    return get_dataset_for_client


def _convert_fn(dataset):
    """
    Converts dataset to tupled dataset.
    Returns tupled dataset.
    """
    spec = dataset.element_spec
    if isinstance(spec, collections.abc.Mapping):
        return dataset.map(lambda observation: (observation["x"], observation["y"]))
    else:
        return dataset.map(lambda x, y: (x, y))


def get_validation_fn(test_dataset, model_fn, loss_fn, metrics_fn):
    """
    This function makes a function for evaluating a model while training.
    Returns a validation function.
    """

    def compiled_model():
        """
        This function compiles an 'empty' model.
        Returns a Keras Model object.
        """
        val_model = model_fn()
        val_model.compile(
            loss=loss_fn(), optimizer=tf.keras.optimizers.Adam(), metrics=metrics_fn()
        )
        return val_model

    test_dataset = _convert_fn(test_dataset)

    def validation_fn(trained_model):
        """
        Validates the model by running model.evaluate() on the keras model with the weights from a state from the interactive process.
        Returns the metrics after evaluation.
        """
        val_model = compiled_model()
        trained_model_weights = tff.learning.ModelWeights(
            trainable=list(trained_model.trainable),
            non_trainable=list(trained_model.non_trainable),
        )

        trained_model_weights.assign_weights_to(val_model)
        metrics = val_model.evaluate(test_dataset, verbose=0)
        return dict(
            zip(val_model.metrics_names, val_model.evaluate(test_dataset, verbose=0))
        )

    return validation_fn


if __name__ == "__main__":
    load_data(
        normalized=False,
        data_analysis=False,
        transform=False,
        data_selector=create_class_distributed_dataset,
    )
