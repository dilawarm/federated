import collections
import functools
import pickle
import random
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.keras.utils import to_categorical
from sklearn.utils import resample

SAMPLES = 20_000
NUM_OF_CLIENTS = 10
S = 100

split_dataframe = lambda df: (
    df.iloc[:, :186].values,
    to_categorical(df[187]).astype(int),
)
split_dataframe.__doc__ = (
    """Function for splitting dataframe into (input, output) pairs."""
)


def create_dataset(
    X: np.ndarray, y: np.ndarray, number_of_clients: int
) -> [None, tff.simulation.ClientData]:
    """Function converts pandas dataframe to tensorflow federated dataset.

    Args:
        X (np.ndarray): Inputs.\n
        y (np.ndarray): Outputs.\n
        number_of_clients (int): The number of clients to split the data between.\n

    Returns:
        [None, tff.simulation.ClientData]: Returns federated data distribution.
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
            (
                ("label", np.array(y[start:end], dtype=np.int32)),
                ("datapoints", np.array(X[start:end], dtype=np.float32)),
            )
        )
        client_dataset[name_of_client] = data

    return None, tff.simulation.FromTensorSlicesClientData(client_dataset)


def create_tff_dataset(clients_data: Dict) -> tff.simulation.ClientData:
    """Function converts dictionary to tensorflow federated dataset.

    Args:
        clients_data (Dict): Inputs.

    Returns:
        tff.simulation.ClientData: Returns federated data distribution.
    """
    client_dataset = collections.OrderedDict()

    for client in clients_data:
        data = collections.OrderedDict(
            (
                ("label", np.array(clients_data[client][1], dtype=np.int32)),
                ("datapoints", np.array(clients_data[client][0], dtype=np.float32)),
            )
        )
        client_dataset[client] = data

    return tff.simulation.FromTensorSlicesClientData(client_dataset)


def create_class_distributed_dataset(
    X: np.ndarray, y: np.ndarray, number_of_clients: int
) -> [Dict, tff.simulation.ClientData]:
    """Function distributes the data in a way such that each client gets one type of data.

    Args:
        X (np.ndarray): Input.\n
        y (np.ndarray): Output.\n
        number_of_clients (int): Number of clients.\n
    Returns:
        [Dict, tff.simulation.ClientData]: A dictionary and a tensorflow federated dataset containing the distributed dataset.
    """
    n = len(X)
    clients_data = {f"client_{i}": [[], []] for i in range(1, 6)}

    for i in range(n):
        index = np.where(y[i] == 1)[0][0]
        clients_data[f"client_{index+1}"][0].append(X[i])
        clients_data[f"client_{index+1}"][1].append(y[i])

    return clients_data, create_tff_dataset(clients_data)


def create_uniform_dataset(
    X: np.ndarray, y: np.ndarray, number_of_clients: int
) -> [Dict, tff.simulation.ClientData]:
    """Function distributes the data equally such that each client holds equal amounts of each class.

    Args:
        X (np.ndarray): Input.\n
        y (np.ndarray): Output.\n
        number_of_clients (int): Number of clients.\n
    Returns:
        [Dict, tff.simulation.ClientData]: A dictionary and a tensorflow federated dataset containing the distributed dataset.
    """
    clients_data = {f"client_{i}": [[], []] for i in range(1, number_of_clients + 1)}
    for i in range(len(X)):
        clients_data[f"client_{(i%number_of_clients)+1}"][0].append(X[i])
        clients_data[f"client_{(i%number_of_clients)+1}"][1].append(y[i])

    return clients_data, create_tff_dataset(clients_data)


def create_unbalanced_data(
    X: np.ndarray, y: np.ndarray, number_of_clients: int
) -> [Dict, tff.simulation.ClientData]:
    """Function distributes the data in such a way that one client only has one type of data, while the rest of the clients has non-iid data.

    Args:
        X (np.ndarray): Input.\n
        y (np.ndarray): Output.\n
        number_of_clients (int): Number of clients.\n
    Returns:
        [Dict, tff.simulation.ClientData]: A dictionary and a tensorflow federated dataset containing the distributed dataset.
    """
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    clients_data = {f"client_{i}": [[], []] for i in range(1, number_of_clients + 1)}
    for i in range(len(X)):
        if np.where(y[i] == 1)[0][0] == 0:
            clients_data[f"client_{1}"][0].append(X[i])
            clients_data[f"client_{1}"][1].append(y[i])
        else:
            client = random.choice([i for i in range(1, 6) if i not in [1]])
            clients_data[f"client_{client}"][0].append(X[i])
            clients_data[f"client_{client}"][1].append(y[i])

    return clients_data, create_tff_dataset(clients_data)


def create_non_iid_dataset(
    X: np.ndarray, y: np.ndarray, number_of_clients: int
) -> [Dict, tff.simulation.ClientData]:
    """Function distributes the data such that each client has non-iid data.

    Args:
        X (np.ndarray): Input.\n
        y (np.ndarray): Output.\n
        number_of_clients (int): Number of clients.\n
    Returns:
        [Dict, tff.simulation.ClientData]: A dictionary and a tensorflow federated dataset containing the distributed dataset.
    """
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    clients_data = {f"client_{i}": [[], []] for i in range(1, number_of_clients + 1)}
    for i in range(len(X)):
        client = random.randrange(
            1, number_of_clients + 1, np.random.randint(1, number_of_clients + 1)
        )
        clients_data[f"client_{client}"][0].append(X[i])
        clients_data[f"client_{client}"][1].append(y[i])

    return clients_data, create_tff_dataset(clients_data)


def create_corrupted_non_iid_dataset(
    X: np.ndarray, y: np.ndarray, number_of_clients: int
) -> [Dict, tff.simulation.ClientData]:
    """Function distributes the data such that each client has non-iid data except client 1, which only has values in the interval [20, 40].

    Args:
        X (np.ndarray): Input.\n
        y (np.ndarray): Output.\n
        number_of_clients (int): Number of clients.\n
    Returns:
        [Dict, tff.simulation.ClientData]: A dictionary and a tensorflow federated dataset containing the distributed dataset.
    """
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    clients_data = {f"client_{i}": [[], []] for i in range(1, number_of_clients + 1)}
    for i in range(len(X)):
        client = random.randrange(
            1, number_of_clients + 1, np.random.randint(1, number_of_clients + 1)
        )

        if client != 1:
            clients_data[f"client_{client}"][0].append(X[i])
            clients_data[f"client_{client}"][1].append(y[i])

    clients_data["client_1"][0] = [
        (40 - 20) * np.random.random_sample((186,)) + 20 for _ in range(SAMPLES)
    ]
    clients_data["client_1"][1] = [
        np.array([1, 0, 0, 0, 0], dtype=np.int32) for _ in range(SAMPLES)
    ]

    return clients_data, create_tff_dataset(clients_data)


def load_data(
    normalized: bool = True,
    data_analysis: bool = False,
    data_selector: Callable[
        [np.ndarray, np.ndarray, int], tff.simulation.ClientData
    ] = None,
    number_of_clients: int = 5,
    save_data: bool = False,
) -> [tff.simulation.ClientData, tff.simulation.ClientData, int]:
    """Function loads data from csv-file and preprocesses the training and test data seperately.

    Args:
        normalized (bool, optional): Whether to normalize the data. Defaults to True.\n
        data_analysis (bool, optional): Load data for data analysis. Defaults to False.\n
        data_selector (Callable[ [np.ndarray, np.ndarray, int], tff.simulation.ClientData ], optional): Data distribution. Defaults to None.\n
        number_of_clients (int, optional): Number of clients. Defaults to 5.\n
        save_data (bool, optional): Whether the data distribution should be written to disk. Defaults to False.\n

    Raises:
        ValueError: The data has to be normalized to use create_uniform_dataset.

    Returns:
        [tff.simulation.ClientData, tff.simulation.ClientData, int]: A tuple of tff.simulation.ClientData.
    """
    train_file = "data/mitbih/mitbih_train.csv"
    test_file = "data/mitbih/mitbih_test.csv"

    if data_analysis or data_selector == create_unbalanced_data:
        train_file = "../" + train_file
        test_file = "../" + test_file

    train_df = pd.read_csv(train_file, header=None)
    test_df = pd.read_csv(test_file, header=None)

    train_df[187], test_df[187] = (
        train_df[187].astype(int),
        test_df[187].astype(int),
    )

    if normalized:
        df_0 = (train_df[train_df[187] == 0]).sample(n=SAMPLES, random_state=42)
        # df_5 = (train_df[train_df[187] == 4]).sample(n=S, random_state=42)
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

    if data_selector == create_uniform_dataset:
        if not normalized:
            raise ValueError(
                "The data has to be normalized to use create_uniform_dataset"
            )

    train_client_data, train_data = data_selector(train_X, train_y, number_of_clients)

    if data_selector == create_corrupted_non_iid_dataset:
        test_client_data, test_data = create_non_iid_dataset(
            test_X, test_y, number_of_clients
        )
    else:
        test_client_data, test_data = data_selector(test_X, test_y, number_of_clients)

    if data_selector == create_unbalanced_data:
        return train_client_data, test_client_data

    if save_data:
        f = open("history/logdir/data_distributions", "ab")
        pickle.dump(train_client_data, f)
        pickle.dump(test_client_data, f)
        f.close()

    return train_data, test_data, len(train_X)


def preprocess_dataset(
    epochs: int, batch_size: int, shuffle_buffer_size: int
) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
    """Function returns a function for preprocessing of a dataset.

    Args:
        epochs (int): How many times to repeat a batch.\n
        batch_size (int): Batch size.\n
        shuffle_buffer_size (int): Buffer size for shuffling the dataset.\n

    Returns:
        Callable[[tf.data.Dataset], tf.data.Dataset]: A callable for preprocessing a dataset object.
    """

    def _reshape(element: collections.OrderedDict) -> tf.Tensor:

        return (tf.expand_dims(element["datapoints"], axis=-1), element["label"])

    @tff.tf_computation(
        tff.SequenceType(
            collections.OrderedDict(
                label=tff.TensorType(tf.int32, shape=(5,)),
                datapoints=tff.TensorType(tf.float32, shape=(186,)),
            )
        )
    )
    def preprocess(dataset: tf.data.Dataset) -> tf.data.Dataset:
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
    train_batch_size: int = 32,
    test_batch_size: int = 32,
    train_shuffle_buffer_size: int = 10000,
    test_shuffle_buffer_size: int = 10000,
    train_epochs: int = 5,
    test_epochs: int = 5,
    centralized: bool = False,
    normalized: bool = True,
    data_selector: Callable[
        [np.ndarray, np.ndarray, int], tff.simulation.ClientData
    ] = create_dataset,
    number_of_clients: int = 5,
    save_data: bool = False,
) -> [tf.data.Dataset, tf.data.Dataset, int]:
    """Function preprocesses datasets. Return input-ready datasets

    Args:
        train_batch_size (int, optional): Training batch size. Defaults to 32.\n
        test_batch_size (int, optional): Test batch size. Defaults to 32.\n
        train_shuffle_buffer_size (int, optional): Training shuffle buffer size. Defaults to 10000.\n
        test_shuffle_buffer_size (int, optional): Testing shuffle buffer size. Defaults to 10000.\n
        train_epochs (int, optional): Training epochs. Defaults to 5.\n
        test_epochs (int, optional): Test epochs. Defaults to 5.\n
        centralized (bool, optional): Whether to create dataset for centralized learning. Defaults to False.\n
        normalized (bool, optional): If the data should be normalized. Defaults to True.\n
        data_selector (Callable[ [np.ndarray, np.ndarray, int], tff.simulation.ClientData ], optional): Which data distribution to use. Defaults to create_dataset.\n
        number_of_clients (int, optional): Number of clients. Defaults to 5.\n
        save_data (bool, optional): If the data should be saved or not. Defaults to False.\n

    Returns:
        [tf.data.Dataset, tf.data.Dataset, int]: Input-ready datasets, and number of datapoints.
    """
    train_dataset, test_dataset, n = load_data(
        normalized=normalized,
        data_selector=data_selector,
        number_of_clients=number_of_clients,
        save_data=save_data,
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

    return (train_dataset, test_dataset, n)
