import tensorflow as tf
import pandas as pd
import tensorflow_federated as tff
import numpy as np
import collections


def _preprocess_dataframe(df):
    df[187] = df[187].astype(int)
    label = df[df.columns[-1]]
    df.drop(df.columns[-1], axis=1, inplace=True)
    return df, label


def create_dataset(X, y):
    num_of_clients = len(y)
    total_ecg_count = len(X)
    ecgs_per_set = int(np.floor(total_ecg_count / num_of_clients))

    client_dataset = collections.OrderedDict()
    for i in range(1, num_of_clients + 1):
        name_of_client = f"client_{i}"
        start = ecgs_per_set * (i - 1)
        end = ecgs_per_set * i

        data = collections.OrderedDict(
            (("label", y[start:end]), ("datapoint", X[start:end]))
        )
        client_dataset[name_of_client] = data

    return tff.simulation.FromTensorSlicesClientData(client_dataset)


def load_data():
    """"""
    train_df = pd.read_csv("../../data/mitbih/mitbih_train.csv", header=None)
    test_df = pd.read_csv("../../data/mitbih/mitbih_test.csv", header=None)

    train_df[187], test_df[187] = (
        train_df[187].astype(int),
        test_df[187].astype(int),
    )

    train_X, train_y = _preprocess_dataframe(train_df)
    test_X, test_y = _preprocess_dataframe(test_df)

    return create_dataset(train_X, train_y), create_dataset(test_X, test_y)