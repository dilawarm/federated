import tensorflow as tf
import pandas as pd
import tensorflow_federated as tff
import numpy as np
import collections


def _preprocess_dataframe(df):
    """
    Function seperates label column from datapoints columns.
    Returns tuple: dataframe without label, labels

    """

    df[187] = df[187].astype(int)
    label = df[df.columns[-1]]
    df.drop(df.columns[-1], axis=1, inplace=True)
    return df, label


def create_dataset(X, y):
    """
    Function converts pandas dataframe to tensorflow federated.
    Returns dataset of type tff.simulation.ClientData
    """
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
    """
    Function loads data from csv-file
    and preprocesses the training and test data seperatly.
    Returns a tuple of tff.simulation.ClientData
    """
    train_df = pd.read_csv("../../data/mitbih/mitbih_train.csv", header=None)
    test_df = pd.read_csv("../../data/mitbih/mitbih_test.csv", header=None)

    train_df[187], test_df[187] = (
        train_df[187].astype(int),
        test_df[187].astype(int),
    )

    train_X, train_y = _preprocess_dataframe(train_df)
    test_X, test_y = _preprocess_dataframe(test_df)

    return create_dataset(train_X, train_y), create_dataset(test_X, test_y)


if __name__ == "__main__":
    train_data, test_data = load_data()
    print(train_data.element_type_structure)
    print(test_data.element_type_structure)

    train_example = train_data.create_tf_dataset_for_client(train_data.client_ids[0])
    test_example = test_data.create_tf_dataset_for_client(test_data.client_ids[0])

    train_element = next(iter(train_example))
    test_element = next(iter(test_example))

    print(train_element["label"].numpy())
    print(test_element["label"].numpy())

    print(train_element["datapoint"].numpy())
    print(test_element["datapoint"].numpy())