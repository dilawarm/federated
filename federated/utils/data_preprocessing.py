import tensorflow as tf
import pandas as pd
import tensorflow_federated as tff
import numpy as np
import collections
from matplotlib import pyplot as plt


def _preprocess_dataframe(df):
    df[187] = df[187].astype(int)
    target = df[df.columns[-1]]
    df.drop(df.columns[-1], axis=1, inplace=True)
    return df, target


def load_data():
    """"""
    train_df = pd.read_csv("../../data/mitbih/mitbih_train.csv", header=None)
    test_df = pd.read_csv("../../data/mitbih/mitbih_test.csv", header=None)

    train_df[187], test_df = (
        train_df[187].astype(int),
        test_df.astype(int),
    )

    train_X, train_y = _preprocess_dataframe(train_df)
    test_X, test_y = _preprocess_dataframe(test_df)

    num_of_clients = len(train_y)
    total_img_count = len(train_X)
    image_per_set = int(np.floor(total_img_count / num_of_clients))

    client_train_dataset = collections.OrderedDict()
    for i in range(1, num_of_clients + 1):
        name_of_client = "client_" + str(i)
        start = image_per_set * (i - 1)
        end = image_per_set * i

        data = collections.OrderedDict(
            (("label", train_y[start:end]), ("pixels", train_X[start:end]))
        )
        client_train_dataset[name_of_client] = data

    train_dataset = tff.simulation.FromTensorSlicesClientData(client_train_dataset)
    samples = train_dataset.create_tf_dataset_for_client(train_dataset.client_ids[0])
    element = next(iter(samples))

    print(element)
    print(samples.element_spec)


if __name__ == "__main__":
    load_data()
