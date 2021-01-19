import tensorflow as tf
import pandas as pd
import tensorflow_federated as tff


def load_data():
    """"""
    train_df = pd.read_csv("../../data/mitbih/mitbih_train.csv", header=None)
    test_df = pd.read_csv("../../data/mitbih/mitbih_test.csv", header=None)

    train_df[187], mitbih_test_df = (
        train_df[187].astype(int),
        test_df.astype(int),
    )


if __name__ == "__main__":
    load_data()
