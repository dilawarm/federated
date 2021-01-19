import tensorflow as tf
import pandas as pd
import tensorflow_federated as tff


def _preprocess_dataframe(df):
    df[187] = df[187].astype(int)
    target = df[df.columns[-1]]
    df.drop(df.columns[-1], axis=1, inplace=True)
    return df, target


def load_data():
    """"""
    mitbih_train_df = pd.read_csv("../../data/mitbih/mitbih_train.csv", header=None)
    mitbih_test_df = pd.read_csv("../../data/mitbih/mitbih_test.csv", header=None)

    train_X, train_y = _preprocess_dataframe(mitbih_train_df)
    test_X, test_y = _preprocess_dataframe(mitbih_test_df)

    def create_tf_dataset_for_client(client_id):
        client_data = train_df[train_df[] == client_id]


if __name__ == "__main__":
    load_data()