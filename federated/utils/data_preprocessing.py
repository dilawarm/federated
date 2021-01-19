import tensorflow as tf
import pandas as pd
import tensorflow_federated as tff


def load_data():
    """"""
    train_df = pd.read_csv("../../data/mitbih/mitbih_train.csv", header=None)
    test_df = pd.read_csv("../../data/mitbih/mitbih_test.csv", header=None)

    train_df[187], test_df = (
        train_df[187].astype(int),
        test_df.astype(int),
    )

    client_ids = train_df[187].unique()

    def create_tf_dataset_for_client(client_id, df):
        client_data = df[df[187] == client_id]
        dataset = tf.data.Dataset.from_tensor_slices(client_data.to_dict("list"))
        return dataset

    train_data = tff.simulation.ClientData.from_clients_and_fn(
        client_ids=train_client_ids,
        create_tf_dataset_for_client_fn=create_tf_dataset_for_client_fn,
    )
    test_data = tff.simulation.ClientData.from_clients_and_fn(
        client_ids=test_client_ids,
        create_tf_dataset_for_client_fn=create_tf_dataset_for_client_fn,
    )


if __name__ == "__main__":
    load_data()
