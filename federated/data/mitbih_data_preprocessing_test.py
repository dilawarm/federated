import tensorflow as tf
from federated.data.mitbih_data_preprocessing import get_datasets, get_client_dataset
import numpy as np
import tensorflow_federated as tff
import collections


def create_tf_dataset_for_client(client_id):
    np.random.seed(client_id)
    x = np.random.rand(6, 1).astype(np.float64)
    y = 17 * x + 1
    dataset = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict([("x", x), ("y", y)])
    )

    return dataset.batch(2)


class DataPreprocessorTest(tf.test.TestCase):
    def test_dataset_shapes_centralized(self):
        """
        Function that tests function for centeralized data preprocessing.
        It tests whether the shape of the data matches what is expected.
        """
        train, test = get_datasets(
            train_batch_size=32, test_batch_size=100, centralized=True
        )

        train_batch = next(iter(train))
        train_batch_shape = train_batch[0].shape
        test_batch = next(iter(test))
        test_batch_shape = test_batch[0].shape
        self.assertEqual(train_batch_shape, [32, 186, 1])
        self.assertEqual(test_batch_shape, [100, 186, 1])

    def test_dataset_shapes_federated(self):
        """
        Function that tests function for federated data preprocessing.
        It tests whether the shape of the data matches what is expected.
        """
        train, test = get_datasets(
            train_batch_size=32, test_batch_size=100, centralized=False
        )

        ds_train = train.create_tf_dataset_for_client(train.client_ids[0])
        ds_test = test.create_tf_dataset_for_client(test.client_ids[0])

        for train_batch in ds_train:
            train_batch_shape = train_batch[0].shape[1:]
        for test_batch in ds_test:
            test_batch_shape = test_batch[0].shape[1:]

        self.assertEqual(train_batch_shape, [186, 1])
        self.assertEqual(test_batch_shape, [186, 1])

    def test_get_client_dataset(self):
        """
        Function that tests get_client_dataset function.
        Test if batch of clients using this function have near values what is expected.
        """

        dataset = tff.simulation.client_data.ConcreteClientData(
            [2], create_tf_dataset_for_client
        )

        client_datasets_function = get_client_dataset(
            dataset, number_of_clients_per_round=1
        )

        client_datasets = client_datasets_function(round_number=6)
        test_batch = next(iter(client_datasets[0]))
        batch = next(iter(create_tf_dataset_for_client(2)))
        self.assertAllClose(test_batch, batch)


if __name__ == "__main__":
    tf.test.main()