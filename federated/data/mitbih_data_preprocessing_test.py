import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from federated.data.mitbih_data_preprocessing import (
    get_client_dataset_fn,
    get_datasets,
    get_validation_dataset_fn,
)
from federated.models.mitbih_model import create_dense_model


def create_test_dataset(client_id=None):
    np.random.seed(client_id)
    x = np.random.rand(6, 1).astype(np.float64)
    dataset = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict([("x", x), ("y", 17 * x + 1)])
    )
    return dataset.batch(2)


def create_test_model():
    model = tf.keras.Sequential(
        tf.keras.layers.Dense(
            units=1,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            input_shape=(1,),
        )
    )
    return model


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

    def test_get_client_dataset_fn(self):
        """
        Function that tests get_client_dataset function.
        Test if batch of clients using this function have near values what is expected.
        """

        dataset = tff.simulation.client_data.ConcreteClientData(
            [2], create_test_dataset
        )

        client_datasets_function = get_client_dataset_fn(
            dataset, number_of_clients_per_round=1
        )

        client_datasets = client_datasets_function(round_number=6)
        test_batch = next(iter(client_datasets[0]))
        batch = next(iter(create_test_dataset(2)))
        self.assertAllClose(test_batch, batch)

    def test_get_validation_dataset_fn(self):
        test_dataset = create_test_dataset()

        loss_fn = lambda: tf.keras.losses.CategoricalCrossentropy()
        metrics_fn = lambda: [tf.keras.metrics.CategoricalCrossentropy()]

        def input_spec():
            input_spec = create_test_dataset().element_spec

            return input_spec

        def model_fn():
            return tff.learning.from_keras_model(
                keras_model=create_test_model(),
                input_spec=input_spec(),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalAccuracy()],
            )

        val_dataset_function = get_validation_dataset_fn(
            test_dataset, create_test_model, loss_fn, metrics_fn
        )

        iterative_process = tff.learning.build_federated_averaging_process(
            model_fn, client_optimizer_fn=tf.keras.optimizers.SGD
        )

        initial_state = iterative_process.initialize()

        trained_model = tff.learning.ModelWeights(
            trainable=tuple(initial_state.model.trainable),
            non_trainable=tuple(initial_state.model.non_trainable),
        )

        metrics_dict = val_dataset_function(trained_model)

        self.assertIsInstance(metrics_dict, dict)


if __name__ == "__main__":
    tf.test.main()
