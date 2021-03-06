import collections
import os

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from federated.utils.training_loops import (
    centralized_training_loop,
    federated_training_loop,
)

BATCH_SIZE = 2


class CentralizedTrainingLoopTest(tf.test.TestCase):
    """Class for testing centralized training loop."""

    def create_test_dataset(self) -> tf.data.Dataset:
        """Creates data for CL tests.

        Returns:
            tf.data.Dataset: Dataset.
        """
        dataset = tf.data.Dataset.from_tensor_slices(
            ([[1.0, 2.0], [3.0, 4.0]], [[5.0], [6.0]])
        )

        return dataset.repeat(4).batch(BATCH_SIZE)

    def create_test_model(self) -> tf.keras.Model:
        """Creates model for CL tests.

        Returns:
            tf.keras.Model: Model.
        """
        model = tf.keras.Sequential(
            tf.keras.layers.Dense(
                units=1,
                kernel_initializer="zeros",
                bias_initializer="zeros",
                input_shape=(2,),
            )
        )

        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            metrics=[tf.keras.metrics.MeanSquaredError()],
        )

        return model

    def test_reduces_loss(self):
        """Test for checking if the CL loop reduces loss."""
        dataset = self.create_test_dataset()
        model = self.create_test_model()
        history, _ = centralized_training_loop(
            model,
            dataset,
            "test_reduces_loss",
            epochs=5,
            output=self.get_temp_dir(),
            validation_dataset=dataset,
        )
        for metric in [
            "loss",
            "mean_squared_error",
            "val_loss",
            "val_mean_squared_error",
        ]:
            self.assertLess(history.history[metric][-1], history.history[metric][0])

    def test_write_metrics(self):
        """Tests if CL loop writes metrics correctly to TensorBoard"""
        dataset = self.create_test_dataset()
        model = self.create_test_model()
        name = "test_write_metrics"
        output = self.get_temp_dir()

        history, _ = centralized_training_loop(
            model, dataset, name, epochs=1, output=output, validation_dataset=dataset
        )

        log_dir = os.path.join(output, "logdir", name)
        train_log_dir = os.path.join(log_dir, "train")
        val_log_dir = os.path.join(log_dir, "validation")
        for gfile in [output, log_dir, train_log_dir, val_log_dir]:
            self.assertTrue(tf.io.gfile.exists(gfile))


class FederatedTrainingLoopTest(tf.test.TestCase):
    """Class for testing federated learning loop."""

    TYPE = collections.namedtuple("B", ["x", "y"])

    def get_batch(self) -> TYPE:
        """Function for getting a batch.

        Returns:
            self.TYPE: Batch.
        """
        return self.TYPE(
            x=np.ones([1, 784], dtype=np.float32), y=np.ones([1, 1], dtype=np.int64)
        )

    def create_tff_model(self) -> TYPE:
        """Function for creating TFF model.

        Returns:
            self.TYPE: TFF Model.
        """
        input_spec = self.TYPE(
            x=tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
            y=tf.TensorSpec(shape=[None, 1], dtype=tf.int64),
        )

        return tff.learning.from_keras_model(
            keras_model=tff.simulation.models.mnist.create_keras_model(
                compile_model=False
            ),
            input_spec=input_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        )

    def get_iterative_process(self):
        """Function for creating Iterative Process.

        Returns:
            Iterative Process.
        """
        return tff.learning.build_federated_averaging_process(
            self.create_tff_model,
            client_optimizer_fn=tf.keras.optimizers.SGD,
            server_optimizer_fn=tf.keras.optimizers.SGD,
        )

    def test_reduces_loss(self):
        """Test for checking if the FL loop reduces loss."""
        training_data = [[self.get_batch()]]
        iterative_process = self.get_iterative_process()

        get_client_dataset = lambda round: training_data

        def validate_model(curr_model):
            model = tff.simulation.models.mnist.create_keras_model(compile_model=True)
            curr_model.assign_weights_to(model)
            return {
                "loss": model.evaluate(training_data[0][0].x, training_data[0][0].y)
            }

        initial_state = iterative_process.initialize()
        untrained_model = iterative_process.get_model_weights(initial_state)

        state, _, = federated_training_loop(
            iterative_process=iterative_process,
            get_client_dataset=get_client_dataset,
            validate_model=validate_model,
            number_of_rounds=7,
            name="test_reduces_loss",
            output=self.get_temp_dir(),
            batch_size=BATCH_SIZE,
        )

        trained_model = iterative_process.get_model_weights(state)

        self.assertLess(
            validate_model(trained_model)["loss"],
            validate_model(untrained_model)["loss"],
        )


if __name__ == "__main__":
    tf.test.main()
