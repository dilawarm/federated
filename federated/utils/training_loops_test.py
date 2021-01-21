import tensorflow as tf
from federated.utils.training_loops import centralized_training_loop
import os


def create_test_dataset():
    dataset = tf.data.Dataset.from_tensor_slices(
        ([[1.0, 2.0], [3.0, 4.0]], [[5.0], [6.0]])
    )

    return dataset.repeat(4).batch(2)


def create_test_model():
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


class TrainingLoopsTest(tf.test.TestCase):
    def test_reduces_loss(self):
        dataset = create_test_dataset()
        model = create_test_model()
        history = centralized_training_loop(
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
        dataset = create_test_dataset()
        model = create_test_model()
        name = "test_write_metrics"
        output = self.get_temp_dir()

        history = centralized_training_loop(
            model, dataset, name, epochs=1, output=output, validation_dataset=dataset
        )

        log_dir = os.path.join(output, "logdir", name)
        train_log_dir = os.path.join(log_dir, "train")
        val_log_dir = os.path.join(log_dir, "validation")
        for gfile in [output, log_dir, train_log_dir, val_log_dir]:
            self.assertTrue(tf.io.gfile.exists(gfile))

    def test_learning_rate_reduction(self):
        dataset = create_test_dataset()
        model = create_test_model()
        history = centralized_training_loop(
            model,
            dataset,
            "test_reduces_loss",
            epochs=7,
            decay_epochs=5,
            learning_rate_decay=0.2,
            output=self.get_temp_dir(),
            validation_dataset=dataset,
        )

        self.assertAllClose(history.history["lr"], [0.01] * 5 + [0.002] * 2)


if __name__ == "__main__":
    tf.test.main()