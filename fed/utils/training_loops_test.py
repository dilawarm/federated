import tensorflow as tf
from training_loops import centralized_training_loop

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
        history = centralized_training_loop(model, dataset, "test_reduces_loss", epochs=5, output=self.get_temp_dir(), validation_dataset=dataset)
        for metric in ["loss", "mean_squared_error", "val_loss", "val_mean_squared_error"]:
            self.assertLess(history.history[metric][-1], history.history[metric][0])

if __name__ == "__main__":
    tf.test.main()
