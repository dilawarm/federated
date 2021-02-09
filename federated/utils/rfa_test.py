import tensorflow as tf
import tensorflow_federated as tff
from federated.utils.rfa import create_rfa_averaging
import numpy as np
import collections


class RFATest(tf.test.TestCase):
    def get_test_data(self):
        random = np.random.RandomState(0)
        X = random.rand(10, 10).astype(np.float32)
        y = random.rand(10, 1).astype(np.float32)

        return [
            tf.data.Dataset.from_tensor_slices(
                collections.OrderedDict(x=X[i : i + 1], y=y[i : i + 1])
            ).batch(1)
            for i in range(X.shape[0])
        ]

    def create_model(self):
        def create_model_fn():
            keras_model = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Input(shape=(10,)),
                    tf.keras.layers.Dense(
                        1, kernel_initializer="zeros", use_bias=False
                    ),
                ]
            )

            return tff.learning.from_keras_model(
                keras_model=keras_model,
                input_spec=self.get_test_data()[0].element_spec,
                loss=tf.keras.losses.MeanSquaredError(),
            )

        return create_model_fn

    def test_rfa(self):
        create_model_fn = self.create_model()
        dataset = self.get_test_data()
        iterative_process = create_rfa_averaging(
            create_model_fn,
            iterations=2,
            v=1e-6,
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
            client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01),
        )
        state = iterative_process.initialize()
        state, _ = iterative_process.next(state, dataset)
        print(type(state))


if __name__ == "__main__":
    tf.test.main()