import tensorflow as tf

from federated.models.models import create_cnn_model


class ModelTest(tf.test.TestCase):
    """Class for testing the models."""

    def test_cnn_model_shape(self):
        """Function that tests model function for output shape."""
        data = tf.random.normal([32, 186, 1])
        model = create_cnn_model()
        logits = model(data)
        self.assertEqual(logits.shape, [32, 5])


if __name__ == "__main__":
    tf.test.main()
