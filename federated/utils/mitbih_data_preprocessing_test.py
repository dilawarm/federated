import tensorflow as tf
from mitbih_data_preprocessing import get_centralized_datasets


class DataPreprocessorTest(tf.test.TestCase):
    def test_dataset_shapes(self):
        """
        Function that tests method for centeralized data preprocessing.
        """
        train, test = get_centralized_datasets(train_batch_size=32, test_batch_size=100)

        train_batch = next(iter(train))
        train_batch_shape = train_batch[0].shape
        test_batch = next(iter(test))
        test_batch_shape = test_batch[0].shape
        self.assertEqual(train_batch_shape, [32, 187, 1])
        self.assertEqual(test_batch_shape, [100, 187, 1])


if __name__ == "__main__":
    tf.test.main()
