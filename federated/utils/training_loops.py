import tensorflow as tf
from absl import logging
import os


def centralized_training_loop(
    model, dataset, name, epochs, output, save_model=False, validation_dataset=None, test_dataset=None
):
    """
    Function trains a model on a dataset, and test its performance.
    Returns a history-object.
    """
    log_dir = os.path.join(output, "logdir", name)
    tf.io.gfile.makedirs(log_dir)

    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir)]

    logging.info("Training model")
    logging.info(f"{model.summary()}")

    history = model.fit(
        dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )

    if save_model:
        model.save(log_dir)

    if validation_dataset:
        validation_metrics = model.evaluate(validation_dataset, return_dict=True)
        logging.info("Evaluating validation metrics")
        for m in model.metrics:
            logging.info(f"\t{m.name}: {validation_metrics[m.name]:.4f}")

    if test_dataset:
        test_metrics = model.evaluate(test_dataset, return_dict=True)
        logging.info("Evaluating test metrics")
        for m in model.metrics:
            logging.info(f"\t{m.name}: {test_metrics[m.name]:.4f}")

    return history

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

if __name__ == "__main__":
    dataset = create_test_dataset()
    model = create_test_model()
    history = centralized_training_loop(model, dataset, "test", epochs=5, output="../../history", validation_dataset=dataset, save_model=True)