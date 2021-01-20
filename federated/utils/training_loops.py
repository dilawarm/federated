import tensorflow as tf
from absl import logging
import os


def centralized_training_loop(
    model, dataset, name, epochs, output, validation_dataset=None, test_dataset=None
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
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )

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
