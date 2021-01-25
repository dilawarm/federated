import tensorflow as tf
from absl import logging
import os


def centralized_training_loop(
    model,
    dataset,
    name,
    epochs,
    output,
    decay_epochs=None,
    learning_rate_decay=0,
    save_model=True,
    validation_dataset=None,
    test_dataset=None,
):
    """
    Function trains a model on a dataset using centralized machine learning, and test its performance.
    Returns a history-object.
    """
    log_dir = os.path.join(output, "logdir", name)
    tf.io.gfile.makedirs(log_dir)

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    ]

    if decay_epochs:

        def decay_fn(epoch, learning_rate):
            if epoch != 0 and epoch % decay_epochs == 0:
                return learning_rate * learning_rate_decay
            else:
                return learning_rate

        callbacks.append(tf.keras.callbacks.LearningRateScheduler(decay_fn, verbose=1))

    logging.info("Training model")
    logging.info(model.summary())

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


def federated_training_loop(
    iterative_process,
    get_client_dataset,
    validate_model,
    number_of_rounds,
    name,
    output,
    get_test_dataset=None,
    save_model=True,
):
    """
    Function trains a model on a dataset using federated learning.
    Returns its state.
    """
    initial_state = iterative_process.initialize()

    state = initial_state
    round_number = 0

    model = iterative_process.get_model_weights(state)

    while round_number < number_of_rounds:
        federated_train_data = get_client_dataset(round_number)

        state, _ = iterative_process.next(state, federated_train_data)

        model = iterative_process.get_model_weights(state)
        round_number += 1

    return state