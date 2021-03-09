import os
import time
from typing import Any, Callable, Dict, List, Optional

import tensorflow as tf
import tensorflow_federated as tff
from federated.models.mitbih_model import create_dense_model
from federated.utils.compression_utils import set_communication_cost_env
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib


def centralized_training_loop(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    name: str,
    epochs: int,
    output: str,
    decay_epochs: int = None,
    learning_rate_decay: int = 0,
    save_model: bool = True,
    validation_dataset: tf.data.Dataset = None,
    test_dataset: tf.data.Dataset = None,
) -> List[tf.keras.callbacks.History, float]:
    """
    Function trains a model on a dataset using centralized machine learning, and tests its performance.
    Returns the history object after fitting the model, and training time.
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

    print("Training model")
    print(model.summary())

    start_time = time.time()
    history = model.fit(
        dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )
    end_time = time.time()

    if save_model:
        model.save(log_dir)

    if validation_dataset:
        validation_metrics = model.evaluate(validation_dataset, return_dict=True)
        print("Evaluating validation metrics")
        for m in model.metrics:
            print(f"\t{m.name}: {validation_metrics[m.name]:.4f}")

    if test_dataset:
        test_metrics = model.evaluate(test_dataset, return_dict=True)
        print("Evaluating test metrics")
        for m in model.metrics:
            print(f"\t{m.name}: {test_metrics[m.name]:.4f}")

    return history, end_time - start_time


def federated_training_loop(
    iterative_process: tff.templates.IterativeProcess,
    get_client_dataset: Callable[[int], List[tf.data.Dataset]],
    number_of_rounds: int,
    name: str,
    output: str,
    batch_size: int,
    number_of_training_points: int = None,
    keras_model_fn: Callable[Optional, tf.keras.Model] = None,
    loss_fn: Callable[Optional, tf.keras.losses.Loss] = None,
    metrics_fn: Callable[Optional, tf.keras.metrics.Metric] = None,
    save_model: bool = False,
    validate_model: Callable[[Any, int], Dict[str, float]] = None,
    noise_multiplier: int = None,
) -> List[tff.Computation, float, float]:
    """
    Function trains a model on a dataset using federated learning.

    Returns the last state of the server, training time, and average round time.
    """

    env = set_communication_cost_env()

    log_dir = os.path.join(output, "logdir", name)
    train_log_dir = os.path.join(log_dir, "train")
    validation_log_dir = os.path.join(log_dir, "validation")
    communication_log_dir = os.path.join(log_dir, "communication")
    moments_accountant_log_dir = os.path.join(log_dir, "moments_accountant")

    tf.io.gfile.makedirs(log_dir)
    tf.io.gfile.makedirs(train_log_dir)
    tf.io.gfile.makedirs(validation_log_dir)
    tf.io.gfile.makedirs(communication_log_dir)
    tf.io.gfile.makedirs(moments_accountant_log_dir)

    initial_state = iterative_process.initialize()

    state = initial_state
    round_number = 1

    model_weights = iterative_process.get_model_weights(state)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    validation_summary_writer = tf.summary.create_file_writer(validation_log_dir)
    compression_summary_writer = tf.summary.create_file_writer(communication_log_dir)
    moments_accountant_writer = tf.summary.create_file_writer(
        moments_accountant_log_dir
    )

    if noise_multiplier:
        moments_accountant_log_dir = os.path.join(log_dir, "moments_accountant")
        tf.io.gfile.makedirs(moments_accountant_log_dir)
        moments_accountant_writer = tf.summary.create_file_writer(
            moments_accountant_log_dir
        )

    print("Model metrics:")

    round_times = []
    start_time = time.time()
    while round_number < number_of_rounds + 1:
        round_start_time = time.time()
        print(f"Round number: {round_number}")
        federated_train_data = get_client_dataset(round_number)

        state, metrics = iterative_process.next(state, federated_train_data)
        with train_summary_writer.as_default():
            for name, value in metrics["train"].items():
                print(f"\t{name}: {value}")
                tf.summary.scalar(name, value, step=round_number)

        if validate_model:
            validation_metrics = validate_model(model_weights)
            with validation_summary_writer.as_default():
                for metric in validation_metrics:
                    value = validation_metrics[metric]
                    print(f"\tvalidation_{metric}: {value:.4f}")
                    tf.summary.scalar(metric, value, step=round_number)

        with compression_summary_writer.as_default():
            communication_info = env.get_size_info()
            broadcasted_bits = communication_info.broadcast_bits[-1]

            tf.summary.scalar(
                "cumulative_broadcasted_bits", broadcasted_bits, step=round_number
            )

            compression_summary_writer.flush()

        if noise_multiplier:
            with moments_accountant_writer.as_default():
                eps, _ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
                    n=number_of_training_points,
                    batch_size=batch_size,
                    noise_multiplier=noise_multiplier,
                    epochs=round_number,
                    delta=1 / number_of_training_points,
                )

                tf.summary.scalar("cumulative_privacy_loss", eps, step=round_number)

        model_weights = iterative_process.get_model_weights(state)

        round_times.append(time.time() - round_start_time)

        round_number += 1

    end_time = time.time()
    avg_round_time = sum(round_times) / number_of_rounds

    if save_model:
        model = keras_model_fn()
        model.compile(
            loss=loss_fn(), optimizer=tf.keras.optimizers.SGD(), metrics=metrics_fn()
        )
        model_weights.assign_weights_to(model)
        model.save(log_dir)

    return state, end_time - start_time, avg_round_time
