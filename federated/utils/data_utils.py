import collections
import functools
from typing import Callable, Optional

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff


def randomly_select_clients_for_round(
    population: int, num_of_clients: int, replace: bool = False, seed: int = None
) -> functools.partial:
    """This function creates a partial function for sampling random clients.

    Args:
        population (int): Client population size.\n
        num_of_clients (int): Number of clients.\n
        replace (bool, optional): With or without replacement. Defaults to False.\n
        seed (int, optional): Random seed. Defaults to None.\n

    Returns:
        functools.partial: A partial function for randomly selecting clients.
    """

    def select(round_number, seed, replace):
        return np.random.RandomState().choice(
            population, num_of_clients, replace=replace
        )

    return functools.partial(select, seed=seed, replace=False)


def get_client_dataset_fn(
    dataset: tf.data.Dataset,
    number_of_clients_per_round: int,
    seed: int = None,
) -> Callable[[], tf.data.Dataset]:
    """This function generates a function for selecting client-datasets for each round number.

    Args:
        dataset (tf.data.Dataset): Dataset.\n
        number_of_clients_per_round (int): Number of clients per round.\n
        seed (int, optional): Random seed. Defaults to None.\n

    Returns:
        Callable[[], tf.data.Dataset]: A function for choosing clients while training.
    """
    sample_clients = randomly_select_clients_for_round(
        dataset.client_ids,
        num_of_clients=number_of_clients_per_round,
        replace=False,
        seed=seed,
    )

    def get_dataset_for_client(round_number: int):
        clients = sample_clients(round_number)
        return [dataset.create_tf_dataset_for_client(client) for client in clients]

    return get_dataset_for_client


def _convert_fn(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Converts dataset to tupled dataset.

    Args:
        dataset (tf.data.Dataset): Dataset.

    Returns:
        tf.data.Dataset: Returns tupled dataset.
    """
    spec = dataset.element_spec
    if isinstance(spec, collections.abc.Mapping):
        return dataset.map(lambda observation: (observation["x"], observation["y"]))
    else:
        return dataset.map(lambda x, y: (x, y))


def get_validation_fn(
    test_dataset: tf.data.Dataset,
    model_fn: Callable[[], tf.keras.models.Model],
    loss_fn: Callable[[], tf.keras.losses.Loss],
    metrics_fn: Callable[[], tf.keras.metrics.Metric],
) -> Callable[[], tf.data.Dataset]:
    """This function makes a function for evaluating a model while training.

    Args:
        test_dataset (tf.data.Dataset): Dataset.\n
        model_fn (Callable[[], tf.keras.models.Model]): Model function.\n
        loss_fn (Callable[[], tf.keras.losses.Loss]): Loss function.\n
        metrics_fn (Callable[[], tf.keras.metrics.Metric]): Which metrics to measure.\n

    Returns:
        Callable[[], tf.data.Dataset]: Returns a validation function.
    """

    def compiled_model() -> tf.keras.Model:
        val_model = model_fn()
        val_model.compile(
            loss=loss_fn(), optimizer=tf.keras.optimizers.Adam(), metrics=metrics_fn()
        )
        return val_model

    test_dataset = _convert_fn(test_dataset)

    def validation_fn(
        trained_model: tff.learning.Model,
    ) -> Callable[[], tf.data.Dataset]:
        val_model = compiled_model()
        trained_model_weights = tff.learning.ModelWeights(
            trainable=list(trained_model.trainable),
            non_trainable=list(trained_model.non_trainable),
        )

        trained_model_weights.assign_weights_to(val_model)
        metrics = val_model.evaluate(test_dataset, verbose=0)
        return dict(
            zip(val_model.metrics_names, val_model.evaluate(test_dataset, verbose=0))
        )

    return validation_fn
