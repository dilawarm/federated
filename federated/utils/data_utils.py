import tensorflow as tf
import functools
import numpy as np
import collections
import tensorflow_federated as tff
from typing import Callable, Optional


def randomly_select_clients_for_round(
    population: int, num_of_clients: int, replace: bool = False, seed: int = None
) -> functools.partial:
    """
    This function creates a partial function for sampling random clients.
    Returns a partial function.
    """

    def select(round_number, seed, replace):
        """
        Function for selecting random clients.
        Returns a random sample from the client id's.
        """
        return np.random.RandomState().choice(
            population, num_of_clients, replace=replace
        )

    return functools.partial(select, seed=seed, replace=False)


def get_client_dataset_fn(
    dataset: tf.data.Dataset,
    number_of_clients_per_round: int,
    seed: int = None,
) -> Callable[[Optional], tf.data.Dataset]:
    """
    This function generates a function for selecting client-datasets for each round number.
    Returns a function for choosing clients while training.
    """
    sample_clients = randomly_select_clients_for_round(
        dataset.client_ids,
        num_of_clients=number_of_clients_per_round,
        replace=False,
        seed=seed,
    )

    def get_dataset_for_client(round_number: int):
        """
        This function chooses the client datasets.
        Returns a list of client datasets.
        """
        clients = sample_clients(round_number)
        return [dataset.create_tf_dataset_for_client(client) for client in clients]

    return get_dataset_for_client


def _convert_fn(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Converts dataset to tupled dataset.
    Returns tupled dataset.
    """
    spec = dataset.element_spec
    if isinstance(spec, collections.abc.Mapping):
        return dataset.map(lambda observation: (observation["x"], observation["y"]))
    else:
        return dataset.map(lambda x, y: (x, y))


def get_validation_fn(
    test_dataset: tf.data.Dataset,
    model_fn: Callable[[], tf.keras.models.Model],
    loss_fn,
    metrics_fn,
):
    """
    This function makes a function for evaluating a model while training.
    Returns a validation function.
    """

    def compiled_model():
        """
        This function compiles an 'empty' model.
        Returns a Keras Model object.
        """
        val_model = model_fn()
        val_model.compile(
            loss=loss_fn(), optimizer=tf.keras.optimizers.Adam(), metrics=metrics_fn()
        )
        return val_model

    test_dataset = _convert_fn(test_dataset)

    def validation_fn(trained_model):
        """
        Validates the model by running model.evaluate() on the keras model with the weights from a state from the interactive process.
        Returns the metrics after evaluation.
        """
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
