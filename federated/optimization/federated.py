import functools
import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

import tensorflow as tf
import tensorflow_federated as tff

from federated.data.data_preprocessing import (
    create_class_distributed_dataset,
    create_corrupted_non_iid_dataset,
    create_non_iid_dataset,
    create_unbalanced_data,
    create_uniform_dataset,
    get_datasets,
)
from federated.models.models import (
    create_cnn_model,
    create_dense_model,
    create_softmax_model,
)
from federated.utils.compression_utils import encoded_broadcast_process
from federated.utils.data_utils import get_client_dataset_fn, get_validation_fn
from federated.utils.differential_privacy import gaussian_fixed_aggregation_factory
from federated.utils.rfa import create_rfa_averaging
from federated.utils.training_loops import federated_training_loop

MODELS = {
    "ann": create_dense_model,
    "cnn": create_cnn_model,
    "softmax_regression": create_softmax_model,
}

CLIENT_WEIGHTING = {
    "UNIFORM": tff.learning.ClientWeighting.UNIFORM,
    "NUM_EXAMPLES": tff.learning.ClientWeighting.NUM_EXAMPLES,
}

DATA_SELECTOR = {
    "non_iid": create_non_iid_dataset,
    "uniform": create_uniform_dataset,
    "class_distributed": create_class_distributed_dataset,
}


def get_optimizer(
    optimizer: str, learning_rate: float
) -> Callable[[], tf.keras.optimizers.Optimizer]:
    """Function for getting correct optimizer.

    Args:
        optimizer (str): Optimizer to be used.\n
        learning_rate (float): Learning rate.\n

    Returns:
        Callable[[], tf.keras.optimizers.Optimizer]: Returns function for optimizer.
    """
    if optimizer == "adam":
        return lambda: tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        return lambda: tf.keras.optimizers.SGD(learning_rate=learning_rate)


def iterative_process_fn(
    tff_model: tff.learning.Model,
    server_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
    aggregation_method: str = "fedavg",
    client_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer] = None,
    iterations: int = None,
    client_weighting: tff.learning.ClientWeighting = None,
    v: float = None,
    compression: bool = False,
    model_update_aggregation_factory: Callable[
        [], tff.aggregators.UnweightedAggregationFactory
    ] = None,
) -> tff.templates.IterativeProcess:
    """Function builds an iterative process that performs federated aggregation. The function offers federated averaging, federated stochastic gradient descent and robust federated aggregation.

    Args:
        tff_model (tff.learning.Model): Federated model object.\n
        server_optimizer_fn (Callable[[], tf.keras.optimizers.Optimizer]): Server optimizer function.\n
        aggregation_method (str, optional): Aggregation method. Defaults to "fedavg".\n
        client_optimizer_fn (Callable[[], tf.keras.optimizers.Optimizer], optional): Client optimizer function. Defaults to None.\n
        iterations (int, optional): [description]. Defaults to None.\n
        client_weighting (tff.learning.ClientWeighting, optional): Client weighting. Defaults to None.\n
        v (float, optional): L2 threshold. Defaults to None.\n
        compression (bool, optional): If the model should be compressed. Defaults to False.\n
        model_update_aggregation_factory (Callable[ [], tff.aggregators.UnweightedAggregationFactory ], optional): If the model should be trained with DP. Defaults to None.\n

    Returns:
        tff.templates.IterativeProcess: An Iterative Process.
    """
    if aggregation_method == "rfa":
        return create_rfa_averaging(
            tff_model,
            iterations,
            v,
            server_optimizer_fn,
            client_optimizer_fn,
            compression=compression,
        )
    if aggregation_method == "fedavg":
        if compression:
            return tff.learning.build_federated_averaging_process(
                tff_model,
                server_optimizer_fn=server_optimizer_fn,
                client_optimizer_fn=client_optimizer_fn,
                client_weighting=client_weighting,
                broadcast_process=encoded_broadcast_process(tff_model),
                model_update_aggregation_factory=model_update_aggregation_factory,
            )
        else:
            return tff.learning.build_federated_averaging_process(
                tff_model,
                server_optimizer_fn=server_optimizer_fn,
                client_optimizer_fn=client_optimizer_fn,
                client_weighting=client_weighting,
                model_update_aggregation_factory=model_update_aggregation_factory,
            )
    if aggregation_method == "fedsgd":
        if compression:
            return tff.learning.build_federated_sgd_process(
                tff_model,
                server_optimizer_fn=server_optimizer_fn,
                broadcast_process=encoded_broadcast_process(tff_model),
                model_update_aggregation_factory=model_update_aggregation_factory,
            )
        else:
            return tff.learning.build_federated_sgd_process(
                tff_model,
                server_optimizer_fn=server_optimizer_fn,
                model_update_aggregation_factory=model_update_aggregation_factory,
            )


def federated_pipeline(
    name: str,
    aggregation_method: str,
    client_weighting: str,
    keras_model_fn: str,
    server_optimizer_fn: str,
    server_optimizer_lr: float,
    client_optimizer_fn: str,
    client_optimizer_lr: float,
    data_selector: str,
    output: str,
    client_epochs: int,
    batch_size: int,
    number_of_clients: int,
    number_of_clients_per_round: int,
    number_of_rounds: int,
    iterations: int,
    v: float,
    compression: bool,
    dp: bool,
    noise_multiplier: float,
    clipping_value: float,
    seed: int = None,
) -> float:
    """Function runs federated training pipeline on the dataset.

    Args:
        name (str): Experiment name.\n
        aggregation_method (str): Aggregation method. Defaults to "fedavg".\n
        client_weighting (str): Client weighting. Either Uniform or Data dependent. Defaults to NUM_EXAMPLES.\n
        keras_model_fn (str): Keras Model.\n
        server_optimizer_fn (str): Server Optimizer. Defaults to sgd.\n
        server_optimizer_lr (float): Learning rate for server optimizer. Defaults to 1.0.\n
        client_optimizer_fn (str): Client optimizer. Defaults to sgd.\n
        client_optimizer_lr (float): Learning rate for client optimizer. Defaults to 0.02.\n
        data_selector (str): Data distribution. Defaults to non-iid.\n
        output (str): Where to log files. Defaults to history.\n
        client_epochs (int): Number of client epochs. Defaults to 10.\n
        batch_size (int): Batch size. Defaults to 32.\n
        number_of_clients (int): Number of clients. Defaults to 10.\n
        number_of_clients_per_round (int): Number of clients per round. Defaults to 5.\n
        number_of_rounds (int): Number of global rounds. Defaults to 15.\n
        iterations (int): Number of RFA iterations. Defaults to 3.\n
        v (float): L2 threshold. Defaults to 1e-6.\n
        compression (bool): If the data should be compressed. Defaults to False.\n
        dp (bool): If the differential privacy should be applied. Defaults to False.\n
        noise_multiplier (float): The noise multipler that shoud be used with DP.\n
        clipping_norm (float): The clipping norm used with DP.\n
        seed (int, optional): Random seed. Defaults to None.\n

    Returns:
        float: Returns training time after federated learning.
    """

    keras_model_fn = MODELS[keras_model_fn]
    data_selector = DATA_SELECTOR[data_selector]
    client_weighting = CLIENT_WEIGHTING[client_weighting]

    server_optimizer_fn = get_optimizer(server_optimizer_fn, server_optimizer_lr)
    client_optimizer_fn = get_optimizer(client_optimizer_fn, client_optimizer_lr)

    train_dataset, _, len_train_X = get_datasets(
        train_batch_size=batch_size,
        centralized=False,
        normalized=True,
        train_epochs=10,
        number_of_clients=number_of_clients,
        data_selector=data_selector,
    )

    _, test_dataset, _ = get_datasets(
        train_batch_size=batch_size,
        centralized=True,
        normalized=True,
        number_of_clients=number_of_clients,
        data_selector=data_selector,
    )

    input_spec = train_dataset.create_tf_dataset_for_client(
        train_dataset.client_ids[0]
    ).element_spec

    get_keras_model = functools.partial(keras_model_fn)

    loss_fn = lambda: tf.keras.losses.CategoricalCrossentropy()
    metrics_fn = lambda: [tf.keras.metrics.CategoricalAccuracy()]

    def model_fn() -> tff.learning.Model:
        """
        Function that takes a keras model and creates an tensorflow federated learning model.
        """
        return tff.learning.from_keras_model(
            keras_model=get_keras_model(),
            input_spec=input_spec,
            loss=loss_fn(),
            metrics=metrics_fn(),
        )

    aggregation_factory = None

    if dp:
        aggregation_factory = gaussian_fixed_aggregation_factory(
            noise_multiplier=noise_multiplier,
            clients_per_round=number_of_clients_per_round,
            clipping_value=clipping_value,
        )

    iterative_process = iterative_process_fn(
        model_fn,
        server_optimizer_fn,
        aggregation_method=aggregation_method,
        client_optimizer_fn=client_optimizer_fn,
        iterations=iterations,
        client_weighting=client_weighting,
        v=v,
        compression=compression,
        model_update_aggregation_factory=aggregation_factory,
    )

    get_client_dataset = get_client_dataset_fn(
        dataset=train_dataset,
        number_of_clients_per_round=number_of_clients_per_round,
        seed=seed,
    )

    validation_fn = get_validation_fn(
        test_dataset, get_keras_model, loss_fn, metrics_fn
    )

    _, training_time = federated_training_loop(
        iterative_process=iterative_process,
        get_client_dataset=get_client_dataset,
        number_of_rounds=number_of_rounds,
        name=name,
        output=output,
        batch_size=batch_size,
        number_of_training_points=len_train_X,
        keras_model_fn=get_keras_model,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        save_model=True,
        validate_model=validation_fn,
        noise_multiplier=noise_multiplier,
    )

    return training_time
