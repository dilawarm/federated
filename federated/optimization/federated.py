# type: ignore

import functools
import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

import tensorflow as tf
import tensorflow_federated as tff
from federated.data.data_preprocessing import (
    create_class_distributed_dataset,
    create_non_iid_dataset,
    create_unbalanced_data,
    create_uniform_dataset,
    get_datasets,
)
from federated.models.models import (
    create_cnn_model,
    create_dense_model,
    create_linear_model,
    create_new_cnn_model,
)
from federated.utils.compression_utils import encoded_broadcast_process
from federated.utils.data_utils import get_client_dataset_fn, get_validation_fn
from federated.utils.differential_privacy import (
    gaussian_adaptive_aggregation_factory,
    gaussian_fixed_aggregation_factory,
)
from federated.utils.rfa import create_rfa_averaging
from federated.utils.training_loops import federated_training_loop


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

    """
    Function builds an iterative process that performs federated aggregation.
    The function offers federated averaging, federated stochastic gradient descent
    and robust federated aggregation.
    Returns an iterativeProcess.

    """
    if aggregation_method not in ["fedavg", "fedsgd", "rfa"]:
        raise ValueError("Aggregation method does not exist")
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
    iterative_process_fn: Any,
    output: str,
    data_selector: Any,
    client_epochs: int,
    batch_size: int,
    number_of_clients: int,
    number_of_clients_per_round: int,
    number_of_rounds: int,
    keras_model_fn: Any,
    server_optimizer_fn: Any,
    normalized: bool = True,
    save_data: bool = True,
    aggregation_method: str = "fedavg",
    client_optimizer_fn: Any = None,
    client_weighting: tff.learning.ClientWeighting = None,
    seed: int = None,
    validate_model: bool = True,
    iterations: int = None,
    v: int = None,
    compression: bool = False,
    model_update_aggregation_factory: Any = None,
) -> None:

    """
    Function runs federated training pipeline on the dataset.
    Also logs training configurations used during training.
    """
    train_dataset, _, len_train_X = get_datasets(
        train_batch_size=batch_size,
        centralized=False,
        normalized=normalized,
        train_epochs=client_epochs,
        number_of_clients=number_of_clients,
        data_selector=data_selector,
        save_data=save_data,
    )

    _, test_dataset, _ = get_datasets(
        train_batch_size=batch_size,
        centralized=True,
        normalized=True,
        number_of_clients=number_of_clients,
        data_selector=data_selector,
        save_data=save_data,
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
    if model_update_aggregation_factory:
        aggregation_factory = model_update_aggregation_factory()

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

    if validate_model:
        validation_fn = get_validation_fn(
            test_dataset, get_keras_model, loss_fn, metrics_fn
        )
    else:
        validation_fn = None

    _, training_time, avg_round_time = federated_training_loop(
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

    server_opt_str = str(inspect.getsourcelines(server_optimizer_fn)[0][0]).strip()

    client_opt_str = str(inspect.getsourcelines(client_optimizer_fn)[0][0]).strip()

    if model_update_aggregation_factory:
        agg_factory_tuple = inspect.getsourcelines(model_update_aggregation_factory)[0]
        agg_factory_str = "".join(str(i).strip() for i in agg_factory_tuple)
    else:
        agg_factory_str = ""

    if save_data:
        os.rename(
            "history/logdir/data_distributions",
            f"history/logdir/{name}/data_distributions",
        )

    with open(f"history/logdir/{name}/training_config.csv", "w+") as f:
        f.writelines(
            "name,training_time,avg_round_time,number_of_rounds,number_of_clients_per_round,client_epochs,iterations,server_optimizer_fn,client_optimizer_fn,aggregation_method,normalized,compression,data_selector,aggregation_factory, model_type\n"
        )
        f.writelines(
            f"{name},{training_time},{avg_round_time},{number_of_rounds},{number_of_clients_per_round},{client_epochs},{iterations},{server_opt_str}{client_opt_str}{aggregation_method},{normalized},{compression},{data_selector},{agg_factory_str}{keras_model_fn}"
        )
        f.close()


if __name__ == "__main__":
    name = input("Experiment name: ")
    aggregation_method = input("Aggregation method: ")
    number_of_clients_per_round = 10
    noise_multiplier = None
    clipping_value = None

    federated_pipeline(
        name=name,
        iterative_process_fn=iterative_process_fn,
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
        data_selector=create_non_iid_dataset,
        output="history",
        client_epochs=10,
        batch_size=32,
        number_of_clients=10,
        number_of_clients_per_round=number_of_clients_per_round,
        number_of_rounds=15,
        keras_model_fn=create_dense_model,
        normalized=True,
        save_data=False,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        aggregation_method=aggregation_method,
        client_weighting=tff.learning.ClientWeighting.NUM_EXAMPLES,
        iterations=3,
        v=1e-6,
        compression=False,
        model_update_aggregation_factory=None,
    )
