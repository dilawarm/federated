import functools

import tensorflow as tf
import tensorflow_federated as tff
from federated.utils.rfa import create_rfa_averaging
from federated.data.mitbih_data_preprocessing import get_datasets
from federated.utils.data_utils import get_validation_fn, get_client_dataset_fn
from federated.models.mitbih_model import (
    create_cnn_model,
    create_dense_model,
    create_new_cnn_model,
)
from federated.utils.training_loops import federated_training_loop
from federated.utils.compression_utils import (
    encoded_broadcast_process,
)
from federated.data.mitbih_data_preprocessing import (
    create_class_distributed_dataset,
    create_non_iid_dataset,
    create_uniform_dataset,
)
import inspect
import os


def iterative_process_fn(
    tff_model,
    server_optimizer_fn,
    aggregation_method="fedavg",
    client_optimizer_fn=None,
    iterations=None,
    v=None,
    compression=False,
):
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
                broadcast_process=encoded_broadcast_process(tff_model),
            )
        else:
            return tff.learning.build_federated_averaging_process(
                tff_model,
                server_optimizer_fn=server_optimizer_fn,
                client_optimizer_fn=client_optimizer_fn,
            )
    if aggregation_method == "fedsgd":
        if compression:
            return tff.learning.build_federated_sgd_process(
                tff_model,
                server_optimizer_fn=server_optimizer_fn,
                broadcast_process=encoded_broadcast_process(tff_model),
            )
        else:
            return tff.learning.build_federated_sgd_process(
                tff_model,
                server_optimizer_fn=server_optimizer_fn,
            )


def federated_pipeline(
    name,
    iterative_process_fn,
    output,
    data_selector,
    client_epochs,
    batch_size,
    number_of_clients,
    number_of_clients_per_round,
    number_of_rounds,
    keras_model_fn,
    server_optimizer_fn,
    normalized=True,
    save_data=True,
    aggregation_method="fedavg",
    client_optimizer_fn=None,
    seed=None,
    validate_model=True,
    iterations=None,
    v=None,
    compression=False,
):
    """
    Function runs federated training pipeline
    """
    train_dataset, _ = get_datasets(
        train_batch_size=batch_size,
        centralized=False,
        normalized=normalized,
        train_epochs=client_epochs,
        number_of_clients=number_of_clients,
        data_selector=data_selector,
        save_data=save_data,
    )

    _, test_dataset = get_datasets(
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

    def model_fn():
        return tff.learning.from_keras_model(
            keras_model=get_keras_model(),
            input_spec=input_spec,
            loss=loss_fn(),
            metrics=metrics_fn(),
        )

    iterative_process = iterative_process_fn(
        model_fn,
        server_optimizer_fn,
        aggregation_method=aggregation_method,
        client_optimizer_fn=client_optimizer_fn,
        iterations=iterations,
        v=v,
        compression=compression,
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
        keras_model_fn=get_keras_model,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        save_model=True,
        validate_model=validation_fn,
    )

    server_opt_str = str(inspect.getsourcelines(server_optimizer_fn)[0][0]).strip()

    client_opt_str = str(inspect.getsourcelines(client_optimizer_fn)[0][0]).strip()

    if save_data:
        os.rename(
            "history/logdir/data_distributions",
            f"history/logdir/{name}/data_distributions",
        )

    with open(f"history/logdir/{name}/training_config.csv", "w+") as f:
        f.writelines(
            "name,training_time,avg_round_time,number_of_rounds,number_of_clients_per_round,client_epochs,server_optimizer_fn,client_optimizer_fn,aggregation_method\n"
        )
        f.writelines(
            f"{name},{training_time},{avg_round_time},{number_of_rounds},{number_of_clients_per_round},{client_epochs},{server_opt_str}{client_opt_str}{aggregation_method}"
        )
        f.close()


if __name__ == "__main__":
    name = input("Experiment name: ")
    aggregation_method = input("Aggregation method: ")

    federated_pipeline(
        name=name,
        iterative_process_fn=iterative_process_fn,
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
        data_selector=create_non_iid_dataset,
        output="history",
        client_epochs=10,
        batch_size=32,
        number_of_clients=10,
        number_of_clients_per_round=10,
        number_of_rounds=15,
        keras_model_fn=create_new_cnn_model,
        normalized=True,
        save_data=False,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        aggregation_method=aggregation_method,
        iterations=2,
        v=1e-6,
        compression=False,
    )