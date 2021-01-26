import tensorflow as tf
import tensorflow_federated as tff

from federated.utils.training_loops import federated_training_loop
from federated.data.mitbih_data_preprocessing import get_datasets, get_client_dataset_fn
from federated.models.mitbih_model import create_cnn_model, create_dense_model


def federated_pipeline(
    name,
    iterative_process_fn,
    output,
    client_epochs,
    batch_size,
    number_of_clients_per_round,
    number_of_rounds,
    keras_model_fn,
    client_optimizer_fn,
    server_optimizer_fn,
    seed=None,
):
    """
    Function runs federated training pipeline
    """
    train_dataset, _ = get_datasets(
        train_batch_size=batch_size,
        transform=False,
        centralized=False,
        normalized=True,
        train_epochs=client_epochs,
    )

    _, test_dataset = get_datasets(
        train_batch_size=batch_size, transform=False, centralized=True, normalized=False
    )

    input_spec = train_dataset.create_tf_dataset_for_client(
        train_dataset.client_ids[0]
    ).element_spec

    keras_model = keras_model_fn()

    model = model_fn(keras_model=keras_model, input_spec=input_spec)

    iterative_process = iterative_process_fn(
        model, client_optimizer_fn, server_optimizer_fn
    )

    get_client_dataset = get_client_dataset_fn(
        dataset=train_dataset,
        number_of_clients_per_round=number_of_clients_per_round,
        seed=seed,
    )

    federated_training_loop(
        iterative_process=iterative_process,
        get_client_dataset=get_client_dataset,
        number_of_rounds=number_of_rounds,
        name=name,
        output=output,
        save_model=True,
    )


def model_fn(keras_model, input_spec):
    return tff.learning.from_keras_model(
        keras_model=keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )


def iterative_process_fn(model_fn, client_optimizer_fn, server_optimizer_fn):
    return tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn,
    )


if __name__ == "__main__":
    name = input("Experiment name: ")

    federated_pipeline(
        name=name,
        iterative_process_fn=iterative_process_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
        output="history",
        client_epochs=1,
        batch_size=32,
        number_of_clients_per_round=10,
        number_of_rounds=10,
        keras_model_fn=create_dense_model,
    )
