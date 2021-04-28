# The implementation is inspired by https://github.com/google-research/federated/tree/master/robust_aggregation

import tensorflow as tf
import tensorflow_federated as tff
from federated.utils.compression_utils import (
    encoded_broadcast_process,
)
from typing import Callable, Optional


def create_robust_measured_process(
    model: tff.learning.Model, iterations: int, v: float, compression: bool = False
) -> tff.templates.MeasuredProcess:
    """Function that creates robust measured process used in federated aggregation.

    Args:
        model (tf.keras.Model): Model to train.\n
        iterations (int): Number of iterations.\n
        v (float): L2 threshold.\n
        compression (bool, optional): If the model should be compressed. Defaults to False.\n

    Returns:
        tff.templates.MeasuredProcess: Returns a `tff.templates.MeasuredProcess` which defines how to aggregate client updates.
    """

    @tff.federated_computation
    def initialize_measured_process() -> tff.FederatedType:
        return tff.federated_value((), tff.SERVER)

    @tff.tf_computation(tf.float32, model, model)
    def calculate_beta(
        alpha: float, server_weights: tf.Tensor, client_weights: tf.Tensor
    ) -> float:

        func = lambda x, y: tf.norm(x - y) ** 2
        return alpha / tf.math.maximum(
            v,
            tf.math.sqrt(
                tf.reduce_sum(
                    tf.nest.map_structure(func, server_weights, client_weights)
                )
            ),
        )

    @tff.federated_computation(
        tff.type_at_server(()),
        tff.type_at_clients(model),
        tff.type_at_clients(tf.float32),
    )
    def weiszfeld_algorithm(
        state: tff.Computation, weights: tf.Tensor, alpha: float
    ) -> tff.templates.MeasuredProcessOutput:
        mean = tff.federated_mean(weights, weight=alpha)
        for _ in range(iterations - 1):
            broadcast_mean = tff.federated_broadcast(mean)
            beta = tff.federated_map(calculate_beta, (alpha, broadcast_mean, weights))
            mean = tff.federated_mean(weights, weight=beta)
        return tff.templates.MeasuredProcessOutput(
            state, mean, tff.federated_value((), tff.SERVER)
        )

    return tff.templates.MeasuredProcess(
        initialize_fn=initialize_measured_process, next_fn=weiszfeld_algorithm
    )


def create_rfa_averaging(
    create_model: Callable[[], tff.learning.Model],
    iterations: int,
    v: float,
    server_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
    client_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
    compression: bool = False,
) -> tff.templates.IterativeProcess:
    """Function for setting up Robust Federated Aggregation.

    Args:
        create_model (Callable[[], tff.learning.Model]): Function for creating a model.\n
        iterations (int): Calls to Secure Average Oracle.\n
        v (float): L2 Threshold.\n
        server_optimizer_fn (Callable[[], tf.keras.optimizers.Optimizer]): Server Optimizer Function.\n
        client_optimizer_fn (Callable[[], tf.keras.optimizers.Optimizer]): Client Optimizer Function.\n
        compression (bool, optional): Whether the model should be compressed. Defaults to False.\n

    Returns:
        tff.templates.IterativeProcess: Returns an Iterative Process with the RFA Averaging scheme
    """
    with tf.Graph().as_default():
        model = tff.framework.type_from_tensors(create_model().weights.trainable)
    robust_measured_process = create_robust_measured_process(
        model, iterations, v, compression=compression
    )

    if compression:
        return tff.learning.build_federated_averaging_process(
            create_model,
            server_optimizer_fn=server_optimizer_fn,
            client_optimizer_fn=client_optimizer_fn,
            aggregation_process=robust_measured_process,
            broadcast_process=encoded_broadcast_process(create_model),
        )

    else:
        return tff.learning.build_federated_averaging_process(
            create_model,
            server_optimizer_fn=server_optimizer_fn,
            client_optimizer_fn=client_optimizer_fn,
            aggregation_process=robust_measured_process,
        )
