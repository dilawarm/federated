import tensorflow as tf
import tensorflow_federated as tff
from federated.utils.compression_utils import (
    encoded_broadcast_process,
    encoded_mean_process,
)


def create_robust_measured_process(model, iterations, v, compression=False):
    """
    Function that creates robust measured process used in federated aggregation.
    """

    @tff.federated_computation
    def initialize_measured_process():
        return tff.federated_value((), tff.SERVER)

    @tff.tf_computation(tf.float32, model, model)
    def calculate_beta(alpha, server_weights, client_weights):
        """
        Function that calculates beta (scaled client weight).
        """

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
    def weiszfeld_algorithm(state, weights, alpha):
        """
        Function that calculates geometric median using the Weiszfeld algorithm.
        """
        if compression:
            min_w = tf.reduce_min(weights)
            max_w = tf.reduce_max(weights)
            weights = tf.quantization.quantize(
                weights, min_w, max_w, tf.quint16, mode="MIN_FIRST"
            )

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
    create_model,
    iterations,
    v,
    server_optimizer_fn,
    client_optimizer_fn,
    compression=False,
):

    """
    Returns the robust measured aggregation process.
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