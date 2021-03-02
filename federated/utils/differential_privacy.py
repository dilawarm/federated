import tensorflow as tf
import tensorflow_federated as tff


def gaussian_fixed_aggregation_factory(
    noise_multiplier, clients_per_round, clipping_value
):
    return tff.aggregators.DifferentiallyPrivateFactory.gaussian_fixed(
        noise_multiplier=noise_multiplier,
        clients_per_round=clients_per_round,
        clip=clipping_value,
    )


def gaussian_adaptive_aggregation_factory(
    noise_multiplier,
    clients_per_round,
    clipping_value,
    target_unclipped_quantile,
    learning_rate,
):
    return tff.aggregators.DifferentiallyPrivateFactory.gaussian_adaptive(
        noise_multiplier=noise_multiplier,
        clients_per_round=clients_per_round,
        initial_l2_norm_clip=clipping_value,
        target_unclipped_quantile=target_unclipped_quantile,
        learning_rate=learning_rate,
    )
