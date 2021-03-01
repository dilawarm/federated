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


if __name__ == "__main__":
    gaussian_fixed_aggregation_factory(0.01, 2, 0.1)
