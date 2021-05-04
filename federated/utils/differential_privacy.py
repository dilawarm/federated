import tensorflow as tf
import tensorflow_federated as tff


def gaussian_fixed_aggregation_factory(
    noise_multiplier: float, clients_per_round: int, clipping_value: float
) -> tff.aggregators.DifferentiallyPrivateFactory:
    """Function for differential privacy with fixed gaussian aggregation.

    Args:
        noise_multiplier (float): Noise multiplier.\n
        clients_per_round (int): Clients per round.\n
        clipping_value (float): Clipping value.\n

    Returns:
        tff.aggregators.DifferentiallyPrivateFactory: Differential Privacy Factory.
    """
    return tff.aggregators.DifferentiallyPrivateFactory.gaussian_fixed(
        noise_multiplier=noise_multiplier,
        clients_per_round=clients_per_round,
        clip=clipping_value,
    )
