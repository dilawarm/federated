import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te


def set_communication_cost_env() -> tff.framework.ExecutorFactory:
    """Set environment for loggging communication cost.

    Returns:
        tff.framework.ExecutorFactory: ExecutorFactory for the environment.
    """
    factory = tff.framework.sizing_executor_factory()
    context = tff.framework.ExecutionContext(executor_fn=factory)
    tff.framework.set_default_context(context)

    return factory


def build_encoded_broadcast_fn(weights: tf.Tensor) -> te.core.Encoder:
    """Function for encoding weights with uniform quantization.

    Args:
        weights (tf.Tensor): Weights of the model.

    Returns:
        te.core.Encoder: Encoder.
    """
    if weights.shape.num_elements() > 0:
        return te.encoders.as_simple_encoder(
            te.encoders.uniform_quantization(bits=4),
            tf.TensorSpec(weights.shape, weights.dtype),
        )

    else:
        return te.encoders.as_simple_encoder(
            te.encoders.identity(), tf.TensorSpec(weights.shape, weights.dtype)
        )


def encoded_broadcast_process(
    tff_model_fn: tff.learning.Model,
) -> tff.templates.MeasuredProcess:
    """Function for creating a MeasuredProcess used in federated learning. Uses `build_encoded_broadcast_fn` defined above. Returns MeasuredProcess

    Args:
        tff_model_fn (tff.learning.Model): Federated learning model.

    Returns:
        tff.templates.MeasuredProcess: MesuredProcess object.
    """
    return tff.learning.framework.build_encoded_broadcast_process_from_model(
        tff_model_fn, build_encoded_broadcast_fn
    )
