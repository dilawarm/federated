import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te


def set_communication_cost_env():
    """"""
    factory = tff.framework.sizing_executor_factory()
    context = tff.framework.ExecutionContext(executor_fn=factory)
    tff.framework.set_default_context(context)

    return factory


def bit_formatter(bits):
    """"""
    bits = float(bits)
    units = ["bit", "Kibit", "Mibit", "Gibit"]
    for u in units:
        if bits < 1024.0:
            return f"{bits:3.2f} {u}"
        bits /= 1024.0
    return f"{bits:3.2f} TiB"


def build_encoded_broadcast_fn(weights):
    """"""
    if weights.shape.num_elements() > 0:
        return te.encoders.as_simple_encoder(
            te.encoders.uniform_quantization(bits=16),
            tf.TensorSpec(weights.shape, weights.dtype),
        )

    else:
        return te.encoders.as_simple_encoder(
            te.encoders.identity(), tf.TensorSpec(weights.shape, weights.dtype)
        )


def build_encoded_mean_fn(weights):
    """"""
    if weights.shape.num_elements() > 0:
        return te.encoders.as_gather_encoder(
            te.encoders.uniform_quantization(bits=16),
            tf.TensorSpec(weights.shape, weights.dtype),
        )

    else:
        return te.encoders.as_gather_encoder(
            te.encoders.identity(), tf.TensorSpec(weights.shape, weights.dtype)
        )


def encoded_broadcast_process(tff_model_fn):
    """"""
    return tff.learning.framework.build_encoded_broadcast_process_from_model(
        tff_model_fn, build_encoded_broadcast_fn
    )


def encoded_mean_process(tff_model_fn):
    """"""
    return tff.learning.framework.build_encoded_mean_process_from_model(
        tff_model_fn, build_encoded_mean_fn
    )