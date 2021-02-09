import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te


def build_encoded_broadcast_fn(weights):
    """"""
    if weights.shape.num_elements() > 10000:
        return te.encoders.as_simple_encoder(
            te.encoders.uniform_quantization(bits=8),
            tf.TensorSpec(weights.shape, weights.dtype),
        )

    else:
        return te.encoders.as_simple_encoder(
            te.encoders.identity(), tf.TensorSpec(weights.shape, weights.dtype)
        )


def build_encoded_mean_fn(weights):
    """"""
    if weights.shape.num_elements() > 10000:
        return te.encoders.as_gather_encoder(
            te.encoders.uniform_quantization(bits=8),
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