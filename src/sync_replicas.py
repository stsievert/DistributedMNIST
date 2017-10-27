import tensorflow as tf
from tensorflow.python.framework import ops


def encode(grads_and_vars):
    to_log = type(grads_and_vars)
    for grad, var in grads_and_vars:
        string = "{} {}".format(grad.get_shape(), var.get_shape())
        tf.logging.info(string)
    return grads_and_vars


def decode(x):
    return x


class LowCommSync(tf.train.SyncReplicasOptimizer):
    def compute_gradients(self, *args, **kwargs):
        grads_and_vars = super(LowCommSync, self).compute_gradients(*args, **kwargs)
        coding = encode(grads_and_vars)
        return coding

    def apply_gradients(self, coding, *args, **kwargs):
        grads_and_vars = decode(coding)
        return super(LowCommSync, self).apply_gradients(grads_and_vars,
                                                        *args, **kwargs)
