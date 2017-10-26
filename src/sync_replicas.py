import tensorflow as tf


def encode(grads_and_vars):
    tf.log([(g.get_shape(), v.get_shape()) for g, v in grads_and_vars])
    return x


def decode(x):
    return x


class LowCommSync(tf.train.SyncReplicasOptimizer):
    def apply_gradients(self, *args, **kwargs):
        grads_and_vars = super(LowCommSync, self).apply_gradients(*args, **kwargs)
        coding = encode(grads_and_vars)
        return coding

    def compute_gradients(self, coding):
        grads_and_vars = decode(coding)
        return super(LowCommSync, self).compute_gradients(grads_and_vars)
