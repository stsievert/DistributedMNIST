import tensorflow as tf
from tensorflow.python.framework import ops


def _4d_to_2d(x):
    shape = x.get_shape().as_list()
    ndim = len(shape)

    if ndim == 4:
        shape = [shape[0]*shape[1], shape[2]*shape[3]]
    else:
        tf.logging.warning("#"*30 + "\n" + "CAREFUL! tensor not 4D")

    return tf.reshape(x, shape)

def encode(grads_and_vars, r=2):
    for i, (grad, var) in enumerate(grads_and_vars):
        shape = grad.get_shape()
        if len(shape) == 4:
            grad = _4d_to_2d(grad)
        if len(shape) == 2:
            s, u, v = tf.svd(grad)
            u = u[:, :r]
            s = s[:r]
            v = v[:, :r]
            coding = {'u': u, 's': s, 'v': v, 'shape': shape}
            grads_and_vars[i] = (coding, var)

    return grads_and_vars


def decode(grads_and_vars):
    for i, (grad, var) in enumerate(grads_and_vars):
        if isinstance(grad, dict):
            tmp = tf.matmul(grad['u'], tf.diag(grad['s']))
            tmp2 = tf.matmul(tmp, tf.transpose(grad['v']))
            grad_hat = tf.reshape(tmp2, grad['shape'])
            grads_and_vars[i] = (grad_hat, var)
    return grads_and_vars


class LowCommSync(tf.train.SyncReplicasOptimizer):
    def compute_gradients(self, *args, **kwargs):
        grads_and_vars = super(LowCommSync, self).compute_gradients(*args, **kwargs)
        coding = encode(grads_and_vars)
        return coding

    def apply_gradients(self, coding, *args, **kwargs):
        grads_and_vars = decode(coding)
        return super(LowCommSync, self).apply_gradients(grads_and_vars,
                                                        *args, **kwargs)
