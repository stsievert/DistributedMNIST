import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
import time
import sys
import numpy as np


def _4d_to_2d(x):
    shape = x.get_shape().as_list()
    ndim = len(shape)

    if ndim == 4:
        shape = [shape[0]*shape[1], shape[2]*shape[3]]
    else:
        tf.logging.warning("#"*30 + "\n" + "CAREFUL! tensor not 4D")

    return tf.reshape(x, shape)


def _sample_svd(s):
    s = s.eval()
    probs = s / s[0]
    idx = np.arange(len(s))
    i = np.random.choice(s, p=probs)
    return list(i)


def _size(x):
    return tf.size(x) * x.dtype.size


def _get_bytes(x):
    if isinstance(x, dict):
        return sum([_get_bytes(v) for k, v in x.items()])
    if isinstance(x, tuple) or isinstance(x, list):
        return sum([_get_bytes(v) for v in x])
    if isinstance(x, tf.Tensor):
        return _size(x)
    return sys.getsizeof(x)


def _list_bytes(tuple_list):
    n_bytes = 0
    for x, y in tuple_list:
        n_bytes += _get_bytes(x)
        n_bytes += _get_bytes(y)
    return n_bytes


def _svd_encode(grad, r=3):
    with tf.device(grad.device):
        shape = grad.get_shape()
        ndims = len(shape)
        #  tf.logging.debug("{name} has len(shape)={shape}".format(name=var.name, shape=ndims))
        if ndims == 4:
            grad = _4d_to_2d(grad)
            ndims = len(grad.get_shape())
        if ndims == 2:
            s, u, v = tf.svd(grad, full_matrices=False)
            #  i = tf.py_func(_sample_svd, [s], tf.int32)
            u = u[:, :r]
            s = s[:r]
            v = v[:, :r]

            coding = {'u': u, 's': s, 'v': v, 'shape': shape}
            return coding
    return grad


def encode(grads_and_vars, r=2):
    for i, (grad, var) in enumerate(grads_and_vars):
        grads_and_vars[i] = (_svd_encode(grad, r=r), var)

    n_bytes = _list_bytes(grads_and_vars)
    for i, (g, v) in enumerate(grads_and_vars):
        if isinstance(g, dict):
            grads_and_vars[i][0]['n_bytes'] = n_bytes
    return grads_and_vars


def decode(grads_and_vars):
    recv_bytes = _list_bytes(grads_and_vars)
    for i, (grad, var) in enumerate(grads_and_vars):
        if isinstance(grad, dict):
            send_bytes = grad['n_bytes']
            tmp = tf.matmul(grad['u'], tf.diag(grad['s']))
            tmp2 = tf.matmul(tmp, tf.transpose(grad['v']))
            grad_hat = tf.reshape(tmp2, grad['shape'])
            grads_and_vars[i] = (grad_hat, var)
    return grads_and_vars, {'recv_bytes': recv_bytes,
                            'send_bytes': send_bytes}


def _py_time():
    t = time.time()
    return np.array(t, dtype=np.float32)


def _get_time(device=None):
    return tf.constant(0)
    if device is not None:
        with tf.device(device):
            return tf.py_func(time.time, [], tf.float32)
    return tf.py_func(_py_time, [], tf.float32)


class LowCommSync(tf.train.SyncReplicasOptimizer):
    """
    subclasses SyncReplicasOptimizer[1]

    [1]:https://github.com/tensorflow/tensorflow/blob/r0.12/tensorflow/python/training/sync_replicas_optimizer.py#L39
    """
    def __init__(self, *args, **kwargs):
        global_step = kwargs.pop('global_step', None)
        self.svd_rank = kwargs.pop('svd_rank', 3)
        self.compress = kwargs.pop('compress', False)
        assert global_step is not None
        #  self.root = global_step.device
        self.root = None
        print('LowCommSync: self.svd_rank =', self.svd_rank)
        return super(LowCommSync, self).__init__(*args, **kwargs)

    def compute_gradients(self, *args, **kwargs):
        grads_and_vars = super(LowCommSync, self).compute_gradients(*args, **kwargs)
        data = {}
        if not self.compress:
            return grads_and_vars, data
        start = _get_time(device=self.root)
        coding = encode(grads_and_vars, r=self.svd_rank)
        return coding, data  # {'encode_time': _get_time(device=self.root) - start}

    def apply_gradients(self, coding, *args, **kwargs):
        global_step = kwargs['global_step']
        with tf.device(global_step.device):
            c0 = _get_time(device=self.root)
            if self.compress:
                coding2 = _to_device(coding, device=global_step.device)
                c1 = _get_time(device=self.root)
                grads_and_vars, decode_data = decode(coding)
            else:
                grads_and_vars = _to_device(coding, global_step.device)
                #  grads_and_vars = coding
                c1 = _get_time(device=self.root)
                decode_data = {}
            data = {'comm_time': c1 - c0,
                    'decode_time': _get_time(device=self.root) - c1}
            data.update(decode_data)

            res = super(LowCommSync, self).apply_gradients(grads_and_vars,
                                                           *args, **kwargs)
        return res, data


def _to_device(x, device):
    with tf.device(device):
        if isinstance(x, tf.TensorShape):
            return x
        if isinstance(x, tf.Tensor):
            return x + 0
        if isinstance(x, (tf.Variable, variables.Variable)):
            return x + 0
        if isinstance(x, list):
            return [_to_device(xi, device) for xi in x]
        if isinstance(x, tuple):
            return tuple([_to_device(xi, device) for xi in x])
        if isinstance(x, dict):
            return {k: _to_device(v, device) for k, v in x.items()}
        return x
    raise Exception("Couldn't move to device")
