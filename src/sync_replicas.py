import tensorflow as tf
from tensorflow.python.framework import ops
import time
import sys


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


#  def _sample_svd(s):



def encode(grads_and_vars, r=2):
    for i, (grad, var) in enumerate(grads_and_vars):
        shape = grad.get_shape()
        ndims = len(shape)
        tf.logging.debug("{name} has len(shape)={shape}".format(name=var.name, shape=ndims))
        if ndims == 4:
            grad = _4d_to_2d(grad)
            ndims = len(grad.get_shape())
        if ndims == 2:
            s, u, v = tf.svd(grad)
            i = tf.py_func(_sample_svd, [s], tf.int32)
            u = u[:, :r]
            s = s[:r]
            v = v[:, :r]

            n_bytes = _size(u) + _size(s) + _size(v)
            coding = {'u': u, 's': s, 'v': v, 'shape': shape,
                      'n_bytes': n_bytes}
            grads_and_vars[i] = (coding, var)

    return grads_and_vars


def _get_bytes(x):
    if issubclass(x, dict):
        return sum([_get_bytes(v) for k, v in x.items()])
    if issubclass(x, tf.Tensor):
        return _size(x)
    return sys.getsizeof(x)


def _list_bytes(tuple_list):
    n_bytes = 0
    for x, y in tuple_list:
        n_bytes += _get_bytes(x)
        n_bytes += _get_bytes(y)
    return tf.int64(n_bytes)


def decode(grads_and_vars):
    for i, (grad, var) in enumerate(grads_and_vars):
        msg = "decode: {name}['u'] is of size {size}"
        tf.logging.debug(msg.format(name=var.name, size=tf.shape(grad['s'])))
        if isinstance(grad, dict):
            tmp = tf.matmul(grad['u'], tf.diag(grad['s']))
            tmp2 = tf.matmul(tmp, tf.transpose(grad['v']))
            grad_hat = tf.reshape(tmp2, grad['shape'])
            grads_and_vars[i] = (grad_hat, var)
    return grads_and_vars, {'n_bytes': _list_bytes(grads_and_vars)}


def _get_time():
    return tf.py_func(time.time, [], tf.float64)


class LowCommSync(tf.train.SyncReplicasOptimizer):
    """
    subclasses SyncReplicasOptimizer[1]

    [1]:https://github.com/tensorflow/tensorflow/blob/r0.12/tensorflow/python/training/sync_replicas_optimizer.py#L39
    """
    def __init__(self, *args, **kwargs):
        self.svd_rank = kwargs.pop('svd_rank', 3)
        self.compress = kwargs.pop('compress', False)
        print('LowCommSync: self.svd_rank =', self.svd_rank)
        return super(LowCommSync, self).__init__(*args, **kwargs)

    def compute_gradients(self, *args, **kwargs):
        grads_and_vars = super(LowCommSync, self).compute_gradients(*args, **kwargs)
        if not self.compress:
            return grads_and_vars
        coding = encode(grads_and_vars, r=self.svd_rank)
        return coding

    def apply_gradients(self, coding, *args, **kwargs):
        data = {}
        global_step = kwargs['global_step']
        with tf.device(global_step.device):
            #  c0 = _get_time()

            if self.compress:
                tmp = coding
                #  c1 = _get_time()
                grads_and_vars, opt_data = decode(coding)
                #  data['decode_time'] = _get_time() - c1
                data.update(opt_data)
            else:
                grads_and_vars = coding
                #  c1 = _get_time()

            #  data['comm_time'] = c1 - c0
        res = super(LowCommSync, self).apply_gradients(grads_and_vars,
                                                       *args, **kwargs)
        return res, data
