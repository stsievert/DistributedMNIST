import time
import sys
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import optimizer
from tensorflow.python.training import queue_runner


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


    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """Apply gradients to variables.
        This contains most of the synchronization implementation and also wraps the
        apply_gradients() from the real optimizer.
        Args:
          grads_and_vars: List of (gradient, variable) pairs as returned by
            compute_gradients().
          global_step: Optional Variable to increment by one after the
            variables have been updated.
          name: Optional name for the returned operation.  Default to the
            name passed to the Optimizer constructor.
        Returns:
          train_op: The op to dequeue a token so the replicas can exit this batch
          and start the next one. This is executed by each replica.
        Raises:
          ValueError: If the grads_and_vars is empty.
          ValueError: If global step is not provided, the staleness cannot be
            checked.
        """
        if not grads_and_vars:
            raise ValueError("Must supply at least one variable")

        if global_step is None:
            raise ValueError("Global step is required to check staleness")

        self._global_step = global_step
        train_ops = []
        aggregated_grad = []
        var_list = []

        self._local_step = variables.Variable(
            initial_value=0,
            trainable=False,
            collections=[ops.GraphKeys.LOCAL_VARIABLES],
            dtype=global_step.dtype.base_dtype,
            name="sync_rep_local_step")
        self.local_step_init_op = state_ops.assign(self._local_step, global_step)
        chief_init_ops = [self.local_step_init_op]
        self.ready_for_local_init_op = variables.report_uninitialized_variables(
            variables.global_variables())

        with ops.name_scope(None, self._name):
            for grad, var in grads_and_vars:
                var_list.append(var)
                with ops.device(var.device):
                    # Dense gradients.
                    if grad is None:
                        aggregated_grad.append(None)  # pass-through.
                        continue
                    elif isinstance(grad, ops.Tensor):
                        grad_accum = data_flow_ops.ConditionalAccumulator(
                            grad.dtype,
                            shape=var.get_shape(),
                            shared_name=var.name + "/grad_accum")
                        train_ops.append(grad_accum.apply_grad(
                            grad, local_step=self._local_step))
                        aggregated_grad.append(grad_accum.take_grad(
                            self._replicas_to_aggregate))
                    else:
                        if not isinstance(grad, ops.IndexedSlices):
                            raise ValueError("Unknown grad type!")
                        grad_accum = data_flow_ops.SparseConditionalAccumulator(
                            grad.dtype, shape=(), shared_name=var.name + "/grad_accum")
                        train_ops.append(grad_accum.apply_indexed_slices_grad(
                            grad, local_step=self._local_step))
                        aggregated_grad.append(grad_accum.take_indexed_slices_grad(
                            self._replicas_to_aggregate))

                    self._accumulator_list.append((grad_accum, var.device))

            aggregated_grads_and_vars = zip(aggregated_grad, var_list)

            # sync_op will be assigned to the same device as the global step.
            with ops.device(global_step.device), ops.name_scope(""):
                #  aggregated_grads_and_vars = self._decode(aggregated_grads_and_vars)
                update_op = self._opt.apply_gradients(aggregated_grads_and_vars,
                                                      global_step)

            # Create token queue.
            with ops.device(global_step.device), ops.name_scope(""):
                sync_token_queue = (
                    data_flow_ops.FIFOQueue(-1,
                                            global_step.dtype.base_dtype,
                                            shapes=(),
                                            shared_name="sync_token_q"))
                self._sync_token_queue = sync_token_queue

                # dummy_queue is passed to the queue runner. Don't use the real queues
                # because the queue runner doesn't automatically reopen it once it
                # closed queues in PS devices.
                dummy_queue = (
                    data_flow_ops.FIFOQueue(1,
                                            types_pb2.DT_INT32,
                                            shapes=(),
                                            shared_name="dummy_queue"))

            with ops.device(global_step.device), ops.name_scope(""):
                # Replicas have to wait until they can get a token from the token queue.
                with ops.control_dependencies(train_ops):
                    token = sync_token_queue.dequeue()
                train_op = state_ops.assign(self._local_step, token)

                with ops.control_dependencies([update_op]):
                    # Sync_op needs to insert tokens to the token queue at the end of the
                    # step so the replicas can fetch them to start the next step.
                    tokens = array_ops.fill([self._tokens_per_step], global_step)
                    sync_op = sync_token_queue.enqueue_many((tokens,))

                if self._variable_averages is not None:
                    with ops.control_dependencies([sync_op]), ops.name_scope(""):
                        sync_op = self._variable_averages.apply(
                            self._variables_to_average)

                self._chief_queue_runner = queue_runner.QueueRunner(dummy_queue,
                                                                    [sync_op])
            for accum, dev in self._accumulator_list:
                with ops.device(dev):
                    chief_init_ops.append(
                        accum.set_global_step(
                            global_step, name="SetGlobalStep"))
            self.chief_init_op = control_flow_ops.group(*(chief_init_ops))
            self._gradients_applied = True
        return train_op
