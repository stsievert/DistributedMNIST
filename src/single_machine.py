# This code is taken and modified from the inception_distribute_train.py file of
# google's tensorflow inception model. The original source is here - https://github.com/tensorflow/models/tree/master/inception.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from threading import Timer
import os.path
import time

import numpy as np
import random
import tensorflow as tf
import signal
import sys
import os
import math

from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.client import timeline
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.training import input as tf_input

import cifar_input
import resnet_model
import sync_replicas_optimizer_modified

np.set_printoptions(threshold=np.nan)
tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('n_train_epochs', 1000, 'Number of epochs to train for')
tf.app.flags.DEFINE_boolean('should_summarize', False, 'Whether Chief should write summaries.')
tf.app.flags.DEFINE_boolean('timeline_logging', False, 'Whether to log timeline of events.')

tf.app.flags.DEFINE_string('train_dir', '/tmp/resnet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('rpc_port', 1235,
                           """Port for timeout communication""")

tf.app.flags.DEFINE_integer('max_steps', 1000000, 'Number of batches to run.')
tf.app.flags.DEFINE_string('subset', 'train', 'Either "train" or "validation".')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')
tf.app.flags.DEFINE_boolean('variable_batchsize', False,
                            'Use variable batchsize comptued using R.')

# Task ID is used to select the chief and also to access the local_step for
# each replica to check staleness of the gradients in sync_replicas_optimizer.
tf.app.flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')

# More details can be found in the sync_replicas_optimizer class:
# tensorflow/python/training/sync_replicas_optimizer.py
tf.app.flags.DEFINE_integer('num_replicas_to_aggregate', -1,
                            """Number of gradients to collect before """
                            """updating the parameters.""")
tf.app.flags.DEFINE_integer('save_interval_secs', 10,
                            'Save interval seconds.')
tf.app.flags.DEFINE_integer('save_summaries_secs', 300,
                            'Save summaries interval seconds.')

# **IMPORTANT**
# Please note that this learning rate schedule is heavily dependent on the
# hardware architecture, batch size and any changes to the model architecture
# specification. Selecting a finely tuned learning rate schedule is an
# empirical process that requires some experimentation. Please see README.md
# more guidance and discussion.
#
# Learning rate decay factor selected from https://arxiv.org/abs/1604.00981
#tf.app.flags.DEFINE_float('initial_learning_rate', 0.045,
#                          'Initial learning rate.')
# For flowers
tf.app.flags.DEFINE_float('initial_learning_rate', 0.01,
                          'Initial learning rate.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0,
                          'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.999,
                          'Learning rate decay factor.')

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

EVAL_BATCHSIZE=2000

def train():
  """Train on dataset for a number of steps."""
  with tf.Graph().as_default(), tf.device('/gpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.Variable(0, name="global_step", trainable=False)

    #images, labels = cifar_input.build_input(FLAGS.dataset, FLAGS.data_dir, FLAGS.batch_size, "train")
    images, labels = cifar_input.placeholder_inputs()
    variable_batchsize_inputs = cifar_input.build_input_multi_batchsize(FLAGS.dataset, FLAGS.data_dir, FLAGS.batch_size, "train")
    
    # get num of examples in training set
    #dataset_num_examples = training_set.shape[0]

    # Calculate the learning rate schedule.
    #num_batches_per_epoch = int(dataset_num_examples / FLAGS.batch_size)

    # set hyperparameter
    hps = resnet_model.HParams(batch_size=FLAGS.batch_size,
                           num_classes=10 if FLAGS.dataset=="cifar10" else 100,
                           min_lrn_rate=0.0001,
                           lrn_rate=FLAGS.initial_learning_rate,
                           num_residual_units=5,
                           use_bottleneck=False,
                           weight_decay_rate=0.0002,
                           relu_leakiness=0.1,
                           optimizer='sgd')

    model = resnet_model.ResNet(hps, images, labels, "train")
    model.build_graph()

    # Create an optimizer that performs gradient descent.
    opt = tf.train.GradientDescentOptimizer(FLAGS.initial_learning_rate)

    # Compute gradients with respect to the loss.
    grads = opt.compute_gradients(model.cost)
    apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)

    with tf.control_dependencies([apply_gradients_op]):
      train_op = tf.identity(model.cost, name='train_op')

    # Train, checking for Nans. Concurrently run the summary operation at a
    # specified interval. Note that the summary_op and train_op never run
    # simultaneously in order to prevent running out of GPU memory.
    next_summary_time = time.time() + FLAGS.save_summaries_secs
    begin_time = time.time()

    # Keep track of own iteration
    cur_iteration = -1
    iterations_finished = set()
    n_examples_processed = 0
    cur_epoch_track = 0
    train_error_time = 0
    loss_value = -1

    checkpoint_save_secs = 60 * 2
    ########################################################################################
    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge_all()

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # these two parameters is used to measure when to enter next epoch
    local_data_batch_idx = 0
    epoch_counter = 0
    batch_counter = 0

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)
    for step in range(FLAGS.max_steps):
      cur_iteration += 1
      sys.stdout.flush()

      start_time = time.time()

      run_options = tf.RunOptions()
      run_metadata = tf.RunMetadata()

      if FLAGS.timeline_logging:
        run_options.trace_level=tf.RunOptions.FULL_TRACE
        run_options.output_partition_graphs=True

      # Dequeue variable batchsize inputs
      batchsize_to_use = FLAGS.batch_size
      #tf.logging.info("Batchsize: %d" % batchsize_to_use)
      images_real, labels_real = sess.run(variable_batchsize_inputs[batchsize_to_use], feed_dict={images:np.zeros([1, 32, 32, 3]), labels: np.zeros([1, 10 if FLAGS.dataset == 'cifar10' else 100])})
      #tf.logging.info("Dequeued...")
      loss_value, step = sess.run([train_op, global_step], run_metadata=run_metadata, options=run_options, feed_dict={images:images_real, labels:labels_real})
      n_examples_processed += batchsize_to_use

      # This uses the queuerunner which does not support variable batch sizes
      #loss_value, step = sess.run([train_op, global_step], run_metadata=run_metadata, options=run_options)
      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      # Log the elapsed time per iteration
      finish_time = time.time()

      # Create the Timeline object, and write it to a json
      if FLAGS.timeline_logging:
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('%s/worker=%d_timeline_iter=%d.json' % (FLAGS.train_dir, FLAGS.task_id, step), 'w') as f:
          f.write(ctf)

      if step > FLAGS.max_steps:
        break

      cur_epoch = n_examples_processed / float(cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
      tf.logging.info("epoch: %f time %f" % (cur_epoch, time.time()-begin_time));
      if cur_epoch >= FLAGS.n_train_epochs:
        break

      duration = time.time() - start_time
      examples_per_sec = FLAGS.batch_size / float(duration)
      format_str = ('Worker %d: %s: step %d, loss = %f'
                    '(%.1f examples/sec; %.3f  sec/batch)')
      tf.logging.info(format_str %
                      (FLAGS.task_id, datetime.now(), step, loss_value,
                       examples_per_sec, duration))

      # Determine if the summary_op should be run on the chief worker.
      if next_summary_time < time.time() and FLAGS.should_summarize:

        tf.logging.info('Running Summary operation on the chief.')
        summary_str = mon_sess.run(summary_op)
        sv.summary_computed(sess, summary_str)
        tf.logging.info('Finished running Summary operation.')

        # Determine the next time for running the summary.
        next_summary_time += FLAGS.save_summaries_secs

    # Save after the training ends.
    saver.save(sess,
               os.path.join(FLAGS.train_dir, 'model.ckpt'),
               global_step=global_step)

def main(_):
  cifar_input.maybe_download_and_extract(FLAGS.dataset)
  train()

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.DEBUG)
  tf.app.run()