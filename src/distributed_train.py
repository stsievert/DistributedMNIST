# This code is taken and modified from the inception_distribute_train.py file of
# google's tensorflow inception model. The original source is here - https://github.com/tensorflow/models/tree/master/inception.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from sync_replicas_optimizer_modified.sync_replicas_optimizer_modified import TimeoutReplicasOptimizer
import os.path
import time

import numpy as np
import random
import tensorflow as tf
import signal
import sys
import os

from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.client import timeline
import mnist

from twisted.spread import pb
from twisted.internet import reactor
from threading import Thread, Timer

np.set_printoptions(threshold=np.nan)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('timeout_method', True, 'Use the timeout straggler killing method')
tf.app.flags.DEFINE_boolean('should_summarize', False, 'Whether Chief should write summaries.')
tf.app.flags.DEFINE_boolean('timeline_logging', False, 'Whether to log timeline of events.')
tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")

tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('rpc_port', 1235,
                           """Port for rpc communication""")

tf.app.flags.DEFINE_integer('max_steps', 1000000, 'Number of batches to run.')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size.')
tf.app.flags.DEFINE_string('subset', 'train', 'Either "train" or "validation".')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')

# Task ID is used to select the chief and also to access the local_step for
# each replica to check staleness of the gradients in sync_replicas_optimizer.
tf.app.flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')

# More details can be found in the sync_replicas_optimizer class:
# tensorflow/python/training/sync_replicas_optimizer.py
tf.app.flags.DEFINE_integer('num_replicas_to_aggregate', -1,
                            """Number of gradients to collect before """
                            """updating the parameters.""")
tf.app.flags.DEFINE_integer('save_interval_secs', 20,
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
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          'Initial learning rate.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0,
                          'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.999,
                          'Learning rate decay factor.')

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

##########################################
# Signal handling for killing iterations #
##########################################
def signal_handler(signal, frame):
  tf.logging.info('SIGNAL RECEIVED - %f' % time.time())
  raise Exception

signal.signal(signal.SIGINT, signal_handler)

##################
# RPC procedures #
##################
class WorkerStatusServer(pb.Root):
  def __init__(self):
    self.worker_id = FLAGS.task_id
    self.n_total_workers = len(FLAGS.worker_hosts.split(","))
    self.iteration_track = [0] * self.n_total_workers
    self.n_to_collect = FLAGS.num_replicas_to_aggregate
    self.ready_to_start = False
    self.iterations_killed = set()
    tf.logging.info("Worker %d: starting status server..." % FLAGS.task_id)

    # When to collect statistico
    self.iteration_start_collect = 5
    self.iteration_end_collect = 50

    # Statistics tracking
    self.iteration_start_times = []
    self.iteration_times = []
    self.elapsed_max_time = -1
    self.elapsed_min_time = -1
    self.elapsed_avg_time = -1
    self.elapsed_stdev_time = -1

    self.start_kill_time = []
    self.end_kill_time = []

    self.collect_statistics = True

  def is_stable(self):
    # In the beginning, workers start at different times.
    # To account for this, the cluster state is stable when all workers
    # are at > N iterations.
    STABLE_ITERATION = 20
    n_stable_required =  self.n_total_workers
    n_stable = sum([1 if x > STABLE_ITERATION else 0 for x in self.iteration_track])
    return n_stable_required == n_stable

  # NOTE THIS IS DEPRECATED.
  # This relies on killing stragglers after knowing for sure one is a straggler.
  # This doesn't work due to a lag between kill signal delivery and reception.
  def check_is_straggler(self):

    if not self.is_stable():
      return False
    self_iteration = self.iteration_track[self.worker_id]
    max_iteration = max(self.iteration_track)
    n_ahead = sum([1 if x > self_iteration else 0 for x in self.iteration_track])
    if n_ahead >= self.n_to_collect:
      # KILL PROCESS
      tf.logging.info("Worker %d: I am a straggler" % self.worker_id)
      if self_iteration not in self.iterations_killed:
        self.iterations_killed.add(self_iteration)
        tf.logging.info("Committing suicide! - %f" % time.time())
        os.kill(os.getpid(), signal.SIGINT)

  def compute_avg_kill_time(self):
    # Computes avg time between signal sent and signal received.
    times = [self.end_kill_time[i] - self.start_kill_time[i] for i in \
             range(min(len(self.end_kill_time), len(self.start_kill_time)))]
    if len(times) == 0:
      return 0
    return sum(times) / float(len(times))

  # Set a timeout upon which we check if we are still computing.
  # If so, we kill self.
  # Assumes that the last worker has just to begun an iteration
  def set_suicide_timeout(self, iter_start_time, cur_iteration):

    # Make sure we have collected necessary data
    if cur_iteration <= self.iteration_end_collect:
      return

    # How far are we from iter start time
    avg_kill_time_delay = self.compute_avg_kill_time()
    time_to_suicide = self.elapsed_avg_time - avg_kill_time_delay + self.elapsed_stdev_time
    #time_to_suicide = .1

    # Make sure we get at least the average amount of compute time.
    #if time_to_suicide <= self.elapsed_avg_time:
    #  return

    def commit_suicide():
      # Still on the current iteration? Kill self.
      if self.iteration_track[self.worker_id] == cur_iteration:
        tf.logging.info("Sending suicide signal on iteration %d! - %f" % (cur_iteration, time.time()))
        self.start_kill_time.append(time.time())
        tf.logging.info("YOYOYO I APPPENDING THE SIGNAL")
        try:
          os.kill(os.getpid(), signal.SIGINT)
        except Exception, e:
          tf.logging.info("WTF HAPPENED? %s" % e)
        tf.logging.info("YOYOYO I SENT THE SIGNAL")

    Timer(time_to_suicide, commit_suicide).start()

  def remote_suicide_signal_received(self, time):
    tf.logging.info("Received suicide signal! - %f" % time)
    self.end_kill_time.append(time)
    tf.logging.info("Average delay between kill signal sending and delivery: %f" % self.compute_avg_kill_time())

  def remote_notify_starting(self, worker_id, iteration):
    # Called when worker_id notifies this machine that it is starting iteration.
    cur_time = time.time()
    tf.logging.info("Worker %d: Was notified that worker %d started iteration %d - t=%f" % (self.worker_id, worker_id, iteration, cur_time))

    self.iteration_track[worker_id] = iteration

    # Keep track of statistics of iterations start times
    while iteration >= len(self.iteration_start_times):
      self.iteration_start_times.append([0] * self.n_total_workers)
    self.iteration_start_times[iteration][worker_id] = cur_time

    # Keep track of statistics for the statistics collecting region.
    if iteration > self.iteration_start_collect and iteration < self.iteration_end_collect:

      # Track statistics
      other_worker_iterations = [x for i,x in enumerate(self.iteration_track) if i != worker_id]
      is_last_to_begin = len([x for x in other_worker_iterations if iteration <= x]) == len(other_worker_iterations)
      if is_last_to_begin and iteration > self.iteration_start_collect:
        tf.logging.info("Statistics")
        tf.logging.info('-----------------------')

        # Elapsed iteration time = max of worker starting times on one iteration
        # - max of worker starting times on the previous
        elapsed_times = [max(self.iteration_start_times[iteration-1]) - \
                         max(self.iteration_start_times[iteration-2])]

        if min(elapsed_times) < .1:
          tf.logging.info(self.iteration_start_times[iteration-1])
          tf.logging.info(self.iteration_start_times[iteration-2])
          tf.logging.info(elapsed_times)

        self.iteration_times.extend(elapsed_times)

        # Calculate stats on elapsed time
        self.elapsed_max_time, self.elapsed_avg_time, \
          self.elapsed_min_time, self.elapsed_stdev_time = max(self.iteration_times), sum(self.iteration_times) / float(len(self.iteration_times)), \
                                                           min(self.iteration_times), np.std(self.iteration_times)

        # Print stats on elapsed time
        if len(self.iteration_times) > 1:
          tf.logging.info("Running max of iteration times: %f" % (self.elapsed_max_time))
          tf.logging.info("Running avg of iteration times: %f" % (self.elapsed_avg_time))
          tf.logging.info("Running min of iteration times: %f" % (self.elapsed_min_time))
          tf.logging.info("Running stdev of iteration times: %f" % (self.elapsed_stdev_time))

        tf.logging.info('-----------------------')

    #self.check_is_straggler()

    other_worker_iters = [x for i,x in enumerate(self.iteration_track) if i != worker_id]
    is_last_to_start = len(other_worker_iters) == len([x for x in other_worker_iters if iteration <= x])

    #if is_last_to_start:
    if worker_id == self.worker_id:
      tf.logging.info("%d if the last to starter iter %d" % (worker_id, iteration))
      self.set_suicide_timeout(cur_time, iteration)
    return 0

  def remote_notify_ready_to_start(self):
    tf.logging.info("Server ready to start!")
    self.ready_to_start = True

  def remote_is_ready_to_start(self):
    return (self.worker_id, self.ready_to_start)

class WorkerStatusClient:
  def __init__(self):
    self.worker_id = FLAGS.task_id
    hosts = FLAGS.worker_hosts.split(",")
    hosts = [x.split(":")[0] for x in hosts]
    self.hosts = hosts
    self.self_perspective = None
    self.perspectives = []
    self.ready = False
    self.servers_ready = set([])

    for i, host in enumerate(hosts):
      factory = pb.PBClientFactory()
      tf.logging.info("Connecting to %s:%d" % (host, FLAGS.rpc_port))
      reactor.connectTCP(host, FLAGS.rpc_port, factory)
      if i == self.worker_id:
        factory.getRootObject().addCallbacks(self.connected_self, self.connect_failure, errbackArgs=[host], errbackKeywords=[])
      else:
        factory.getRootObject().addCallbacks(self.connected, self.connect_failure, errbackArgs=[host], errbackKeywords=[])

  def server_ready_to_start(self, *args):
    wid, ready = args[0]
    if ready:
      tf.logging.info("Worker %d is ready to begin..." % wid)
      self.servers_ready.add(wid)

  def check_ready_to_start(self):
    for persp in self.perspectives:
      persp.callRemote("is_ready_to_start").addCallbacks(self.server_ready_to_start, self.fail)

  def ready_to_start(self):
    return self.ready and len(self.servers_ready) == len(self.hosts)

  def signal_server_ready(self):
    tf.logging.info("Signaling ready to self's server")
    self.self_perspective.callRemote("notify_ready_to_start").addCallbacks(self.success, self.fail)

  def notify_self_server_suicide_signal_received(self, time):
    self.self_perspective.callRemote("suicide_signal_received", time)

  def broadcast_starting(self, iteration):
    for persp in self.perspectives:
      persp.callRemote("notify_starting", self.worker_id, iteration).addCallbacks(self.success, self.fail)

  def connected(self, perspective):
    self.perspectives.append(perspective)
    tf.logging.info("Connected!")
    self.ready = (len(self.hosts) == len(self.perspectives))
    if self.ready:
      tf.logging.info("Ready!")
      self.signal_server_ready()
    else:
      tf.logging.info("%d of %d" % (len(self.perspectives), len(self.hosts)))

  def connected_self(self, perspective):
    self.self_perspective = perspective
    self.connected(perspective)

  def success(self, result):
    #tf.logging.info("Success!")
    pass

  def fail(self, _):
    tf.logging.info("Fail")
    tf.logging.info(_)

  def connect_failure(self, *args, **kwargs):
    tf.logging.info("RPC error, something failed: ")
    time.sleep(1)
    host = "".join(args[1:])
    factory = pb.PBClientFactory()
    tf.logging.info("Trying reconnecting to %s:%d" % (host, FLAGS.rpc_port))
    reactor.connectTCP(host, FLAGS.rpc_port, factory)
    factory.getRootObject().addCallbacks(self.connected, self.connect_failure, errbackArgs=(host))

# Separate manager process to oversee training on workers.
def launch_manager():
  # Launch a separate thread in the background that checks whether the
  # machine is a straggler.
  rpc_server = pb.PBServerFactory(WorkerStatusServer())
  reactor.listenTCP(FLAGS.rpc_port, rpc_server)
  rpc_client = WorkerStatusClient()
  Thread(target=reactor.run, args=(False,)).start()

  while not rpc_client.ready_to_start():
    rpc_client.check_ready_to_start()
    time.sleep(1)

  return rpc_client, rpc_server

def train(target, dataset, cluster_spec):

  if FLAGS.timeout_method:
    rpc_client, rpc_server = launch_manager()

  """Train Inception on a dataset for a number of steps."""
  # Number of workers and parameter servers are infered from the workers and ps
  # hosts string.
  num_workers = len(cluster_spec.as_dict()['worker'])
  num_parameter_servers = len(cluster_spec.as_dict()['ps'])
  # If no value is given, num_replicas_to_aggregate defaults to be the number of
  # workers.
  if FLAGS.num_replicas_to_aggregate == -1:
    num_replicas_to_aggregate = num_workers
  else:
    num_replicas_to_aggregate = FLAGS.num_replicas_to_aggregate

  # Both should be greater than 0 in a distributed training.
  assert num_workers > 0 and num_parameter_servers > 0, (' num_workers and '
                                                         'num_parameter_servers'
                                                         ' must be > 0.')

  # Choose worker 0 as the chief. Note that any worker could be the chief
  # but there should be only one chief.
  is_chief = (FLAGS.task_id == 0)

  # Ops are assigned to worker by default.
  with tf.device(
      tf.train.replica_device_setter(
        worker_device='/job:worker/task:%d' % FLAGS.task_id,
        cluster=cluster_spec)):

    # Create a variable to count the number of train() calls. This equals the
    # number of updates applied to the variables. The PS holds the global step.
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (dataset.num_examples / FLAGS.batch_size)

    # Decay steps need to be divided by the number of replicas to aggregate.
    # This was the old decay schedule. Don't want this since it decays too fast with a fixed learning rate.
    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay / num_replicas_to_aggregate)
    # New decay schedule. Decay every few steps.
    #decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    FLAGS.learning_rate_decay_factor,
                                    staircase=True)

    # Add a summary to track the learning rate.
    tf.scalar_summary('learning_rate', lr)

    images, labels = mnist.placeholder_inputs(FLAGS.batch_size)

    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    logits, reg = mnist.inference(images, train=True)

    # Add classification loss.
    total_loss = mnist.loss(logits, labels) + reg

    # Create an optimizer that performs gradient descent.
    opt = tf.train.AdamOptimizer(lr)

    # Use V2 optimizer
    if not FLAGS.timeout_method:
      opt = tf.train.SyncReplicasOptimizerV2(
        opt,
        replicas_to_aggregate=num_replicas_to_aggregate,
        total_num_replicas=num_workers)
    else:
      opt = TimeoutReplicasOptimizer(
        opt,
        global_step,
        total_num_replicas=num_workers)

    # Compute gradients with respect to the loss.
    grads = opt.compute_gradients(total_loss)
    if not FLAGS.timeout_method:
      apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)
    else:
      apply_gradients_op = opt.apply_gradients(grads, FLAGS.task_id, global_step=global_step)
      timeout_op = opt.timeout_op
      wait_op = opt.wait_op

    with tf.control_dependencies([apply_gradients_op]):
      train_op = tf.identity(total_loss, name='train_op')
    # Get chief queue_runners, init_tokens and clean_up_op, which is used to
    # synchronize replicas.
    # More details can be found in sync_replicas_optimizer.
    chief_queue_runners = [opt.get_chief_queue_runner()]
    init_tokens_op = opt.get_init_tokens_op()
    #clean_up_op = opt.get_clean_up_op()

    # Create a saver.
    saver = tf.train.Saver()

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init_op = tf.initialize_all_variables()

    # We run the summaries in the same thread as the training operations by
    # passing in None for summary_op to avoid a summary_thread being started.
    # Running summaries and training operations in parallel could run out of
    # GPU memory.
    if is_chief:
      local_init_op = opt.chief_init_op
    else:
      local_init_op = opt.local_step_init_op

    local_init_opt = [local_init_op]
    ready_for_local_init_op = opt.ready_for_local_init_op

    sv = tf.train.Supervisor(is_chief=is_chief,
                             local_init_op=local_init_op,
                             ready_for_local_init_op=ready_for_local_init_op,
                             logdir=FLAGS.train_dir,
                             init_op=init_op,
                             summary_op=None,
                             global_step=global_step,
                             saver=saver,
                             save_model_secs=FLAGS.save_interval_secs)


    tf.logging.info('%s Supervisor' % datetime.now())

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement)

    # Get a session.
    sess = sv.prepare_or_wait_for_session(target, config=sess_config)

    # Start the queue runners.
    queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
    sv.start_queue_runners(sess, queue_runners)
    tf.logging.info('Started %d queues for processing input data.',
                    len(queue_runners))

    if is_chief:
      sv.start_queue_runners(sess, chief_queue_runners)
      sess.run(init_tokens_op)

    # Train, checking for Nans. Concurrently run the summary operation at a
    # specified interval. Note that the summary_op and train_op never run
    # simultaneously in order to prevent running out of GPU memory.
    next_summary_time = time.time() + FLAGS.save_summaries_secs
    begin_time = time.time()
    cur_iteration = 0
    iterations_finished = set()
    while not sv.should_stop():
      try:

        # Timeout method
        if FLAGS.timeout_method:
          #sess.run([wait_op])
          #cur_iteration = int(sess.run(global_step))
          if len(iterations_finished) == 0:
            cur_iteration = 0
          else:
            cur_iteration = max(iterations_finished) + 1
          tf.logging.info("Starting iteration... %d" % cur_iteration)
          iterations_finished.add(cur_iteration)
          rpc_client.broadcast_starting(cur_iteration)

        start_time = time.time()
        feed_dict = mnist.fill_feed_dict(dataset, images, labels, FLAGS.batch_size)

        if FLAGS.timeline_logging:
          run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          run_metadata = tf.RunMetadata()
          loss_value, step = sess.run([train_op, global_step], options=run_options, run_metadata=run_metadata, feed_dict=feed_dict)
        else:
          loss_value, step = sess.run([train_op, global_step], feed_dict=feed_dict)

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

        duration = time.time() - start_time
        examples_per_sec = FLAGS.batch_size / float(duration)
        format_str = ('Worker %d: %s: step %d, loss = %f'
                      '(%.1f examples/sec; %.3f  sec/batch)')
        tf.logging.info(format_str %
                        (FLAGS.task_id, datetime.now(), step, loss_value,
                           examples_per_sec, duration))

        # Determine if the summary_op should be run on the chief worker.
        if is_chief and next_summary_time < time.time() and FLAGS.should_summarize:

          tf.logging.info('Running Summary operation on the chief.')
          summary_str = sess.run(summary_op)
          sv.summary_computed(sess, summary_str)
          tf.logging.info('Finished running Summary operation.')

          # Determine the next time for running the summary.
          next_summary_time += FLAGS.save_summaries_secs
      except Exception, e:
        rpc_client.notify_self_server_suicide_signal_received(time.time())
        tf.logging.info("%s" % e)
        sess.run([timeout_op])

    if is_chief:
      tf.logging.info('Elapsed Time: %f' % (time.time()-begin_time))

    # Stop the supervisor.  This also waits for service threads to finish.
    sv.stop()

    # Save after the training ends.
    if is_chief:
      saver.save(sess,
                 os.path.join(FLAGS.train_dir, 'model.ckpt'),
                 global_step=global_step)
