# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A binary to evaluate Inception on the flowers data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

import nn_eval
from cifar10_input import *

FLAGS = tf.app.flags.FLAGS

def main(unused_argv=None):
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    maybe_download_and_extract()
    eval_data, eval_labels = read_validation_data()
    nn_eval.evaluate(eval_data, eval_labels, out_dir=FLAGS.out_dir)


if __name__ == '__main__':
    tf.app.run()
