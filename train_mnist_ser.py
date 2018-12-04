#!/usr/bin/env python 

from mpi4py import MPI
import argparse
import sys
import os


import tensorflow as tf

FLAGS = None

def main(_):

  # Assigns ops to the local worker by default.
  # Build model...
  y_  = tf.Variable(tf.zeros([1]))
#      y   = tf.placeholder(tf.float32,[None,10])
#      loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
  loss = tf.reduce_mean(y_)
  train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
  # The StopAtStepHook handles stopping after running given steps.
  hooks=[tf.train.StopAtStepHook(last_step=2)]
  # The MonitoredTrainingSession takes care of session initialization,
  # restoring from a checkpoint, saving to a checkpoint, and closing when done
  # or an error occurs.
  mon_sess.run(train_op)
  print y_



if __name__ == "__main__":
  tf.app.run(main=main, argv=[sys.argv[0]] )


