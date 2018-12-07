#!/usr/bin/env python3 

from mpi4py import MPI
import argparse
import sys
import os


from tensorflow.examples.tutorials.mnist import input_data

import tensorflow.app 
import tensorflow.train 

FLAGS = None

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  # Create a cluster from the parameter server and worker hosts.
  cluster = tensorflow.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  # Create and start a server for the local task.
  server = tensorflow.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

  print ("I'm the",FLAGS)
  print ("ps_hosts",ps_hosts)
  print ("worker_hosts",worker_hosts)

  # Import data
  mnist = input_data.read_data_sets("train")


  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    worker_device="/job:worker/task:%d" % FLAGS.task_index
    with tf.device(tf.train.replica_device_setter(worker_device=worker_device, cluster=cluster)):
      # Build model...
      y_  = tf.Variable(tf.zeros([1]))
      print ("---------------->", y_)
#      y   = tf.placeholder(tf.float32,[None,10])
#      loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
      loss = tf.reduce_mean(y_)
      global_step = tf.train.get_or_create_global_step()
      train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)
    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=2)]
    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(FLAGS.task_index == 0), checkpoint_dir="/galileo/home/userinternal/stagliav/tf_distributed/train_logs", hooks=hooks) as mon_sess:
      while not mon_sess.should_stop():
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        # mon_sess.run handles AbortedError in case of preempted PS.
        mon_sess.run(train_op)
  print ("-<-<-<->",y_[0])



if __name__ == "__main__":
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()
  name = MPI.Get_processor_name()

  dt = {'host':name,'rank':rank,'local_rank':0,'type':'ps','task':0}


  print("hostname:",name," number of processes:",size)

 
  # build local rank
  nodes = []
  lr = comm.allgather(dt)
  for i in lr:
    if i['host'] not in [x[1] for x in nodes]: 
      a=[lr.index(i),i['host']]
      nodes.append(a)
  local_rank = rank-[x[0] for x in nodes if name==x[1]][0]
  dt['local_rank']=local_rank
  if  rank==0:
    dt['type']='ps'    
  else:
    dt['type']='worker'
  comm_ps=comm.Split(dt['type']=='ps')
  comm_workers=comm.Split(dt['type']=='worker')

  if dt['type']=='ps':
    dt['task'] = comm_ps.Get_rank()  
  if dt['type']=='worker':
    dt['task'] = comm_workers.Get_rank()  

  dt = comm.allgather(dt)

  ps_list = []               #only  first rank
  for (ind,vv) in enumerate([nodes[0]]):
    port=2222
    ps_list.append(vv[1]+':'+str(port))
  ps_list=str(ps_list).strip("[]").replace('\'','') 

  worker_list = []
  for (ind,vv) in enumerate(dt):
    port=4444
    #if dt[ind]['type']=='worker':
    worker_list.append(vv.get("host")+':'+str(port))
  worker_list=str(worker_list).strip("[]").replace('\'','') 

  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument( "--ps_hosts", type=str, default="", help="Comma-separated list of hostname:port pairs")
  parser.add_argument( "--worker_hosts", type=str, default="", help="Comma-separated list of hostname:port pairs")
  parser.add_argument( "--job_name", type=str, default="", help="One of 'ps', 'worker'")
  parser.add_argument( "--task_index", type=int, default=0, help="Index of task within the job")

  FLAGS, unparsed    = parser.parse_known_args()
  FLAGS.ps_hosts     = ps_list
  FLAGS.worker_hosts = worker_list

  print ("ps:",FLAGS.ps_hosts)
  print ("worker:",FLAGS.worker_hosts)

  FLAGS.job_name   = dt[rank]['type']
  FLAGS.task_index = dt[rank]['task'] 

  tensorflow.app.run(main=main, argv=[sys.argv[0]] + unparsed)

