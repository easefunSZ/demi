import tensorflow as tf
import numpy as np
#flags = tf.app.flags
tf.flags.DEFINE_integer('epoch', 25, 'Epoch to train [25]')
tf.flags.DEFINE_string('gpu_no', '0', 'gpu_no')
tf.flags.DEFINE_string('acpc_server','localhost','acpc_server')
tf.flags.DEFINE_integer('acpc_server_port',20000,'acpc_server_port')
tf.flags.DEFINE_integer('street_count',2,'strees_count')
tf.flags.DEFINE_integer('ante', 100, 'the size of the game ante, in chips')
tf.flags.DEFINE_integer('stack', 1200, 'stack')
tf.flags.DEFINE_integer('cfr_iters', 1000, 'cfr_iters')
tf.flags.DEFINE_integer('cfr_skip_iters', 500, 'cfr_skip_iters')
tf.flags.DEFINE_integer('gen_batch_size', 10, 'gen_batch_size')
tf.flags.DEFINE_integer('train_batch_size', 100, 'train_batch_size')
tf.flags.DEFINE_string('data_path', '../Data/TrainSamples/PotBet/', 'data_path')
tf.flags.DEFINE_string('model_path', '../Data/Models/PotBet/', 'model_path')
tf.flags.DEFINE_string('value_net_name', 'final', 'value_net_name')
tf.flags.DEFINE_string('net', '{nn.Linear(input_size, 50), nn.PReLU(), nn.Linear(50, output_size)}', 'net')
tf.flags.DEFINE_integer('save_epoch', 2, 'save_epoch')
tf.flags.DEFINE_integer('epoch_count', 10, 'epoch_count')
tf.flags.DEFINE_integer('train_data_count', 100, 'train_data_count')
tf.flags.DEFINE_integer('valid_data_count', 100, 'valid_data_count')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
FLAGS = tf.flags.FLAGS
params = {}
params['gpu']=False
params['bet_sizing']={1}
params['acpc_server']= FLAGS.acpc_server
params['acpc_server_port']=FLAGS.acpc_server_port
params['street_count']=FLAGS.street_count
params['ante']=FLAGS.ante
params['stack']=FLAGS.stack
params['cfr_iters']=FLAGS.cfr_iters
params['cfr_skip_iters']=FLAGS.cfr_skip_iters
params['gen_batch_size']=FLAGS.gen_batch_size
params['train_batch_size']=FLAGS.train_batch_size
params['data_path']=FLAGS.data_path
params['model_path']=FLAGS.model_path
params['value_net_name']=FLAGS.value_net_name
params['net']=FLAGS.net
params['save_epoch']=FLAGS.save_epoch
params['epoch_count']=FLAGS.epoch_count
params['train_data_count']=FLAGS.train_data_count
params['valid_data_count']=FLAGS.valid_data_count
params['learning_rate']=FLAGS.learning_rate
params['Tensor']=np.zeros([2,2],dtype=float) # try not use this


