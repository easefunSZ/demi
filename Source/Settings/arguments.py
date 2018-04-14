import tensorflow as tf
flags = tf.app.flags
flags.DEFINE_integer('epoch', 25, 'Epoch to train [25]')
flags.DEFINE_string('gpu_no', '0', 'gpu_no')
flags.DEFINE_string('acpc_server','localhost','acpc_server')
flags.DEFINE_integer('ante', 100, 'the size of the game ante, in chips')
