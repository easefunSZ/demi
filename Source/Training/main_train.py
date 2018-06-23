''''
--- Script that trains the neural network.
-- 
-- Uses data previously generated with @{data_generation_call}.
-- @script main_train
'''''

nnBuilder = require 'Nn.net_builder'
'Training.data_stream'
train = require 'Training.train'
arguments = require 'Settings.arguments'

  
##--build the network
network = nnBuilder.build_net()

if arguments.gpu:
  network = network.cuda()

data_stream = DataStream()
train.train(network, data_stream, arguments.epoch_count)
