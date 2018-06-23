--- Wraps the calls to the final neural net.
-- @classmod value_nn

require 'torch'
require 'nn'
arguments = require 'Settings.arguments'

class ValueNn(object):

  ##--- Constructor. Loads the neural net from disk.
  def __init__(self):
    net_file = arguments.model_path + arguments.value_net_name

    ##--0.0 select the correct model cpu/gpu
    if arguments.gpu:
      net_file = net_file + '_gpu'
    else:
      net_file = net_file + '_cpu'

    ##--1.0 load model information
    model_information = torch.load(net_file + '.info')

    print('NN information.')
    for k, v in pairs(model_information):
      print(k, v)

    ##--import GPU modules only if needed
    if arguments.gpu:
      require 'cunn'
      require 'cutorch'

    ##--2.0 load model
    self.mlp = torch.load(net_file + '.model')
    print('NN architecture.')
    print(self.mlp)

  '''
  --- Gives the neural net output for a batch of inputs.
  -- @param inputs An NxI tensor containing N instances of neural net inputs.
  -- See @{net_builder} for details of each input.
  -- @param output An NxO tensor in which to store N sets of neural net outputs.
  -- See @{net_builder} for details of each output.
  '''
  def get_value(self,inputs, output):
    output.copy(self.mlp.forward(inputs))
