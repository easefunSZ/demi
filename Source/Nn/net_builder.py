'''''
--- Builds the neural net architecture.
-- 
-- Uses torch's [nn package](https.//github.com/torch/nn/blob/master/README.md).
-- 
-- For M buckets, the neural net inputs have size 2*M+1, containing range 
-- vectors over buckets for each player, as well as a feature capturing the 
-- pot size. These are arranged as [{p1\_range}, {p2\_range}, pot\_size].
--
-- The neural net outputs have size 2*M, containing counterfactual value 
-- vectors over buckets for each player. These are arranged as 
-- [{p1\_cfvs}, {p2\_cfvs}].
-- @module net_builder
'''''
M = {}

print "Loading Net Builder"
require "nn"
require 'torch'
require 'math'
require 'Nn.bucketer'


arguments = require 'Settings.arguments'
game_settings = require 'Settings.game_settings'

--import GPU modules if needed
if arguments.gpu then
  require 'cunn'
  require 'cutorch'  
end

--- Builds a neural net with architecture specified by @{arguments.net}.
-- @return a newly constructed neural net
def M.build_net()
  bucketer = Bucketer()
  bucket_count = bucketer.get_bucket_count()
  player_count = 2
  output_size = bucket_count * player_count
  input_size = output_size + 1
  
  --run the lua interpreter on the architecture from the command line to get the list of layers
  layers_text = 'return ' .. arguments.net
  layers_text = string.gsub(layers_text, 'input_size', input_size)
  layers_text = string.gsub(layers_text, 'output_size', output_size)
  f = loadstring(layers_text)
  layers = f() 
  
  feedforward_part = nn.Sequential()
  
  --build the network from the layers  
  for _k, layer in pairs(layers) do
    feedforward_part.add(layer)
  end
  
  right_part = nn.Sequential()
  right_part.add(nn.Narrow(2, 1, output_size))
  
  first_layer = nn.ConcatTable()
  first_layer.add(feedforward_part)
  first_layer.add(right_part)
  
  left_part_2 = nn.Sequential()
  left_part_2.add(nn.SelectTable(1))
  
  right_part_2  = nn.Sequential()
  right_part_2.add(nn.DotProduct())
  right_part_2.add(nn.Replicate(output_size, 2))
  right_part_2.add(nn.MulConstant(-0.5)) 
  
  second_layer = nn.ConcatTable()
  second_layer.add(left_part_2)
  second_layer.add(right_part_2)
  
  final_mlp = nn.Sequential()
  final_mlp.add(first_layer)
  final_mlp.add(second_layer)
  --final layer that used delta
  final_mlp.add(nn.CAddTable())
  
  return final_mlp
  
end

return M