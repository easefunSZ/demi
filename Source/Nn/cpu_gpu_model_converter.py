''''
--- Generates a neural net model in CPU format from a neural net model saved
-- in GPU format.
-- @script cpu_gpu_model_converter
'''''
import sys
import torch

sys.path.insert(0, '../Settings')
sys.path.insert(0, '../Game')
sys.path.insert(0, '../Nn')

# require 'cunn'
import arguments

'''
--- Generates a neural net model in CPU format from a neural net model saved
-- in GPU format.
-- @param gpu_model_path the prefix of the path to the gpu model, which is
-- appended with `_gpu.info` and `_gpu.model`
'''


def convert_gpu_to_cpu(gpu_model_path):
    info = torch.load(gpu_model_path + '_gpu.info')
    assert (info.gpu)
    info.gpu = False

    model = torch.load(gpu_model_path + '_gpu.model')
    model = model.float()

    torch.save(gpu_model_path + '_cpu.info', info)
    torch.save(gpu_model_path + '_cpu.model', model)


'''
--- Generates a neural net model in GPU format from a neural net model saved
-- in CPU format.
-- @param cpu_model_path the prefix of the path to the cpu model, which is
-- appended with `_cpu.info` and `_cpu.model`
'''


def convert_cpu_to_gpu(cpu_model_path):
    info = torch.load(cpu_model_path + '_cpu.info')
    assert (not info.gpu)
    info.gpu = True

    model = torch.load(cpu_model_path + '_cpu.model')
    model = model.cuda()

    torch.save(cpu_model_path + '_gpu.info', info)
    torch.save(cpu_model_path + '_gpu.model', model)


convert_gpu_to_cpu('+/Data/Models/PotBet/final')
