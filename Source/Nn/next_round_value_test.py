import sys

sys.path.insert(0, '../Settings')
sys.path.insert(0, '../Game')
sys.path.insert(0, '../Nn')

import next_round_value
import mock_nn
import mock_nn_terminal
from TerminalEquity import terminal_equity

import torch
import value_nn



import arguments
import game_settings
import card_to_string
import card_tools


next_round_value =  NextRoundValue()
#--print(next_round_value._range_matrix)
#--[[ test of card to bucket range translation
range = torch.range(1, 6).float().view(1, -1)
next_round_range = arguments.Tensor(1, next_round_value.bucket_count * next_round_value.board_count)
next_round_value._card_range_to_bucket_range(range, next_round_range)
print(next_round_range)
#]]

#--test of get_value functionality
mock_nn = MockNnTerminal()
#--local mock_nn = ValueNn()
next_round_value = NextRoundValue(mock_nn)

#--local bets = torch.range(1,1):float():mul(100)
bets = torch.Tensor(1).fill(1200)

next_round_value.start_computation(bets)

ranges = arguments.Tensor(1, 2, game_settings.card_count).fill(1/4)
values = arguments.Tensor(1, 2, game_settings.card_count)


x = arguments.Tensor()
torch.manualSeed(0)
ranges[1][1].copy(torch.Tensor({1,1,0,0,0,0}))
ranges[1][2].copy(torch.Tensor({1,1,1,1,1,1}))

next_round_value.get_value(ranges, values)

print(values)

#----[[
ranges_2 = ranges.view(2, game_settings.card_count).clone()
values_2 = ranges_2.clone().fill(-1)

terminal_equity = TerminalEquity()
terminal_equity.set_board(torch.Tensor)
terminal_equity.call_value(ranges_2, values_2)
print('terminal_equity')
print(values_2)
#---]]

#--[[
# board = card_to_string.string_to_board('Ks')
#
# values_3 = values.clone().fill(-1)
# next_round_value.get_value_on_board(board, values_3)
#
# print(values_3)
#]]



