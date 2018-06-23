'''''
--- Implements the same interface as @{value_nn}, but without uses terminal
-- equity evaluation instead of a neural net.
-- 
-- Can be used to replace the neural net during debugging.
-- @classmod mock_nn_terminal
'''''

require 'torch'
require 'Nn.bucketer'
require 'TerminalEquity.terminal_equity'
game_settings = require  'Settings.game_settings'
card_tools = require 'Game.card_tools'
arguments = require 'Settings.arguments'

class MockNnTerminal(object):

'''
--- Constructor. Creates an equity matrix with entries for every possible
-- pair of buckets.
'''
def __init__(self):
  self.bucketer = Bucketer()
  self.bucket_count = self.bucketer.get_bucket_count()
  self.equity_matrix = arguments.Tensor(self.bucket_count, self.bucket_count).zero()
  ##--filling equity matrix
  boards = card_tools.get_second_round_boards()
  self.board_count = boards.size(1)
  self.terminal_equity = TerminalEquity()
  for i in range(1, self.board_count):
    board = boards[i]
    self.terminal_equity.set_board(board)
    call_matrix = self.terminal_equity.get_call_matrix()
    buckets = self.bucketer.compute_buckets(board)
    for c1 in range(1, game_settings.card_count):
      for c2 in range(1, game_settings.card_count):
        b1 = buckets[c1]
        b2 = buckets[c2]
        if( b1 > 0 and b2 > 0 ):
          matrix_entry = call_matrix[c1][c2]
          self.equity_matrix[b1][b2] = matrix_entry

'''
--- Gives the expected showdown equity of the two players' ranges.
-- @param inputs An NxI tensor containing N instances of neural net inputs. 
-- See @{net_builder} for details of each input.
-- @param outputs An NxO tensor in which to store N sets of expected showdown
-- counterfactual values for each player.
'''
def get_value(self,inputs, outputs):

  assert(outputs.dim() == 2 )
  bucket_count = outputs.size(2) / 2
  batch_size = outputs.size(1)
  player_indexes = {{1, self.bucket_count}, {self.bucket_count + 1, 2 * self.bucket_count}}
  players_count = 2
  for player in range(1, players_count):
    player_idx = player_indexes[player]
    opponent_idx = player_indexes[3- player]
    outputs[{{}, player_idx}].mm(inputs[{{}, opponent_idx}], self.equity_matrix)