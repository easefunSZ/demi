'''''
--- Assigns hands to buckets on the given board.
-- 
-- For the Leduc implementation, we simply assign every possible set of
-- private and board cards to a unique bucket.
-- @classmod bucketer
'''''
game_settings = require 'Settings.game_settings'
card_tools = require 'Game.card_tools'

class bucketer(object):

  '''
  --- Gives the total number of buckets across all boards.
  -- @return the number of buckets
  '''
  def get_bucket_count(self):
    return game_settings.card_count * card_tools.get_boards_count()

  '''
  --- Gives a vector which maps private hands to buckets on a given board.
  -- @param board a non-empty vector of board cards
  -- @return a vector which maps each private hand to a bucket index
  '''
  def compute_buckets(self,board):
    shift = (card_tools.get_board_index(board) - 1) * game_settings.card_count
    buckets = torch.range(1, game_settings.card_count).float().add(shift)
    ##--impossible hands will have bucket number -1
    for i in range(1,board.size(1)):
      buckets[board[i]] = -1
    return buckets