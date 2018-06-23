# --- Evaluates hand strength in Leduc Hold'em and variants.
# --
# -- Works with hands which contain two or three cards, but assumes that
# -- the deck contains no more than two cards of each rank (so three-of-a-kind
# -- is not a possible hand).
# --
# -- Hand strength is given as a numerical value, where a lower strength means
# -- a stronger hand: high pair < low pair < high card < low card
# -- @module evaluator
#
# require 'torch'
# require 'math'
# local game_settings = require 'Settings.game_settings'
# local card_to_string = require 'Game.card_to_string_conversion'
# local card_tools = require 'Game.card_tools'
# local arguments = require 'Settings.arguments'
#
import sys
import os
import random

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, '../Game')
sys.path.insert(0, '../Settings')
import game_settngs


# local M = {}
# --- Gives a strength representation for a hand containing two cards.
# -- @param hand_ranks the rank of each card in the hand
# -- @return the strength value of the hand
# -- @local
class Evaluator(object):

    def __init__(self):
        self.m = game_settings.basic_setting()

    def evaluate_two_card_hand(self, hand_ranks):
        if hand_ranks[0] == hand_ranks[1]:
            # pair
            return hand_ranks[0]
        # high card
        return hand_ranks[0] * self.m['rank_count'] + hand_ranks[1]

    # --- Gives strength representations for all private hands on the given board.
    # -- @param board a possibly empty vector of board cards
    # -- @param impossible_hand_value the value to assign to hands which are invalid
    # -- on the board
    # -- @return a vector containing a strength value or `impossible_hand_value` for
    # -- every private hand
    # def batch_eval(self, board):

    #

    # function M:batch_eval(board, impossible_hand_value)
    #   local hand_values = arguments.Tensor(game_settings.card_count):fill(-1)
    #   if board:dim() == 0 then
    #     for hand = 1, game_settings.card_count do
    #       hand_values[hand] = math.floor((hand -1 ) / game_settings.suit_count ) + 1
    #     end
    #   else
    #     local board_size = board:size(1)
    #     assert(board_size == 1 or board_size == 2, 'Incorrect board size for Leduc' )
    #     local whole_hand = arguments.Tensor(board_size + 1)
    #     whole_hand[{{1, -2}}]:copy(board)
    #     for card = 1, game_settings.card_count do
    #       whole_hand[-1] = card;
    #       hand_values[card] = self:evaluate(whole_hand, impossible_hand_value)
    #     end
    #   end
    #   return hand_values
    # end
    #
    # return M
