# --- A set of tools for basic operations on cards and sets of cards.
# --
# -- Several of the functions deal with "range vectors", which are probability
# -- vectors over the set of possible private hands. For Leduc Hold'em,
# -- each private hand consists of one card.
# -- @module card_tools
# local game_settings = require 'Settings.game_settings'
# local arguments = require 'Settings.arguments'
# local constants = require 'Settings.constants'
import random
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0,'../Settings')

import game_settings
import math
import numpy as np


class CardTool(object):
    def __init__(self):
        self.m = game_settings.basic_setting()

    def hand_is_possible(self, hand):
        # print(min(hand)>0)
        # print(max(hand) <= self.m['card_count'])
        assert min(hand) >= 0 and max(hand) < self.m['card_count']
        used_cards = [False] * self.m['card_count']
        for i in range(0, len(hand)):
            if used_cards[hand[i]]:
                return False
            used_cards[hand[i]] = True
        return True

    # --- Gives the private hands which are valid with a given board.
    # -- @param board a possibly empty vector of board cards
    # -- @return a vector with an entry for every possible hand (private card), which
    # --  is `1` if the hand shares no cards with the board and `0` otherwise
    def get_possible_hand_indexes(self, board):
        out = [0] * self.m['card_count']
        if board == 0:
            return out

        whole_hand = np.zeros(board + 1, dtype=int)
        whole_hand[0:-1] = board
        for card in range(0, self.m['card_count']):
            whole_hand[-1] = card
            if self.hand_is_possible(whole_hand):
                out[card] = 1
        return out

    # --- Gives the private hands which are invalid with a given board.
    # -- @param board a possibly empty vector of board cards
    # -- @return a vector with an entry for every possible hand (private card), which
    # -- is `1` if the hand shares at least one card with the board and `0` otherwise
    def get_impossible_hand_indexes(self, board):
        out = self.get_possible_hand_indexes(board)
        for i in range(0, len(out)):
            out[i] = (out[i] - 1) * -1
        return out

    # --- Gives a range vector that has uniform probability on each hand which is
    # -- valid with a given board.
    # -- @param board a possibly empty vector of board cards
    # -- @return a range vector where invalid hands have 0 probability and valid
    # -- hands have uniform probability
    def get_uniform_range(self, board):
        out = self.get_possible_hand_indexes(board)
        total = sum(out)
        for i in range(0, len(out)):
            out[i] = out[i]/(total + 0.0)
        return out

    # --- Randomly samples a range vector which is valid with a given board.
    # -- @param board a possibly empty vector of board cards
    # -- @param[opt] seed a seed for the random number generator
    # -- @return a range vector where invalid hands are given 0 probability, each
    # -- valid hand is given a probability randomly sampled from the uniform
    # -- distribution on [0,1), and the resulting range is normalized
    def get_random_range(self, board,seed):
        out = self.get_uniform_range(seed)
        total = 0.0
        for i in range(0, len(out)):
            if out[i] != 0:
                out[i] += random.random()
                total += out[i]
        for i in range(0, len(out)):
            out[i] = out[i]/total
        return out

    # --- Checks if a range vector is valid with a given board.
    # -- @param range a range vector to check
    # -- @param board a possibly empty vector of board cards
    # -- @return `true` if the range puts 0 probability on invalid hands and has
    # -- total probability 1
    def is_valid_range(self, rangeVector, board):
        possibleHand = self.get_possible_hand_indexes(board)
        totalProb = 0.0
        for i in range(0, len(rangeVector)):
            if rangeVector[i] != 0 and possibleHand[i] == 0:
                return False
            totalProb += rangeVector[i]
        return math.fabs(1-totalProb) < 0.0001

    # --- Gives the current betting round based on a board vector.
    # -- @param board a possibly empty vector of board cards
    # -- @return the current betting round
    def board_to_street(self, board):
        if(len(board) == 0):
            return 1
        return 2

    # --- Gives the number of possible boards.
    # -- @return the number of possible boards
    def get_boards_count(self):
        if self.m['board_card_count'] == 1:
            return self.m['card_count']
        elif self.m['board_card_count'] == 2:
            return self.m['card_count'] * (self.m['card_count'] - 1) /2
        return self.m['card_count']

    # --- Gives all possible sets of board cards for the game.
    # -- @return an NxK tensor, where N is the number of possible boards, and K is
    # -- the number of cards on each board
    # only support 1 card for now
    def get_second_round_boards(self):
        # out = [0] * self.m['card_count']
        out = np.full([self.m['card_count'], 1], 0)
        if self.m['board_card_count'] == 1:
            for card in range(0, len(out)):
                out[card][0] = card
        return out

    # def _init_board_index_table(self):

    # --- Normalizes a range vector over hands which are valid with a given board.
    # -- @param board a possibly empty vector of board cards
    # -- @param range a range vector
    # -- @return a modified version of `range` where each invalid hand is given 0
    # -- probability and the vector is normalized
    def normalize_range(self, board, rangeVector):
        possibleHand = self.get_possible_hand_indexes(board)
        total = 0.0
        for i in range(0, len(rangeVector)):
            if(possibleHand[i] == 0):
                rangeVector[i] = 0
            total += rangeVector[i]
        for i in range(0, len(rangeVector)):
            rangeVector[i] = rangeVector[i] / total
        return rangeVector


# cardTool = CardTool()
# print(cardTool.hand_is_possible([2]))
# print(cardTool.get_possible_hand_indexes([2,3]))
# print(cardTool.get_impossible_hand_indexes([3]))
# print(cardTool.get_uniform_range([1]))
# print(cardTool.get_random_range([0]))
# print(cardTool.is_valid_range([0.1, 0.2, 0.3, 0.0, 0.1, 0.3], [3]))
# print(cardTool.board_to_street([1]))
# print(cardTool.get_boards_count())
# print(cardTool.get_second_round_boards())
# print(cardTool.normalize_range([1], [0.1, 0.5, 0.23, 0.6, 0.0, 1]))



#
# --- Initializes the board index table.
# -- @local
# function M:_init_board_index_table()
#   if game_settings.board_card_count == 1 then
#     self._board_index_table = torch.range(1, game_settings.card_count):float()
#   elseif game_settings.board_card_count == 2 then
#     self._board_index_table = arguments.Tensor(game_settings.card_count, game_settings.card_count):fill(-1)
#     local board_idx = 0;
#     for card_1 = 1, game_settings.card_count do
#       for card_2 = card_1 + 1, game_settings.card_count do
#         board_idx = board_idx + 1
#         self._board_index_table[card_1][card_2] = board_idx
#         self._board_index_table[card_2][card_1] = board_idx
#       end
#     end
#   else
#     assert(false, 'unsupported board size')
#   end
# end
#
# M:_init_board_index_table()
#
# --- Gives a numerical index for a set of board cards.
# -- @param board a non-empty vector of board cards
# -- @return the numerical index for the board
# function M:get_board_index(board)
#   local index = self._board_index_table
#   for i = 1, board:size(1) do
#     index = index[board[i]]
#   end
#   assert( index > 0, index)
#   return index
# end
#

#
#
# return M