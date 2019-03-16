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
import math
import numpy as np

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, '../Game')
sys.path.insert(0, '../Settings')
import game_settings
import card_to_string_conversion
import card_tool
import arguments
import torch
from arguments import params

card_to_string = card_to_string_conversion.CardToString()
card_tools = card_tool.CardTool()


# M = {}

# local M = {}
# --- Gives a strength representation for a hand containing two cards.
# -- @param hand_ranks the rank of each card in the hand
# -- @return the strength value of the hand
# -- @local
def evaluate_two_card_hand(hand_ranks):
    if hand_ranks[0] == hand_ranks[1]:
        # hand is a pair (refers to the pair in card game)
        hand_value = hand_ranks[0]
    else:
        # hand is a high card
        hand_value = hand_ranks[0] * game_settings.rank_count + hand_ranks[1]
    return hand_value


# --- Gives a strength representation for a hand containing three cards.
# -- @param hand_ranks the rank of each card in the hand
# -- @return the strength value of the hand
# -- @local
def evaluate_three_card_hand(hand_ranks):
    ##--check for the pair
    if hand_ranks[0] == hand_ranks[1]:
        ##--paired hand, value of the pair goes first, value of the kicker goes second
        hand_value = hand_ranks[0] * game_settings.rank_count + hand_ranks[2]
    elif hand_ranks[1] == hand_ranks[2]:
        ##--paired hand, value of the pair goes first, value of the kicker goes second
        hand_value = hand_ranks[1] * game_settings.rank_count + hand_ranks[0]
    else:
        ##--hand is a high card
        hand_value = hand_ranks[0] * game_settings.rank_count * game_settings.rank_count + hand_ranks[
            1] * game_settings.rank_count + hand_ranks[2]
    return hand_value


# --- Gives a strength representation for a two or three card hand.
# -- @param hand a vector of two or three cards
# -- @param[opt] impossible_hand_value the value to return if the hand is invalid
# -- @return the strength value of the hand, or `impossible_hand_value` if the
# -- hand is invalid
def evaluate(hand, impossible_hand_value):
    ##not sure what differences between lua assert and python assert
    assert np.max(hand) <= game_settings.card_count and np.min(hand) > 0, 'hand does not correspond to any cards'
    impossible_hand_value = impossible_hand_value or -1
    if not card_tool.hand_is_possible(hand):
        return impossible_hand_value
    ##--we are not interested in the hand suit - we will use ranks instead of cards
    hand_ranks = np.copy(hand)
    for i in range(len(hand_ranks)):
        hand_ranks[i] = card_to_string.card_to_rank(hand_ranks[i])
    hand_ranks = np.sort(hand_ranks)
    if hand[0] == 2:
        return evaluate_two_card_hand(hand_ranks)
    elif len(hand) == 3:
        return evaluate_three_card_hand(hand_ranks)
    else:
        assert('unsupported size of hand!')



# --- Gives strength representations for all private hands on the given board.
# -- @param board a possibly empty vector of board cards
# -- @param impossible_hand_value the value to assign to hands which are invalid
# -- on the board
# -- @return a vector containing a strength value or `impossible_hand_value` for
# -- every private hand
def batch_eval(board, impossible_hand_value):
    hand_values = arguments.Tensor(game_settings.card_count).full(-1)
    ##not sure what board.dim() is
    if board.size() == 0:
        for hand in range(game_settings.card_count):
            hand_values[hand] = math.floor((hand - 1) / game_settings.suit_count) + 1
    else:
        board_size = board.size()
        assert board_size == 1 or board_size == 2, 'Incorrect board size for Leduc'
        whole_hand = arguments.Tensor(board_size + 1)
        whole_hand = torch.FloatTensor
        whole_hand[-2] = board  ### it seems that whole_hand[{{1, -2}}]:copy(board) means that we copy board value into the second to the last of whole hand
        for card in range(1, game_settings.card_count):
            whole_hand[-1] = card
            hand_values[card] = evaluate(whole_hand, impossible_hand_value)
    return hand_values
