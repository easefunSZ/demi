# --- Converts between string and numeric representations of cards.
# -- @module card_to_string_conversion
import sys
import os
import random
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0,'../Game')
sys.path.insert(0,'../Settings')
import game_settings
import math

# require "string"
# require "torch"
# local arguments = require 'Settings.arguments'
# local  game_settings =  require 'Settings.game_settings'
#

class CardToString(object):

    # --- Gets the suit of a card.
    # -- @param card the numeric representation of the card
    # -- @return the index of the suit
    def card_to_suit(self, card):
        return int(card % self.m['suit_count'] + 1)

    # --- Gets the rank of a card.
    # -- @param card the numeric representation of the card
    # -- @return the index of the rank
    def card_to_rank(self, card):
        return int(math.floor((card -1) / self.m['suit_count']) + 1)

    def __init__(self):
        self.m = game_settings.basic_setting()
        # ---All possible card suits - only the first 2 are used in Leduc Hold'em.
        self.suit_table = ['h', 's', 'c', 'd']

        # ---All possible card ranks - only the first 3-4 are used in Leduc Hold'em and
        # -- variants.
        self.rank_table = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']

        # --- Holds the string representation for every possible card, indexed by its
        # -- numeric representation.
        self.card_to_string_table = []

        for card in range(0, self.m['card_count']):
            rank_name = self.rank_table[self.card_to_rank(card+1)]
            suit_name = self.suit_table[self.card_to_suit(card+1)]
            self.card_to_string_table.append(rank_name + suit_name)

        self.string_to_card_table ={}
        for card in range(0, self.m['card_count']):
            self.string_to_card_table[self.card_to_string_table[card]] = card


    def card_to_string(self, card):
        return self.card_to_string_table[card -1]

    # --- Converts several cards' numeric representations to their string
    # -- representations.
    # -- @param cards a vector of numeric representations of cards
    # -- @return a string containing each card's string representation, concatenated
    def cards_to_string(self, cards):
        if(len(cards) == 0):
            return ""
        out = ""
        for i in range(0, len(cards)):
            out += self.card_to_string(cards[i])
        return out

    #
    # --- Converts a card's string representation to its numeric representation.
    # -- @param card_string the string representation of a card
    # -- @return the numeric representation of the card
    def string_to_card(self, card_string):
        return self.string_to_card_table[card_string]

    # --- Converts a string representing zero or one board cards to a
    # -- vector of numeric representations.
    # -- @param card_string either the empty string or a string representation of a
    # -- card
    # -- @return either an empty tensor or a tensor containing the numeric
    # -- representation of the card
    def string_to_board(self, card_string):
        if(card_string == ''):
            return []
        result =[]
        result.append(self.string_to_card(card_string))
        return result

cardToString = CardToString();
print(cardToString.card_to_string_table)
for card in range(1, 7):
    print(str(cardToString.card_to_suit(card)) + ": " + str(cardToString.card_to_rank(card)))
    print(cardToString.card_to_string(card))
cards = [1, 2, 3, 4, 5,6]
print(cardToString.cards_to_string(cards))
print(cardToString.string_to_card_table)
print(cardToString.string_to_card('Ks'))
print(cardToString.string_to_board('Ks'))