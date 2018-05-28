import sys
import os
import random
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0,'/Users/lidingcheng/Documents/software/demi/Source/Game')
sys.path.insert(0,'/Users/lidingcheng/Documents/software/demi/Source/Settings')
from Settings import arguments
from Settings import constants
from Settings import game_settings

class DataGeneration:
    def __init__(self):
        pass

    # Sample a set of random cards.
    # uniform distribution
    def generate_cards(self, count):
        M = game_settings.basic_setting()
        cards = []
        for x in range(1, M['card_count'] + 1):
            cards.append(x)
        print(cards)
        random.shuffle(cards)
        return cards[0:count]

dataGeneration = DataGeneration()

result = dataGeneration.generate_cards(3)
print("result: ")
print(result)
#
# ##the number of card ranks in the deck
# M['rank_count'] = 2
# ##the total number of cards in the deck
# M['card_count'] = 2
# ##the number of public cards dealt in the game (revealed after the first-- betting round)
# M['board_card_count'] = 1
# ##the number of players in the game
# M['player_count'] = 2



