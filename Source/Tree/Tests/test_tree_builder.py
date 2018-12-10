import os
import sys
import numpy as np
import math
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../../Game'))
sys.path.insert(0,os.path.abspath('../../Settings'))
sys.path.insert(0,os.path.abspath('../../Tree'))
from arguments import params
from constants import constants
import game_settings
import bet_sizing
import card_tool
from card_to_string_conversion import CardToString
import tree_builder
card_to_string = CardToString()
# constants = constants.set_constants()
builder = tree_builder.PokerTreeBuilder()
params = {}
params['root_node'] = {}
params['root_node']['board'] = card_to_string.string_to_board('')
params['root_node']['street'] = 1
params['root_node']['current_player'] = constants['players']['P1']
# params['root_node']['bets'] = np.zeros((1,1)).fill(100)
params['root_node']['bets'] = np.array([100, 100])

tree = builder.build_tree(params)
print('tree is built')