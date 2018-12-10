import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../../Game'))
sys.path.insert(0,os.path.abspath('../../Settings'))
sys.path.insert(0,os.path.abspath('../../Tree'))
from arguments import params
import arguments
import constants
import game_settings
import bet_sizing
from card_tool import CardTool
import card_to_string_conversion
from card_to_string_conversion import CardToString
import math
from tree_builder import PokerTreeBuilder
from tree_cfr import TreeCFR
from tree_values import TreeValues

card_to_string = CardToString()
constants = constants.set_constants()
builder = PokerTreeBuilder()

params = {}
params['root_node'] = {}
params['root_node']['board'] = card_to_string.string_to_board('')
params['root_node']['street'] = 1
params['root_node']['current_player'] = constants['players']['P1']
params['root_node']['bets'] = np.zeros((1,1)).fill(100)

tree = builder.build_tree(params)
starting_ranges = arguments.Tensor(constants.players_count, game_settings.card_count)

starting_ranges[1].copy(CardTool.get_uniform_range(params['root_node']['board']))
starting_ranges[2].copy(CardTool.get_uniform_range(params['root_node']['board']))
#--starting_ranges[1]:copy(card_tools:get_random_range(params.root_node.board, 2))
#--starting_ranges[2]:copy(card_tools:get_random_range(params.root_node.board, 4))

tree_cfr = TreeCFR()
tree_cfr.run_cfr(tree, starting_ranges)

tree_values = TreeValues()
tree_values.compute_values(tree, starting_ranges)

print('Exploitability: %f [chips]' % tree.exploitability )

#--local visualiser = TreeVisualiser()
#--visualiser:graphviz(tree)