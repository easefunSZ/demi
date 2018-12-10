import os
import sys
import numpy as np
import math
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../../Game'))
sys.path.insert(0,os.path.abspath('../../Settings'))
sys.path.insert(0,os.path.abspath('../../Tree'))
sys.path.insert(0, '../Lookahead')

from arguments import params
import arguments
from constants import constants
import game_settings
import bet_sizing
import card_tool
from card_to_string_conversion import CardToString
import tree_builder
#from lookahead import Lookahead
from resolving import Resolving

resolving = Resolving()
current_node = {}

current_node['board'] = card_to_string.string_to_board('Ks')
current_node['street'] = 2
current_node['current_player'] = constants.players.P1
current_node['bets'] = np.array(100,100)

player_range = card_tool.get_random_range(current_node['board'], 2)
opponent_range = card_tool.get_random_range(current_node['board'], 4)

'''
##--resolving:resolve_first_node(current_node, player_range, opponent_range)
'''

resolving.resolve(current_node, player_range, opponent_range)

'''
#--[[
lookahead = Lookahead()

current_node = {}
current_node.board = card_to_string.string_to_board('Ks')
current_node.street = 2
current_node.current_player = constants.players.P1
current_node.bets = arguments.Tensor{100, 100}


lookahead.build_lookahead(current_node)
#]]

##--[[
starting_ranges = arguments.Tensor(constants.players_count, constants.card_count)
starting_ranges[1].copy(card_tools.get_random_range(current_node.board, 2))
starting_ranges[2].copy(card_tools.get_random_range(current_node.board, 4))

lookahead.resolve_first_node(starting_ranges)

lookahead.get_strategy()
#]]

#--[[
player_range = card_tools:get_random_range(current_node.board, 2)
opponent_cfvs = card_tools:get_random_range(current_node.board, 4)

lookahead.resolve(player_range, opponent_cfvs)


lookahead.get_results()
#]]
'''''