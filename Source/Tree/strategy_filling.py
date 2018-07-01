'''''
--- Fills a game's public tree with a uniform strategy. In particular, fills
-- the chance nodes with the probability of each outcome.
--
-- A strategy is represented at each public node by a NxK tensor where:
--
-- * N is the number of possible child nodes.
--
-- * K is the number of information sets for the active player in the public
-- node. For the Leduc Hold'em variants we implement, there is one for each
-- private card that the player could hold.
--
-- For a player node, `strategy[i][j]` gives the probability of taking the
-- action that leads to the `i`th child when the player holds the `j`th card.
--
-- For a chance node, `strategy[i][j]` gives the probability of reaching the
-- `i`th child for either player when that player holds the `j`th card.
-- @classmod strategy_filling
'''''
import sys
import os

# sys.path.insert(0,'../../Settings')

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, '../Game')
sys.path.insert(0, '../Settings')
import arguments
from constants import constants
import game_settings
from card_tool import CardTool
import math
import numpy as np

game_settings_M = game_settings.basic_setting()

class StrategyFilling(object):

    ##--- Constructor
    def _init_(self):
        self.value = 0
        self.game

    '''
    --- Fills a chance node with the probability of each outcome.
    -- @param node the chance node
    -- @local
    '''

    def _fill_chance(self, node):
        assert (not ('terminal' in node and node['terminal']))

        '''
        --filling strategy
        --we will fill strategy with an uniform probability, but it has to be zero for hands that are not possible on
        --corresponding board
        '''

        node['strategy'] = np.full([len(node['children']), game_settings_M['card_count']], 0.0)
        ##--setting probability of impossible hands to 0
        for i in range(0, len(node['children'])):
            child_node = node['children'][i]
            card_tools = CardTool() # correct way?
            possible_hand_indexes = card_tools.get_possible_hand_indexes(child_node['board'])

            node['strategy'][i].fill(0)
            for card in range(0, len(node['strategy'])):
            ##--remove 2 because each player holds one card
                if possible_hand_indexes[card] != 0:
                    node['strategy'][i][card] = 1.0 / (game_settings_M['card_count'] - 2)

    '''
      --- Fills a player node with a uniform strategy.
    -- @param node the player node
    -- @local
  
    '''

    def _fill_uniformly(self, node):
        assert (node['current_player'] == constants['players']['P1'] or node['current_player'] == constants['players']['P2'])

        if 'terminal' in node and node['terminal']:
            return

        node['strategy'] = np.full([len(node['children']), game_settings_M['card_count']], (1.0 / len(node['children'])), dtype=float)

    '''
    --- Fills a node with a uniform strategy and recurses on the children.
    -- @param node the node
    -- @local
    '''

    def _fill_uniform_dfs(self, node):
        if node['current_player'] == constants['players']['chance']:
            self._fill_chance(node)
        else:
            self._fill_uniformly(node)

        for i in range(0, len(node['children'])):
            self._fill_uniform_dfs(node['children'][i])

    '''
    --- Fills a public tree with a uniform strategy.
    -- @param tree a public tree for Leduc Hold'em or variant
    '''

    def fill_uniform(self, tree):
        self._fill_uniform_dfs(tree)
