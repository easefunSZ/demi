# --- Implements the re-solving interface used by @{resolving} with functions
# -- that do nothing.
# --
# -- Used for debugging.
# -- @classmod mock_resolving
import sys
import os
sys.path.insert(0, '../TerminalEquity')
sys.path.insert(0, os.path.abspath('../Tree'))
sys.path.insert(0, os.path.abspath('../Game'))
sys.path.insert(0, os.path.abspath('../Settings'))
import arguments
from lookahead import Lookahead
import cfrd_gadget
import tree_builder
import tree_visualiser

from constants import constants
from card_tool import CardTool
import tools
import game_settings


class MockResolving(object):

    ##--- Constructor
    # def __init__(self):
    #     print('this is MockResolving')


    # --- Does nothing.
    # -- @param node the node to "re-solve"
    # -- @param[opt] player_range not used
    # -- @param[opt] opponent_range not used
    # -- @see resolving.resolve_first_node
    def resolve_first_node(self, node, player_range, opponent_range):
        self.node = node
        self.action_count = self.node.actions.size(1)


    # --- Does nothing.
    # -- @param node the node to "re-solve"
    # -- @param[opt] player_range not used
    # -- @param[opt] opponent_cfvs not used
    # -- @see resolving.resolve
    def resolve(self, node, player_range, opponent_cfvs):
        self.node = node
        self.action_count = self.node.actions.size(1)


    # --- Gives the possible actions at the re-solve node.
    # -- @return the actions that can be taken at the re-solve node
    # -- @see resolving.get_possible_actions
    def get_possible_actions(self):
        return self.node.actions


    # --- Returns an arbitrary vector.
    # -- @return a vector of 1s
    # -- @see resolving.get_root_cfv
    def get_root_cfv(self):
        return arguments.Tensor(game_settings.card_count).fill(1)


    # --- Returns an arbitrary vector.
    # -- @param[opt] action not used
    # -- @return a vector of 1s
    # -- @see resolving.get_action_cfv
    def get_action_cfv(self, action):
        return arguments.Tensor(game_settings.card_count).fill(1)


    # --- Returns an arbitrary vector.
    # -- @param[opt] player_action not used
    # -- @param[opt] board not used
    # -- @return a vector of 1s
    # -- @see resolving.get_chance_action_cfv
    def get_chance_action_cfv(self, player_action, board):
        return arguments.Tensor(game_settings.card_count).fill(1)


    # --- Returns an arbitrary vector.
    # -- @param[opt] action not used
    # -- @return a vector of 1s
    # -- @see resolving.get_action_strategy
    def get_action_strategy(self, action):
        return arguments.Tensor(game_settings.card_count).fill(1)
