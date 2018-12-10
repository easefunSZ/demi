'''''
--- Builds the internal data structures of a @{lookahead|Lookahead} object.
-- @classmod lookahead_builder
'''''
# local arguments = require 'Settings.arguments'
# local constants = require 'Settings.constants'
# local game_settings = require 'Settings.game_settings'
# local tools = require 'tools'
# require 'Tree.tree_builder'
# require 'Tree.tree_visualiser'
# require 'Nn.next_round_value'
# require 'Nn.value_nn'
import sys
import os
sys.path.insert(0, '../TerminalEquity')
sys.path.insert(0, '../Lookahead')
sys.path.insert(0, os.path.abspath('../Nn'))
sys.path.insert(0, os.path.abspath('../Tree'))
sys.path.insert(0, os.path.abspath('Game'))
sys.path.insert(0, os.path.abspath('Settings'))
import arguments
from lookahead import Lookahead
import cfrd_gadget
import tree_builder
import tree_visualiser
from terminal_equity import TerminalEquity
from constants import constants
from card_tool import CardTool
import tools
import game_settings
import value_nn
import next_round_value
import torch
import math

neural_net = None

class LookaheadBuilder(object):
    '''
    --used to load NN only once

    
    --- Constructor
    -- @param lookahead the @{lookahead|Lookahead} to generate data structures for
    '''

    def __init__(self,lookahead):
        self.lookahead = lookahead
        self.lookahead.ccall_action_index = 1
        self.lookahead.fold_action_index = 2

    '''
    --- Builds the neural net query boxes which estimate counterfactual values
    -- at depth-limited states of the lookahead.
    -- @local
    '''

    def _construct_transition_boxes(self):
        if self.lookahead.tree.street == 2:
            return

        '''
        --load neural net if not already loaded
        '''
        neural_net = ValueNn()
        self.lookahead.next_street_boxes = {}
        for d in range(2, self.lookahead.depth):
            self.lookahead.next_street_boxes[d] = NextRoundValue(neural_net)
            # self.lookahead.next_street_boxes[d].start_computation(
            #     self.lookahead.pot_size[d][{2, {}, {}, 1, 1}]:clone():view(-1))

        '''
        --- Computes the number of nodes at each depth of the tree.
        -- 
        -- Used to find the size for the tensors which store lookahead data.
        -- @local
        '''

    def _compute_structure(self):
        assert (self.lookahead.tree.street >= 1 and self.lookahead.tree.street <= 2)

        self.lookahead.regret_epsilon = 1.0 / 1000000000

        ##--which player acts at particular depth
        self.lookahead.acting_player = torch.Tensor(self.lookahead.depth + 1).fill(-1)
        self.lookahead.acting_player[1] = 1 ## in lookahead, 1 does not stand for player IDs, it's just the first player to act
        for d in range(2, self.lookahead.depth+1):
            self.lookahead.acting_player[d] = 3 - self.lookahead.acting_player[d - 1]

        self.lookahead.bets_count[-1] = 1
        self.lookahead.bets_count[0] = 1
        self.lookahead.nonallinbets_count[-1] = 1
        self.lookahead.nonallinbets_count[0] = 1
        self.lookahead.terminal_actions_count[-1] = 0
        self.lookahead.terminal_actions_count[0] = 0
        self.lookahead.actions_count[-1] = 1
        self.lookahead.actions_count[0] = 1

        ##--compute the node counts
        self.lookahead.nonterminal_nodes_count = {}
        self.lookahead.nonterminal_nonallin_nodes_count = {}
        self.lookahead.all_nodes_count = {}
        self.lookahead.terminal_nodes_count = {}
        self.lookahead.allin_nodes_count = {}
        self.lookahead.inner_nodes_count = {}

        self.lookahead.nonterminal_nodes_count[1] = 1
        self.lookahead.nonterminal_nodes_count[2] = self.lookahead.bets_count[1]
        self.lookahead.nonterminal_nonallin_nodes_count[0] = 1
        self.lookahead.nonterminal_nonallin_nodes_count[1] = 1
        self.lookahead.nonterminal_nonallin_nodes_count[2] = self.lookahead.nonterminal_nodes_count[2] - 1
        self.lookahead.all_nodes_count[1] = 1
        self.lookahead.all_nodes_count[2] = self.lookahead.actions_count[1]
        self.lookahead.terminal_nodes_count[1] = 0
        self.lookahead.terminal_nodes_count[2] = 2
        self.lookahead.allin_nodes_count[1] = 0
        self.lookahead.allin_nodes_count[2] = 1
        self.lookahead.inner_nodes_count[1] = 1
        self.lookahead.inner_nodes_count[2] = 1

        for d in range(2, self.lookahead.depth):
            self.lookahead.all_nodes_count[d + 1] = self.lookahead.nonterminal_nonallin_nodes_count[d - 1] * \
                                                    self.lookahead.bets_count[d - 1] * self.lookahead.actions_count[
                                                        d]
            self.lookahead.allin_nodes_count[d + 1] = self.lookahead.nonterminal_nonallin_nodes_count[d - 1] * \
                                                      self.lookahead.bets_count[d - 1] * 1

            self.lookahead.nonterminal_nodes_count[d + 1] = self.lookahead.nonterminal_nonallin_nodes_count[d - 1] * \
                                                            self.lookahead.nonallinbets_count[d - 1] * \
                                                            self.lookahead.bets_count[d]
            self.lookahead.nonterminal_nonallin_nodes_count[d + 1] = \
            self.lookahead.nonterminal_nonallin_nodes_count[d - 1] * self.lookahead.nonallinbets_count[d - 1] * \
            self.lookahead.nonallinbets_count[d]
            self.lookahead.terminal_nodes_count[d + 1] = self.lookahead.nonterminal_nonallin_nodes_count[d - 1] * \
                                                         self.lookahead.bets_count[d - 1] * \
                                                         self.lookahead.terminal_actions_count[d]

    '''
    --- Builds the tensors that store lookahead data during re-solving.
    '''

    def construct_data_structures(self):
        self._compute_structure()
        ##--lookahead main data structures
        ##--all the structures are per-layer tensors, that is, each layer holds the data in n-dimensional tensors
        self.lookahead.pot_size = {}
        self.lookahead.ranges_data = {}
        self.lookahead.average_strategies_data = {}
        self.lookahead.current_strategy_data = {}
        self.lookahead.cfvs_data = {}
        self.lookahead.average_cfvs_data = {}
        self.lookahead.regrets_data = {}
        self.lookahead.current_regrets_data = {}
        self.lookahead.positive_regrets_data = {}
        self.lookahead.placeholder_data = {}
        self.lookahead.regrets_sum = {}
        self.lookahead.empty_action_mask = {}
        ##--used to mask empty actions
        ##--used to hold and swap inner (nonterminal) nodes when doing some transpose operations
        self.lookahead.inner_nodes = {}
        self.lookahead.inner_nodes_p1 = {}
        self.lookahead.swap_data = {}

        ##--create the data structure for the first two layers

        ##--data structures [actions x parent_action x grandparent_id x batch x players x range]
        self.lookahead.ranges_data[1] = arguments.Tensor(1, 1, 1, constants.players_count,
                                                         game_settings.card_count).fill(
            1.0 / game_settings.card_count)
        self.lookahead.ranges_data[2] = arguments.Tensor(self.lookahead.actions_count[1], 1, 1,
                                                         constants.players_count, game_settings.card_count).fill(
            1.0 / game_settings.card_count)
        self.lookahead.pot_size[1] = self.lookahead.ranges_data[1].clone().fill(0)
        self.lookahead.pot_size[2] = self.lookahead.ranges_data[2].clone().fill(0)
        self.lookahead.cfvs_data[1] = self.lookahead.ranges_data[1].clone().fill(0)
        self.lookahead.cfvs_data[2] = self.lookahead.ranges_data[2].clone().fill(0)
        self.lookahead.average_cfvs_data[1] = self.lookahead.ranges_data[1].clone().fill(0)
        self.lookahead.average_cfvs_data[2] = self.lookahead.ranges_data[2].clone().fill(0)
        self.lookahead.placeholder_data[1] = self.lookahead.ranges_data[1].clone().fill(0)
        self.lookahead.placeholder_data[2] = self.lookahead.ranges_data[2].clone().fill(0)

        ##--data structures for one player [actions x parent_action x grandparent_id x 1 x range]
        self.lookahead.average_strategies_data[1] = None
        self.lookahead.average_strategies_data[2] = arguments.Tensor(self.lookahead.actions_count[1], 1, 1,
                                                                     game_settings.card_count).fill(0)
        self.lookahead.current_strategy_data[1] = None
        self.lookahead.current_strategy_data[2] = self.lookahead.average_strategies_data[2].clone().fill(0)
        self.lookahead.regrets_data[1] = None
        self.lookahead.regrets_data[2] = self.lookahead.average_strategies_data[2].clone().fill(0)
        self.lookahead.current_regrets_data[1] = None
        self.lookahead.current_regrets_data[2] = self.lookahead.average_strategies_data[2].clone().fill(0)
        self.lookahead.positive_regrets_data[1] = None
        self.lookahead.positive_regrets_data[2] = self.lookahead.average_strategies_data[2].clone().fill(0)
        self.lookahead.empty_action_mask[1] = None
        self.lookahead.empty_action_mask[2] = self.lookahead.average_strategies_data[2].clone().fill(1)

        ##--data structures for summing over the actions [1 x parent_action x grandparent_id x range]
        self.lookahead.regrets_sum[1] = arguments.Tensor(1, 1, 1, game_settings.card_count).fill(0)
        self.lookahead.regrets_sum[2] = arguments.Tensor(1, self.lookahead.bets_count[1], 1,
                                                         game_settings.card_count).fill(0)

        ##--data structures for inner nodes (not terminal nor allin) [bets_count x parent_nonallinbetscount x gp_id x batch x players x range]
        self.lookahead.inner_nodes[1] = arguments.Tensor(1, 1, 1, constants.players_count,
                                                         game_settings.card_count).fill(0)
        self.lookahead.swap_data[1] = self.lookahead.inner_nodes[1].transpose(2, 3).clone()
        self.lookahead.inner_nodes_p1[1] = arguments.Tensor(1, 1, 1, 1, game_settings.card_count).fill(0)

        if self.lookahead.depth > 2:
            self.lookahead.inner_nodes[2] = arguments.Tensor(self.lookahead.bets_count[1], 1, 1,
                                                             constants.players_count,
                                                             game_settings.card_count).fill(0)
            self.lookahead.swap_data[2] = self.lookahead.inner_nodes[2].transpose(2, 3).clone()
            self.lookahead.inner_nodes_p1[2] = arguments.Tensor(self.lookahead.bets_count[1], 1, 1, 1,
                                                                game_settings.card_count).fill(0)

        ##--create the data structures for the rest of the layers
        for d in range(3, self.lookahead.depth):
            ##--data structures [actions x parent_action x grandparent_id x batch x players x range]
            self.lookahead.ranges_data[d] = arguments.Tensor(self.lookahead.actions_count[d - 1],
                                                             self.lookahead.bets_count[d - 2],
                                                             self.lookahead.nonterminal_nonallin_nodes_count[d - 2],
                                                             constants.players_count,
                                                             game_settings.card_count).fill(0)
            self.lookahead.cfvs_data[d] = self.lookahead.ranges_data[d].clone()
            self.lookahead.placeholder_data[d] = self.lookahead.ranges_data[d].clone()
            self.lookahead.pot_size[d] = self.lookahead.ranges_data[d].clone().fill(arguments.stack)

            ##--data structures [actions x parent_action x grandparent_id x batch x 1 x range]
            self.lookahead.average_strategies_data[d] = arguments.Tensor(self.lookahead.actions_count[d - 1],
                                                                         self.lookahead.bets_count[d - 2],
                                                                         self.lookahead.nonterminal_nonallin_nodes_count[
                                                                             d - 2], game_settings.card_count).fill(0)
            self.lookahead.current_strategy_data[d] = self.lookahead.average_strategies_data[d].clone()
            self.lookahead.regrets_data[d] = self.lookahead.average_strategies_data[d].clone().fill(
                self.lookahead.regret_epsilon)
            self.lookahead.current_regrets_data[d] = self.lookahead.average_strategies_data[d].clone().fill(0)
            self.lookahead.empty_action_mask[d] = self.lookahead.average_strategies_data[d].clone().fill(1)
            self.lookahead.positive_regrets_data[d] = self.lookahead.regrets_data[d].clone()

            ##--data structures [1 x parent_action x grandparent_id x batch x players x range]
            self.lookahead.regrets_sum[d] = arguments.Tensor(1, self.lookahead.bets_count[d - 2],
                                                             self.lookahead.nonterminal_nonallin_nodes_count[d - 2],
                                                             constants.players_count,
                                                             game_settings.card_count).fill(0)

        ##--data structures for the layers except the last one
        if d < self.lookahead.depth:
            self.lookahead.inner_nodes[d] = arguments.Tensor(self.lookahead.bets_count[d - 1],
                                                             self.lookahead.nonallinbets_count[d - 2],
                                                             self.lookahead.nonterminal_nonallin_nodes_count[d - 2],
                                                             constants.players_count,
                                                             game_settings.card_count).fill(0)
            self.lookahead.inner_nodes_p1[d] = arguments.Tensor(self.lookahead.bets_count[d - 1],
                                                                self.lookahead.nonallinbets_count[d - 2],
                                                                self.lookahead.nonterminal_nonallin_nodes_count[
                                                                    d - 2], 1, game_settings.card_count).fill(0)

            self.lookahead.swap_data[d] = self.lookahead.inner_nodes[d].transpose(2, 3).clone()

    '''
    --- Traverses the tree to fill in lookahead data structures that summarize data
    -- contained in the tree.
    -- 
    -- For example, saves pot sizes and numbers of actions at each lookahead state.
    --
    -- @param node the current node of the public tree
    -- @param layer the depth of the current node
    -- @param action_id the index of the action that led to this node
    -- @param parent_id the index of the current node's parent
    -- @param gp_id the index of the current node's grandparent
    -- @local
    '''

    def set_datastructures_from_tree_dfs(self,node, layer, action_id, parent_id, gp_id):

        ##--fill the potsize
        assert (node.pot)
        self.lookahead.pot_size[layer][{action_id, parent_id, gp_id, {}, {}}] = node.pot

        node.lookahead_coordinates = arguments.Tensor({action_id, parent_id, gp_id})

        ###--transition call cannot be allin call
        if node.current_player == constants.players.chance:
            assert (parent_id <= self.lookahead.nonallinbets_count[layer - 2])

        if layer < self.lookahead.depth + 1:
            gp_nonallinbets_count = self.lookahead.nonallinbets_count[layer - 2]
            prev_layer_terminal_actions_count = self.lookahead.terminal_actions_count[layer - 1]
            gp_terminal_actions_count = self.lookahead.terminal_actions_count[layer - 2]
            prev_layer_bets_count = 0

            prev_layer_bets_count = self.lookahead.bets_count[layer - 1]

            ##--compute next coordinates for parent and grandparent
            next_parent_id = action_id - prev_layer_terminal_actions_count
            next_gp_id = (gp_id - 1) * gp_nonallinbets_count + (parent_id)

            if (not node.terminal) and (node.current_player != constants.players.chance):

                ##--parent is not an allin raise
                assert (parent_id <= self.lookahead.nonallinbets_count[layer - 2])

                ##--do we need to mask some actions for that node? (that is, does the node have fewer children than the max number of children for any node on this layer)
                node_with_empty_actions = ( len(node.children) < self.lookahead.actions_count[layer])

                if node_with_empty_actions:
                    ###--we need to mask nonexisting padded bets
                    assert (layer > 1)

                terminal_actions_count = self.lookahead.terminal_actions_count[layer]
                assert (terminal_actions_count == 2)

                existing_bets_count =  len(node.children - terminal_actions_count)

                ###--allin situations
                if existing_bets_count == 0:
                    assert (action_id == self.lookahead.actions_count[layer - 1])

                for child_id in range(1, terminal_actions_count):
                    child_node = node.children[child_id]
                ###--go deeper
                self.set_datastructures_from_tree_dfs(child_node, layer + 1, child_id, next_parent_id, next_gp_id)

                ##--we need to make sure that even though there are fewer actions, the last action/allin is has the same last index as if we had full number of actions
                ##--we manually set the action_id as the last action (allin)
                for b in range(1, existing_bets_count):
                    self.set_datastructures_from_tree_dfs(node.children[len(node.children-b+1)], layer+1, self.lookahead.actions_count[layer]-b+1, next_parent_id, next_gp_id)

                    ##--mask out empty actions
                    self.lookahead.empty_action_mask[layer + 1][{
                                                              {terminal_actions_count + 1,
                                                               -(existing_bets_count + 1)}, next_parent_id,
                                                              next_gp_id, {}}] = 0

            else:
            ###--node has full action count, easy to handle
                for child_id in range(1,node.children):
                    ##--go deeper
                    child_node = node.children[child_id]
                    self.set_datastructures_from_tree_dfs(child_node, layer + 1, child_id, next_parent_id,
                                                          next_gp_id)

    '''
    --- Builds the lookahead's internal data structures using the public tree.
    -- @param tree the public tree used to construct the lookahead
    '''
    def build_from_tree(self,tree):
        self.lookahead.tree = tree
        self.lookahead.depth = tree.depth

        ##--per layer information about tree actions
        ##--per layer actions are the max number of actions for any of the nodes on the layer
        self.lookahead.bets_count = {}
        self.lookahead.nonallinbets_count = {}
        self.lookahead.terminal_actions_count = {}
        self.lookahead.actions_count = {}

        self._compute_tree_structures({tree}, 1)

        ###--construct the initial data structures using the bet counts
        self.construct_data_structures()

        ##--traverse the tree and fill the datastructures (pot sizes, non-existin actions, ...)
        ##--node, layer, action, parent_action, gp_id
        self.set_datastructures_from_tree_dfs(tree, 1, 1, 1, 1)

        ##--set additional info
        assert (self.lookahead.terminal_actions_count[1] == 1 or self.lookahead.terminal_actions_count[1] == 2)

        self.lookahead.first_call_terminal = self.lookahead.tree.children[2].terminal
        self.lookahead.first_call_transition = self.lookahead.tree.children[
                                                   2].current_player == constants.players.chance
        self.lookahead.first_call_check = (not self.lookahead.first_call_terminal) and (
            not self.lookahead.first_call_transition)

        ###--we mask out fold as a possible action when check is for free, due to
        ##--1) fewer actions means faster convergence
        ##--2) we need to make sure prob of free fold is zero because ACPC dealer changes such action to check
        if self.lookahead.tree.bets[1] == self.lookahead.tree.bets[2]:
            self.lookahead.empty_action_mask[2][1].fill(0)

        ###--construct the neural net query boxes
        self._construct_transition_boxes()

    '''
     --- Computes the maximum number of actions at each depth of the tree.
     -- 
     -- Used to find the size for the tensors which store lookahead data. The 
     -- maximum number of actions is used so that every node at that depth can 
     -- fit in the same tensor.
     -- @param current_layer a list of tree nodes at the current depth
     -- @param current_depth the depth of the current tree nodes
     -- @local
     '''

    def _compute_tree_structures(self, current_layer, current_depth):

        layer_actions_count = 0
        layer_terminal_actions_count = 0
        next_layer = []

        for n in range(1, len(current_layer)):
            node = current_layer[n]
            layer_actions_count = math.max(layer_actions_count, len(node.children))

            node_terminal_actions_count = 0
            for c in range(1, len(current_layer[n].children)):
                if node.children[c].terminal or node.children[c].current_player == constants.players.chance:
                    node_terminal_actions_count = node_terminal_actions_count + 1

            layer_terminal_actions_count = math.max(layer_terminal_actions_count, node_terminal_actions_count)

            ###--add children of the node to the next layer for later pass of BFS
            if not node.terminal:
                for c in range(1, len(node.children)):
                    next_layer.append(node.children[c])

        assert ((layer_actions_count == 0) == (len(next_layer) == 0))
        assert ((layer_actions_count == 0) == (current_depth == self.lookahead.depth))

        ##--set action and bet counts
        self.lookahead.bets_count[current_depth] = layer_actions_count - layer_terminal_actions_count

        self.lookahead.nonallinbets_count[
            current_depth] = layer_actions_count - layer_terminal_actions_count - 1  # - -remove allin
        ###--if no alllin...
        if layer_actions_count == 2:
            assert (layer_actions_count == layer_terminal_actions_count)
        self.lookahead.nonallinbets_count[current_depth] = 0

        self.lookahead.terminal_actions_count[current_depth] = layer_terminal_actions_count
        self.lookahead.actions_count[current_depth] = layer_actions_count

        if len(next_layer) > 0:
            assert (layer_actions_count >= 2)
            ##--go deeper
            self._compute_tree_structures(next_layer, current_depth + 1)


