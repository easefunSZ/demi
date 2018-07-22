import lookahead_builder

import sys

sys.path.insert(0, '/Users/lidingcheng/Documents/software/demi/Source/TerminalEquity')
from terminal_equity import TerminalEquity
from cfrd_gadget import CFRDGadget

# local arguments = require 'Settings.arguments'
# local constants = require 'Settings.constants'
# local game_settings = require 'Settings.game_settings'
# local tools = require 'tools'

'''
--- A depth-limited lookahead of the game tree used for re-solving.
-- @classmod lookahead
'''


class Lookahead():
    ##--- Constructor
    def __init__(self):
        self.builder = LookaheadBuilder(self)

    '''
    --- Constructs the lookahead from a game's public tree.
    --
    -- Must be called to initialize the lookahead.
    -- @param tree a public tree
    '''

    def build_lookahead(self, tree):
        self.builder.build_from_tree(tree)
        self.terminal_equity = TerminalEquity()
        self.terminal_equity.set_board(tree.board)

    '''
    --- Re-solves the lookahead using input ranges.
    --
    -- Uses the input range for the opponent instead of a gadget range, so only
    -- appropriate for re-solving the root node of the game tree (where ranges 
    -- are fixed).
    --
    -- @{build_lookahead} must be called first.
    --
    -- @param player_range a range vector for the re-solving player
    -- @param opponent_range a range vector for the opponent
    '''

    def resolve_first_node(self, player_range, opponent_range):
        self.ranges_data[1][{{}, {}, {}, 1, {}}].copy(player_range)
        self.ranges_data[1][{{}, {}, {}, 2, {}}].copy(opponent_range)
        self._compute()

    '''
    --- Re-solves the lookahead using an input range for the player and
    -- the @{cfrd_gadget|CFRDGadget} to generate ranges for the opponent.
    -- 
    -- @{build_lookahead} must be called first.
    -- 
    -- @param player_range a range vector for the re-solving player
    -- @param opponent_cfvs a vector of cfvs achieved by the opponent
    -- before re-solving
    '''

    def resolve(self, player_range, opponent_cfvs):
        assert (player_range)
        assert (opponent_cfvs)

        self.reconstruction_gadget = CFRDGadget(self.tree.board, player_range, opponent_cfvs)

        self.ranges_data[1][{{}, {}, {}, 1, {}}]:copy(player_range)
        self.reconstruction_opponent_cfvs = opponent_cfvs
        self._compute()

    '''
    --- Re-solves the lookahead.
    -- @local
    '''

    def _compute(self):
        ##--1.0 main loop
        for iter=1, arguments.cfr_iters:
            self._set_opponent_starting_range(iter)
            self._compute_current_strategies()
            self._compute_ranges()
            self._compute_update_average_strategies(iter)
            self._compute_terminal_equities()
            self._compute_cfvs()
            self._compute_regrets()
            self._compute_cumulate_average_cfvs(iter)

        ##--2.0 at the end normalize average strategy
        self._compute_normalize_average_strategies()
        ##--2.1 normalize root's CFVs
        self._compute_normalize_average_cfvs()

    '''
    --- Uses regret matching to generate the players' current strategies.
    -- @local
    '''

    def _compute_current_strategies(self):
        for d=2, self.depth:
            self.positive_regrets_data[d].copy(self.regrets_data[d])
            self.positive_regrets_data[d].clamp(self.regret_epsilon, tools.max_number())

            ##1.0 set regret of empty actions to 0
            self.positive_regrets_data[d].cmul(self.empty_action_mask[d])

            ##1.1  regret matching
            ##note that the regrets as well as the CFVs have switched player indexing
            torch.sum(self.regrets_sum[d], self.positive_regrets_data[d], 1)
            player_current_strategy = self.current_strategy_data[d]
            player_regrets = self.positive_regrets_data[d]
            player_regrets_sum = self.regrets_sum[d]

        player_current_strategy.cdiv(player_regrets, player_regrets_sum.expandAs(player_regrets))

    ##- Using the players' current strategies, computes their probabilities of
    ## reaching each state of the lookahead.
    ## @local
    def _compute_ranges(self):
        for d=1, self.depth-1:
            current_level_ranges = self.ranges_data[d]
            next_level_ranges = self.ranges_data[d + 1]

            prev_layer_terminal_actions_count = self.terminal_actions_count[d - 1]
            prev_layer_actions_count = self.actions_count[d - 1]
            prev_layer_bets_count = self.bets_count[d - 1]
            gp_layer_nonallin_bets_count = self.nonallinbets_count[d - 2]
            gp_layer_terminal_actions_count = self.terminal_actions_count[d - 2]

            ##copy the ranges of inner nodes and transpose
            self.inner_nodes[d].copy(current_level_ranges[{{prev_layer_terminal_actions_count + 1, -1},
                                                           {1, gp_layer_nonallin_bets_count}, {}, {}, {}}].transpose(2,
                                                                                                                     3))

            super_view = self.inner_nodes[d]
            super_view = super_view.view(1, prev_layer_bets_count, -1, constants.players_count,
                                         game_settings.card_count)

            super_view = super_view.expandAs(next_level_ranges)
            next_level_strategies = self.current_strategy_data[d + 1]

            next_level_ranges.copy(super_view)

            ##multiply the ranges of the acting player by his strategy
            next_level_ranges[{{}, {}, {}, self.acting_player[d], {}}].cmul(next_level_strategies)

    ##- Updates the players' average strategies with their current strategies.
    ## @param iter the current iteration number of re-solving
    ## @local
    def _compute_update_average_strategies(self, iter):
        if iter > arguments.cfr_skip_iters:
            ##no need to go through layers since we care for the average strategy only in the first node anyway
            ##note that if you wanted to average strategy on lower layers, you would need to weight the current strategy by the current reach probability
            self.average_strategies_data[2].add(self.current_strategy_data[2])

    ##- Using the players' reach probabilities, computes their counterfactual
    ## values at each lookahead state which is a terminal state of the game.
    ## @local
    def _compute_terminal_equities_terminal_equity(self):
        for d=2, self.depth:
            ##call term eq evaluation
            if self.tree.street == 1:
                if d > 2 or self.first_call_terminal:
                    self.terminal_equity.call_value(self.ranges_data[d][2][-1].view(-1, game_settings.card_count),
                                                    self.cfvs_data[d][2][-1].view(-1, game_settings.card_count))
            else:
                assert (self.tree.street == 2)
                ##on river, any call is terminal
                if d > 2 or self.first_call_terminal:
                    self.terminal_equity.call_value(self.ranges_data[d][2].view(-1, game_settings.card_count),
                                                    self.cfvs_data[d][2].view(-1, game_settings.card_count))

        ##folds
        self.terminal_equity.fold_value(self.ranges_data[d][1].view(-1, game_settings.card_count),
                                        self.cfvs_data[d][1].view(-1, game_settings.card_count))

        ##correctly set the folded player by mutliplying by -1
        fold_mutliplier = (self.acting_player[d] * 2 - 3)
        self.cfvs_data[d][{1, {}, {}, 1, {}}].mul(fold_mutliplier)
        self.cfvs_data[d][{1, {}, {}, 2, {}}].mul(-fold_mutliplier)

    ##- Using the players' reach probabilities, calls the neural net to compute the
    ## players' counterfactual values at the depth-limited states of the lookahead.
    ## @local
    def _compute_terminal_equities_next_street_box(self):

        assert (self.tree.street == 1)
        for d=2, self.depth:

            if d > 2 or self.first_call_transition:
                self.next_street_boxes_inputs = self.next_street_boxes_inputs or {}
                self.next_street_boxes_outputs = self.next_street_boxes_outputs or {}
                self.next_street_boxes_inputs[d] = self.next_street_boxes_inputs[d] or self.ranges_data[d][
                    {2, {}, {}, {}, {}}].view(-1, constants.players_count, game_settings.card_count).clone().fill(0)
                self.next_street_boxes_outputs[d] = self.next_street_boxes_outputs[d] or self.next_street_boxes_inputs[
                    d].clone()
                ##now the neural net accepts the input for P1 and P2 respectively, so we need to swap the ranges if necessary
                self.next_street_boxes_outputs[d].copy(self.ranges_data[d][{2, {}, {}, {}, {}}])

            if self.tree.current_player == 1:
                self.next_street_boxes_inputs[d].copy(self.next_street_boxes_outputs[d])
            else:
                self.next_street_boxes_inputs[d][{{}, 1, {}}].copy(self.next_street_boxes_outputs[d][{{}, 2, {}}])
                self.next_street_boxes_inputs[d][{{}, 2, {}}].copy(self.next_street_boxes_outputs[d][{{}, 1, {}}])
                self.next_street_boxes[d].get_value(self.next_street_boxes_inputs[d], self.next_street_boxes_outputs[d])
            ##now the neural net outputs for P1 and P2 respectively, so we need to swap the output values if necessary
        if self.tree.current_player == 1:
            self.next_street_boxes_inputs[d].copy(self.next_street_boxes_outputs[d])
            self.next_street_boxes_outputs[d][{{}, 1, {}}].copy(self.next_street_boxes_inputs[d][{{}, 2, {}}])
            self.next_street_boxes_outputs[d][{{}, 2, {}}].copy(self.next_street_boxes_inputs[d][{{}, 1, {}}])

        self.cfvs_data[d][{2, {}, {}, {}, {}}].copy(self.next_street_boxes_outputs[d])

    ##- Gives the average counterfactual values for the opponent during re-solving
    ## after a chance event (the betting round changes and more cards are dealt).
    ##
    ## Used during continual re-solving to track opponent cfvs. The lookahead must
    ## first be re-solved with @{resolve} or @{resolve_first_node}.
    ##
    ## @param action_index the action taken by the re-solving player at the start
    ## of the lookahead
    ## @param board a tensor of board cards, updated by the chance event
    ## @return a vector of cfvs
    def get_chance_action_cfv(self, action_index, board):
        box_outputs = None
        next_street_box = None
        batch_index = None
        pot_mult = None
        assert (not ((action_index == 2) and (self.first_call_terminal)))
        ##check if we should not use the first layer for transition call
        if action_index == 2 and self.first_call_transition:
            box_outputs = self.next_street_boxes_inputs[2].clone().fill(0)
            assert (box_outputs.size(1) == 1)
            batch_index = 1
            next_street_box = self.next_street_boxes[2]
            pot_mult = self.pot_size[2][2]
        else:
            batch_index = action_index - 1  ## remove fold
        if self.first_call_transition:
            batch_index = batch_index - 1
        box_outputs = self.next_street_boxes_inputs[3].clone().fill(0)
        next_street_box = self.next_street_boxes[3]
        pot_mult = self.pot_size[3][2]

        if box_outputs == None:
            assert (false)
        next_street_box.get_value_on_board(board, box_outputs)
        box_outputs.cmul(pot_mult)
        out = box_outputs[batch_index][3 - self.tree.current_player]
        return out

    ##- Using the players' reach probabilities, computes their counterfactual
    ## values at all terminal states of the lookahead.
    ##
    ## These include terminal states of the game and depth-limited states.
    ## @local
    def _compute_terminal_equities(self):

        if self.tree.street == 1:
            self._compute_terminal_equities_next_street_box()
        self._compute_terminal_equities_terminal_equity()
        ##multiply by pot scale factor
        for d=2, self.depth:
            self.cfvs_data[d].cmul(self.pot_size[d])

    ##- Using the players' reach probabilities and terminal counterfactual
    ## values, computes their cfvs at all states of the lookahead.
    ## @local
    def _compute_cfvs(self):
        for d=self.depth, 2, -1:
            gp_layer_terminal_actions_count = self.terminal_actions_count[d - 2]
            ggp_layer_nonallin_bets_count = self.nonallinbets_count[d - 3]

        self.cfvs_data[d][{{}, {}, {}, {1}, {}}].cmul(self.empty_action_mask[d])
        self.cfvs_data[d][{{}, {}, {}, {2}, {}}].cmul(self.empty_action_mask[d])

        self.placeholder_data[d].copy(self.cfvs_data[d])

        ##player indexing is swapped for cfvs
        self.placeholder_data[d][{{}, {}, {}, self.acting_player[d], {}}].cmul(self.current_strategy_data[d])

        torch.sum(self.regrets_sum[d], self.placeholder_data[d], 1)

        ##use a swap placeholder to change {{1,2,3}, {4,5,6}} into {{1,2}, {3,4}, {5,6}}
        swap = self.swap_data[d - 1]
        swap.copy(self.regrets_sum[d])
        self.cfvs_data[d - 1][
            {{gp_layer_terminal_actions_count + 1, -1}, {1, ggp_layer_nonallin_bets_count}, {}, {}, {}}].copy(
            swap.transpose(2, 3))

    ##- Updates the players' average counterfactual values with their cfvs from the
    ## current iteration.
    ## @param iter the current iteration number of re-solving
    ## @local
    def _compute_cumulate_average_cfvs(self, iter):
        if iter > arguments.cfr_skip_iters:
            self.average_cfvs_data[1].add(self.cfvs_data[1])
            self.average_cfvs_data[2].add(self.cfvs_data[2])

    ##- Normalizes the players' average strategies.
    ##
    ## Used at the end of re-solving so that we can track un-normalized average
    ## strategies, which are simpler to compute.
    ## @local
    def _compute_normalize_average_strategies(self):
        ##using regrets_sum as a placeholder container
        player_avg_strategy = self.average_strategies_data[2]
        player_avg_strategy_sum = self.regrets_sum[2]
        torch.sum(player_avg_strategy_sum, player_avg_strategy, 1)
        player_avg_strategy.cdiv(player_avg_strategy_sum.expandAs(player_avg_strategy))

        ##if the strategy is 'empty' (zero reach), strategy does not matter but we need to make sure
        ##it sums to one -> now we set to always fold
        player_avg_strategy[1][player_avg_strategy[1].ne(player_avg_strategy[1])] = 1
        player_avg_strategy[player_avg_strategy.ne(player_avg_strategy)] = 0

    ##- Normalizes the players' average counterfactual values.
    ##
    ## Used at the end of re-solving so that we can track un-normalized average
    ## cfvs, which are simpler to compute.
    ## @local
    def _compute_normalize_average_cfvs(self):
        self.average_cfvs_data[1].div(arguments.cfr_iters - arguments.cfr_skip_iters)

    ##- Using the players' counterfactual values, updates their total regrets
    ## for every state in the lookahead.
    ## @local
    def _compute_regrets(self):
        for d=self.depth, 2, -1:
            gp_layer_terminal_actions_count = self.terminal_actions_count[d - 2]
            gp_layer_bets_count = self.bets_count[d - 2]
            ggp_layer_nonallin_bets_count = self.nonallinbets_count[d - 3]

            current_regrets = self.current_regrets_data[d]
            current_regrets.copy(self.cfvs_data[d][{{}, {}, {}, self.acting_player[d], {}}])

            next_level_cfvs = self.cfvs_data[d - 1]

            parent_inner_nodes = self.inner_nodes_p1[d - 1]
            parent_inner_nodes.copy(next_level_cfvs[
                                        {{gp_layer_terminal_actions_count + 1, -1}, {1, ggp_layer_nonallin_bets_count},
                                         {}, self.acting_player[d], {}}].transpose(2, 3))
            parent_inner_nodes = parent_inner_nodes.view(1, gp_layer_bets_count, -1, game_settings.card_count)
            parent_inner_nodes = parent_inner_nodes.expandAs(current_regrets)

            current_regrets.csub(parent_inner_nodes)

            self.regrets_data[d].add(self.regrets_data[d], current_regrets)

            ##(CFR+)
            self.regrets_data[d].clamp(0, tools.max_number())

    ##- Gets the results of re-solving the lookahead.
    ##
    ## The lookahead must first be re-solved with @{resolve} or
    ## @{resolve_first_node}.
    ##
    ## @return a table containing the fields.
    ##
    ## * `strategy`. an AxK tensor containing the re-solve player's strategy at the
    ## root of the lookahead, where A is the number of actions and K is the range size
    ##
    ## * `achieved_cfvs`. a vector of the opponent's average counterfactual values at the
    ## root of the lookahead
    ##
    ## * `children_cfvs`. an AxK tensor of opponent average counterfactual values after
    ## each action that the re-solve player can take at the root of the lookahead
    def get_results(self):
        out = {}
        actions_count = self.average_strategies_data[2].size(1)

        ##1.0 average strategy
        ##[actions x range]
        ##lookahead already computes the averate strategy we just convert the dimensions
        out.strategy = self.average_strategies_data[2].view(-1, game_settings.card_count).clone()

        ##2.0 achieved opponent's CFVs at the starting node
        out.achieved_cfvs = self.average_cfvs_data[1].view(constants.players_count, game_settings.card_count)[1].clone()

        ##3.0 CFVs for the acting player only when resolving first node
        if self.reconstruction_opponent_cfvs:
            out.root_cfvs = None
        else:
            out.root_cfvs = self.average_cfvs_data[1].view(constants.players_count, game_settings.card_count)[2].clone()

        ##swap cfvs indexing
        out.root_cfvs_both_players = self.average_cfvs_data[1].view(constants.players_count,
                                                                    game_settings.card_count).clone()
        out.root_cfvs_both_players[2].copy(
            self.average_cfvs_data[1].view(constants.players_count, game_settings.card_count)[1])
        out.root_cfvs_both_players[1].copy(
            self.average_cfvs_data[1].view(constants.players_count, game_settings.card_count)[2])

        ##4.0 children CFVs
        ##[actions x range]
        out.children_cfvs = self.average_cfvs_data[2][{{}, {}, {}, 1, {}}].clone().view(-1, game_settings.card_count)

        ##IMPORTANT divide average CFVs by average strategy in here
        scaler = self.average_strategies_data[2].view(-1, game_settings.card_count).clone()

        range_mul = self.ranges_data[1][{{}, {}, {}, 1, {}}].view(1, game_settings.card_count).clone()
        range_mul = range_mul.expandAs(scaler)

        scaler = scaler.cmul(range_mul)
        scaler = scaler.sum(2).expandAs(range_mul).clone()
        scaler = scaler.mul(arguments.cfr_iters - arguments.cfr_skip_iters)

        out.children_cfvs.cdiv(scaler)

        assert (out.strategy)
        assert (out.achieved_cfvs)
        assert (out.children_cfvs)
        return out

    ##- Generates the opponent's range for the current re-solve iteration using
    ## the @{cfrd_gadget|CFRDGadget}.
    ## @param iteration the current iteration number of re-solving
    ## @local
    def _set_opponent_starting_range(self, iteration):
        if self.reconstruction_opponent_cfvs:
            ##note that CFVs indexing is swapped, thus the CFVs for the reconstruction player are for player '1'
            opponent_range = self.reconstruction_gadget.compute_opponent_range(self.cfvs_data[1][{{}, {}, {}, 1, {}}],
                                                                               iteration)
            self.ranges_data[1][{{}, {}, {}, 2, {}}].copy(opponent_range)
