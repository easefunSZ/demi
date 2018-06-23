# --- Runs Counterfactual Regret Minimization (CFR) to approximately
# -- solve a game represented by a complete game tree.
# --
# -- As this class does full solving from the root of the game with no
# -- limited lookahead, it is not used in continual re-solving. It is provided
# -- simply for convenience.
# -- @classmod tree_cfr

import sys

sys.path.insert(0, '../Settings')
sys.path.insert(0, '../Game')
sys.path.insert(0, '../TerminalEquity')

import arguments
import constants
import game_settings
import card_tools
import terminal_equity


class TreeCFR():


    def __init(self):
        ##--for ease of implementation, we use small epsilon rather than zero when working with regrets
        self.regret_epsilon = 1 / 1000000000
        self._cached_terminal_equities = {}


    '''
    --- Gets an evaluator for player equities at a terminal node.
    --
    -- Caches the result to minimize creation of @{terminal_equity|TerminalEquity}
    -- objects. 
    -- @param node the terminal node to evaluate
    -- @return a @{terminal_equity|TerminalEquity} evaluator for the node
    -- @local
    '''
    def _get_terminal_equity(self, node):
        cached = self._cached_terminal_equities[node.board]
        if cached == None:
            cached = TerminalEquity()
            cached.set_board(node.board)
            self._cached_terminal_equities[node.board] = cached
        return cached


    '''
    --- Recursively walks the tree, applying the CFR algorithm.
    -- @param node the current node in the tree
    -- @param iter the current iteration number
    -- @local
    '''
    def cfrs_iter_dfs(self, node, iter):
        assert (
                node.current_player == constants.players.P1 or node.current_player == constants.players.P2 or node.current_player == constants.players.chance)
        opponent_index = 3 - node.current_player
        ##--dimensions in tensor
        action_dimension = 1
        card_dimension = 2
        ##--compute values using terminal_equity in terminal nodes
        if (node.terminal):
            terminal_equity = self._get_terminal_equity(node)
            values = node.ranges_absolute.clone().fill(0)

            if (node.type == constants.node_types.terminal_fold):
                terminal_equity.tree_node_fold_value(node.ranges_absolute, values, opponent_index)
            else:
                terminal_equity.tree_node_call_value(node.ranges_absolute, values)

            ##--multiply by the pot
            values = values * node.pot
            node.cf_values = values.viewAs(node.ranges_absolute)
        else:

            actions_count = len(node.children)
            current_strategy = None

            if node.current_player == constants.players.chance:
                current_strategy = node.strategy
            else:
                ##--we have to compute current strategy at the beginning of each iteraton

                ##--initialize regrets in the first iteration
                node.regrets = node.regrets or arguments.Tensor(actions_count, game_settings.card_count).fill(
                    self.regret_epsilon)  ##--[[actions_count x card_count]]
                node.possitive_regrets = node.possitive_regrets or arguments.Tensor(actions_count,
                                                                                    game_settings.card_count).fill(
                    self.regret_epsilon)

                ##--compute positive regrets so that we can compute the current strategy fromm them
                node.possitive_regrets.copy(node.regrets)
                node.possitive_regrets[torch.le(node.possitive_regrets, self.regret_epsilon)] = self.regret_epsilon

                ##--compute the current strategy
                regrets_sum = node.possitive_regrets.sum(action_dimension)
                current_strategy = node.possitive_regrets.clone()
                current_strategy.cdiv(regrets_sum.expandAs(current_strategy))

            ##--current cfv [[actions, players, ranges]]
            cf_values_allactions = arguments.Tensor(actions_count, constants.players_count, game_settings.card_count).fill(
                0)

            children_ranges_absolute = {}

            if node.current_player == constants.players.chance:
                ranges_mul_matrix = node.ranges_absolute[1].repeatTensor(actions_count, 1)
                children_ranges_absolute[1] = torch.cmul(current_strategy, ranges_mul_matrix)

                ranges_mul_matrix = node.ranges_absolute[2].repeatTensor(actions_count, 1)
                children_ranges_absolute[2] = torch.cmul(current_strategy, ranges_mul_matrix)
            else:
                ranges_mul_matrix = node.ranges_absolute[node.current_player].repeatTensor(actions_count, 1)
                children_ranges_absolute[node.current_player] = torch.cmul(current_strategy, ranges_mul_matrix)

                children_ranges_absolute[opponent_index] = node.ranges_absolute[opponent_index].repeatTensor(actions_count,
                                                                                                             1).clone()

            for i in range(len(node.children)):
                child_node = node.children[i]
                ##--set new absolute ranges (after the action) for the child
                child_node.ranges_absolute = node.ranges_absolute.clone()

                child_node.ranges_absolute[1].copy(children_ranges_absolute[1][{i}])
                child_node.ranges_absolute[2].copy(children_ranges_absolute[2][{i}])
                self.cfrs_iter_dfs(child_node, iter, card_count)
                cf_values_allactions[i] = child_node.cf_values

            node.cf_values = arguments.Tensor(constants.players_count, game_settings.card_count).fill(0)

            if node.current_player != constants.players.chance:
                strategy_mul_matrix = current_strategy.viewAs(arguments.Tensor(actions_count, game_settings.card_count))

                node.cf_values[node.current_player] = torch.cmul(strategy_mul_matrix,
                                                                 cf_values_allactions[{{}, node.current_player, {}}]).sum(1)
                node.cf_values[opponent_index] = (cf_values_allactions[{{}, opponent_index, {}}]).sum(1)
            else:
                node.cf_values[1] = (cf_values_allactions[{{}, 1, {}}]).sum(1)
                node.cf_values[2] = (cf_values_allactions[{{}, 2, {}}]).sum(1)

            if node.current_player != constants.players.chance:
                ##--computing regrets
                current_regrets = cf_values_allactions[{{}, {node.current_player}, {}}]:reshape(actions_count,
                                                                                                game_settings.card_count).clone()
                current_regrets.csub(node.cf_values[node.current_player]:view(1, game_settings.card_count).expandAs(
                    current_regrets))

                self.update_regrets(node, current_regrets)

                ##--accumulating average strategy
                self.update_average_strategy(node, current_strategy, iter)


    '''
    --- Update a node's total regrets with the current iteration regrets.
    -- @param node the node to update
    -- @param current_regrets the regrets from the current iteration of CFR
    -- @local
    '''


    def update_regrets(self, node, current_regrets):
        # --node.regrets.add(current_regrets)
        # --local negative_regrets = node.regrets[node.regrets:lt(0)]
        # --node.regrets[node.regrets:lt(0)] = negative_regrets
        node.regrets.add(current_regrets)
        node.regrets[torch.le(node.regrets, self.regret_epsilon)] = self.regret_epsilon


    '''
    --- Update a node's average strategy with the current iteration strategy.
    -- @param node the node to update
    -- @param current_strategy the CFR strategy for the current iteration
    -- @param iter the iteration number of the current CFR iteration
    
    '''


    def update_average_strategy(self, node, current_strategy, iter):
        if iter > arguments.cfr_skip_iters:
            node.strategy = node.strategy or arguments.Tensor(actions_count, game_settings.card_count).fill(0)
            node.iter_weight_sum = node.iter_weight_sum or arguments.Tensor(game_settings.card_count).fill(0)
            iter_weight_contribution = node.ranges_absolute[node.current_player].clone()
            iter_weight_contribution[torch.le(iter_weight_contribution, 0)] = self.regret_epsilon
            node.iter_weight_sum.add(iter_weight_contribution)
            iter_weight = torch.cdiv(iter_weight_contribution, node.iter_weight_sum)

            expanded_weight = iter_weight.view(1, game_settings.card_count).expandAs(node.strategy)
            old_strategy_scale = expanded_weight * (-1) + 1  ##--same as 1 - expanded weight
            node.strategy.cmul(old_strategy_scale)
            strategy_addition = current_strategy.cmul(expanded_weight)
            node.strategy.add(strategy_addition)


    '''
    --- Run CFR to solve the given game tree.
    -- @param root the root node of the tree to solve.
    -- @param[opt] starting_ranges probability vectors over player private hands
    -- at the root node (default uniform)
    -- @param[opt] iter_count the number of iterations to run CFR for
    -- (default @{arguments.cfr_iters})
    
    '''


    def run_cfr(self, root, starting_ranges, iter_count):
        assert (starting_ranges)
        iter_count = iter_count or arguments.cfr_iters

        root.ranges_absolute = starting_ranges

        for iter in range(iter_count):
            self.cfrs_iter_dfs(root, iter)
