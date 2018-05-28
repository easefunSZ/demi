'''''
--- Gives allowed bets during a game.
-- Bets are restricted to be from a list of predefined fractions of the pot.
-- @classmod bet_sizing
'''''

import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0,'../Settings')

from arguments import params

class BetSizing:

    '''''
    --- Constructor
    -- @param pot_fractions a list of fractions of the pot which are allowed
    -- as bets, sorted in ascending order
    '''''
    def __init__(self,pot_fractions):
        self.pot_fractions = pot_fractions

    '''
    --- Gives the bets which are legal at a game state.
-- @param node a representation of the current game state, with fields:
-- 
-- * `bets`: the number of chips currently committed by each player
-- 
-- * `current_player`: the currently acting player
-- @return an Nx2 tensor where N is the number of new possible game states,
-- containing N sets of new commitment levels for each player
    '''
    def get_possible_bets(self,node):
        current_player = node.current_player
        assert(current_player == 1 or current_player == 2, 'Wrong player for bet size computation')
        opponent = 3 - node.current_player
        opponent_bet = node.bets[opponent]
        assert(node.bets[current_player] <= opponent_bet)
        ##--compute min possible raise size
        max_raise_size = params['stack'] - opponent_bet
        min_raise_size = opponent_bet - node.bets[current_player]
        min_raise_size = np.max(min_raise_size, params['ante'])
        min_raise_size = np.min(max_raise_size, min_raise_size)
        ###not so sure how we should obtain the tensor parameters.
        ###it looks that we should directly use tensorflow to get float tensor
        if min_raise_size == 0:
            return params['Tensor']
        elif min_raise_size == max_raise_size:
            out = params['Tensor'] = np.zeros([1,1],dtype=float)
            out[1,current_player] = opponent_bet+min_raise_size
            return out
        else:
            ##--iterate through all bets and check if they are possible
            max_possible_bets_count = len(self.pot_fractions) + 1
            ##--we can always go allin
            out = params['Tensor'].fill(opponent_bet)
            ###--take pot size after opponent bet is called
            pot = opponent_bet * 2
            used_bets_count = 0
            ###--try all pot fractions bet and see if we can use them
            for i in range(1,len(self.pot_fractions)):
                raise_size = pot * self.pot_fractions[i]
                if raise_size >= min_raise_size and raise_size < max_raise_size:
                    used_bets_count = used_bets_count + 1
                    out[used_bets_count, current_player] = opponent_bet + raise_size
            ##--adding allin
            used_bets_count    = used_bets_count + 1
            assert(used_bets_count <= max_possible_bets_count)
            out[used_bets_count, current_player] = opponent_bet + max_raise_size
            return out[0:used_bets_count,]
