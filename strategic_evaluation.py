import copy
import numpy as np
from betting_games import BettingGame


# strategy is a np array of size number_of_hands*number_of_public_state

# This function is going to compute reach probabilities, utilities, and anything else that it can
# and return them as np array

def world_node_strategic_evaluation(strategy, game):
    number_of_hands = np.shape(strategy)[0]
    number_of_public_nodes = game.public_tree.number_of_nodes

    # word_state_values[op_hand, ip_hand, public_node] induced by given strategy are going to be stored in this matrix
    # values at terminal states are computed already
    world_state_values = game.terminal_values_all_node

    # world_state_value[ , , t.parent] += world_state_value[ , , t]**strategy[ , t]
    nonzero_nodes = game.node[1:]
    reverse_node_list = nonzero_nodes[::-1]

    for current_node in reverse_node_list:
        current_player = game.public_state[current_node].to_move
        parent_node = game.public_state[current_node].parent
        parent_node_player = 1 - current_player

        # if parent_node player is op we multiply rows of value matrix
        if parent_node_player == 0:
            world_state_values[:, :, parent_node] \
                += world_state_values[:, :, current_node] * (strategy[:, current_node].reshape(number_of_hands, 1))

        # else if parent_node player is ip we multiply cols of value matrix
        elif parent_node_player == 1:
            world_state_values[:, :, parent_node] \
                += world_state_values[:, :, current_node] * strategy[:, current_node]

    return world_state_values


# -------------------------------- INITIALIZING EXAMPLE OF STRATEGIC BETTING GAMES ----------------------------------- #


KUHN_BETTING_GAME = BettingGame(bet_size=0.5, max_number_of_bets=2,

                                deck={0: 1, 1: 1, 2: 1}, deal_from_deck_with_substitution=False)
K = KUHN_BETTING_GAME
KU = K.uniform_strategy()
KV = world_node_strategic_evaluation(K.uniform_strategy(), K)
K_p_states = K.public_state

G = BettingGame(4, deck={i: 1 for i in range(8)})
GV = world_node_strategic_evaluation(G.uniform_strategy(), G)
G_p_state = G.public_state
