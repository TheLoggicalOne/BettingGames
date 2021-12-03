import copy
import numpy as np
from betting_games import BettingGame


def plain_cfr(game, number_of_iteration, initial_strategy):
    pass


# This function is going to compute reach probabilities, utilities, and anything else that it can
# and return them as np array
# strategy is a of size number_of_hands*number_of_public_state
def world_node_strategic_evaluation(strategy, game):
    number_of_hands = np.shape(strategy)[0]
    number_of_public_nodes = game.public_tree.number_of_nodes
    number_of_info_nodes = number_of_hands * number_of_public_nodes

    # reach_probs[player, op_hand, ip_hand, public_node] induced by given strategy are going to be stored in this matrix
    reach_probs = np.ones((3, number_of_hands,number_of_hands, number_of_public_nodes))

    # word_state_values[op_hand, ip_hand, public_node] induced by given strategy are going to be stored in this matrix
    world_state_values = np.zeros((number_of_hands, number_of_hands, number_of_public_nodes))

    # not really necessary , just in case
    S = copy.deepcopy(strategy)

    # make looping over all hands dealing easier, or does it?!
    deck_of_paired_hands = BettingGame.cards_dealing(game)

    # world_state_value[ , , t.parent] += world_state_value[ , , t]**strategy[ , t]
    for terminal_node in game.terminal_node:
        for pair_of_hands in deck_of_paired_hands:
            op_hand, ip_hand = pair_of_hands
            world_state_values[op_hand, ip_hand, terminal_node] = game.terminal_value(terminal_node, op_hand, ip_hand)
        current_node = terminal_node
        while current_node > 0:
            next_node = game.public_state[current_node].parent
            world_state_values[:, :, next_node] \
                += world_state_values[:, :, current_node]*strategy[:, current_node]
            current_node = next_node
    return world_state_values, reach_probs


def strategy_matrix(game):
    S = np.zeros((len(game.deck.keys()), game.public_tree.number_of_nodes))
    for i in range(len(game.deck.keys())):
        for _decision_node in game.public_tree.decision_nodes:
            pass

# -------------------------------- INITIALIZING EXAMPLE OF STRATEGIC BETTING GAMES ----------------------------------- #


KUHN_BETTING_GAME = BettingGame(bet_size=0.5, max_number_of_bets=2,

                                deck={0: 1, 1: 1, 2: 1}, deal_from_deck_with_substitution=False)
K = KUHN_BETTING_GAME
KU = K.uniform_strategy()
KV = world_node_strategic_evaluation(K.uniform_strategy(), K)
K_p_states = K.public_state


G = BettingGame(4, deck={i: 1 for i in range(8)})
GV = world_node_strategic_evaluation(G.uniform_strategy(), G)
G_reach = GV[1]
