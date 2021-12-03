"""
Basics components of counterfactual regret minimization algorithms: definition and implementation

    Strategy
    Reach Probabilities
    Utilities
    Counterfactual Utilities
    Regret

    Note that everything we need can be computed from reach probability and utilities.
    So for given strategy we compute reach probabilities and utilities.
    Strategies are defined as:
    Strat as list of  willingness probs of coming to public node k. Note that each public node k  has only one parent
    which is accessible by G.public_state[k].parent
"""
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

    # reach_probs[player, hand, public_node] induced by given strategy are going to be stored in this matrix
    reach_probs = np.ones((2, number_of_hands, number_of_public_nodes))

    # word_state_values[op_hand, ip_hand, public_node] induced by given strategy are going to be stored in this matrix
    world_state_values = np.zeros((number_of_hands, number_of_hands, number_of_public_nodes))

    # not really necessary , just in case
    S = copy.deepcopy(strategy)

    # make looping over all hands dealing easier, or does it?!
    deck_of_paired_hands = BettingGame.cards_dealing(game)

    # main function that traverse game world tree and compute reach_probs and state values
    def recursive_evaluation(current_node, op_hand, ip_hand, op_reach, ip_reach):

        # just to prevent if over player
        hands = (op_hand, ip_hand)
        reach = (op_reach, ip_reach)

        current_state = game.public_state[current_node]

        if current_state.is_terminal:
            current_EV = game.terminal_value(current_node, op_hand, ip_hand)
            world_state_values[op_hand, ip_hand, current_node] = current_EV
            return current_EV

        player = current_state.to_move
        current_EV = 0
        for child in current_state.children:

            child_reach = [reach[0], reach[1]]
            child_reach[player] *= strategy[hands[player], child]
            reach_probs[player, hands[player], child] = child_reach[player]
            reach_probs[1-player, hands[1-player], child] = child_reach[1-player]
            current_EV += strategy[hands[player], child]*recursive_evaluation(
                child, op_hand, ip_hand, child_reach[0], child_reach[1])
        world_state_values[op_hand, ip_hand, current_node] = current_EV
        return current_EV

    for pair_of_hands in deck_of_paired_hands:
        op_hand, ip_hand = pair_of_hands
        recursive_evaluation(0, op_hand, ip_hand, 1, 1)
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
G = BettingGame(4, deck={i: 1 for i in range(8)})

KV = world_node_strategic_evaluation(K.uniform_strategy(), K)
GV = world_node_strategic_evaluation(G.uniform_strategy(), G)
G_reach = GV[1]


