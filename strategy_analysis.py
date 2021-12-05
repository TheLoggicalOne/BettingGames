import numpy as np
from betting_games import BettingGame


class StrategyAnalyzer:
    """ Provide tools to analysis and study strategies of given betting game.
    Strategy: 2*number_of_hands*number_of_nodes np array, call it S. Then  S[position,hand,i] is chance of moving to
     node i from its parent holding given hand by the player who is to act at the parent of i:
         players hand:[op_hand, ip_hand]
         i column of strategy matrix which relates to player = 1-to_move[i]
         S[player,
    strategy_base: this is the main input


    """
    def __init__(self, game, strategy_base=None):
        self.game = game

        self.number_of_hands = self.game.number_of_hands
        self.number_of_nodes = self.game.public_tree.number_of_nodes
        if strategy_base is None:
            strategy_base = np.ones((self.number_of_hands, self.number_of_nodes))
        self.strategy_base = strategy_base

        self.decision_node = self.game.decision_node
        self.is_decision_node = [not self.game.public_state[i].is_terminal for i in self.game.node]
        self.check_decision_branch = np.array([node for node in self.decision_node
                                               if self.game.public_state[node].first_played_action == 'Check'])
        self.bet_decision_branch = np.array([node for node in self.decision_node
                                             if self.game.public_state[node].first_played_action == 'Bet'])

        self.op_turn_nodes = np.array([node for node in self.game.node if self.game.public_state[node].to_move == 0])
        self.ip_turn_nodes = np.array([node for node in self.game.node if self.game.public_state[node].to_move == 1])
        self.depth_of_node = self.game.depth_of_node
        self.reach_player = [1 - (self.depth_of_node[i] % 2) + int(i == 0) for i in self.game.node]
        self.parent = self.game.parent

    # even cols are op cols
    def check_branch_strategy(self):
        """ return strategy columns of decision nodes in check branch of game.
        First column corresponds to op first column in strategy_base: 1
        even indexed columns are for op """
        return np.hstack((self.strategy_base[:, 1:2], self.strategy_base[:, 4:self.check_decision_branch[-1] + 1:6]
                          ))

    # odd cols are op cols
    def bet_branch_strategy(self):
        """ return strategy columns of decision nodes in bet branch of game.
        First column corresponds to ip first column: 2
        odd indexed columns are op  """
        return np.hstack((self.strategy_base[:, 2:3], self.strategy_base[:, 7:self.bet_decision_branch[-1] + 1:6]))

    def reach_probs_of_check_decision_branch(self):
        """ return reach probability columns of decision nodes in check branch of game. First column corresponds to op
        first decision node  """
        return np.cumprod(self.check_branch_strategy(), axis=1)

    def reach_probs_of_bet_decision_branch(self):
        return np.cumprod(self.bet_branch_strategy(), axis=1)

    def op_reach_probs_of_check_decision_branch(self):
        return np.cumprod(self.check_branch_strategy()[:, 0::2], axis=1)

    def ip_reach_probs_of_check_decision_branch(self):
        return np.cumprod(self.check_branch_strategy()[:, 1::2], axis=1)

    def op_reach_probs_of_bet_decision_branch(self):
        return np.cumprod(self.bet_branch_strategy()[:, 0::2], axis=1)

    def ip_reach_probs_of_bet_decision_branch(self):
        return np.cumprod(self.bet_branch_strategy()[:, 1::2], axis=1)

    def reach_probs_calc(self, node):
        if self.game.public_state[node].first_played_action == 'Check':
            if self.is_decision_node[node]:
                return self.reach_probs_of_check_decision_branch()[:,
                       self.depth_of_node[node] - 1:self.depth_of_node[node]]
            else:
                parent = self.parent[node]
                return self.strategy_base[:, node:node + 1] * self.reach_probs_of_check_decision_branch()[:,
                                                              self.depth_of_node[parent] - 1:self.depth_of_node[parent]]

        else:
            if self.is_decision_node[node]:
                return self.reach_probs_of_bet_decision_branch()[:,
                       self.depth_of_node[node] - 1:self.depth_of_node[node]]
            else:
                parent = self.parent[node]
                return self.strategy_base[:, node:node + 1] \
                       * self.reach_probs_of_bet_decision_branch()[:,
                         self.depth_of_node[parent] - 1:self.depth_of_node[parent]]


# ------------------------------------ INITIALIZING BETTING GAMES WITH USUAL SIZES ----------------------------------- #

# Start a Game
if __name__ == '__main__':
    J = 1;
    Q = 2;
    K = 3
    KUHN_BETTING_GAME = BettingGame(bet_size=0.5, max_number_of_bets=2,
                                    deck={J: 1, Q: 1, K: 1}, deal_from_deck_with_substitution=False)

    K = KUHN_BETTING_GAME
    max_n = 12
    G = BettingGame(max_n)

    SK = StrategyAnalyzer(K, strategy_base=K.uniform_strategy())
    GK = StrategyAnalyzer(G, strategy_base=G.uniform_strategy())
a = [0 for i in range(3)]
for i in range(3):
    a[i] = np.array([i + 2, 2 * (i + 2), 3 * (i + 2)]).reshape((3, 1))
