# TO BE COMPLETED
# This is the old version of strategy which was suffering from players different hand not letting manipulating strategy
# as column for each node
""" Can Each Strat just be a list of probs?
 Strat as list of  willingness probs of coming to public node k. Note that each public node k  has only one parent
 which is accessible by G.public_state[k].parent
 Strat List could be initialized using tree or using or public_node_tree.PublicTree.children_of_nodes or...?
 Anyway after initialization can be updated using tree.
 Is there a way of updating the start list by matrices and in numpy?
 PublicTree.Tree.ActionIDs
 Best Rep Possible?:
 let d=deck size. Let S be a k*k*x np.array where A=S[i,j,:] shows the state where op_hand=i and ip_hand=j
 Now let A be 1-d array of length that A[k] is the willingness probs of coming to public node k.
 On the other hand if we are at node r with hand i and we want to know chance of playing given action a according to
 our strat, it is S[i,j, G.PublicNodeAsTuple[r].children[a]] """
# ------------------------------------ ------------------------------------------- ----------------------------------- #
import numpy as np
from betting_games import BettingGame


class Strategy:
    """ Provide tools to analysis and study strategies of given betting game.
    Strategy: 2*number_of_hands*number_of_nodes np array, call it S. Then  S[position,hand,i] is chance of moving to
     node i from its parent holding given hand by the player who is to act at the parent of i:
         players hand:[op_hand, ip_hand]
         i column of strategy matrix which relates to reach_player[i] = 1-to_move[i] Then
         S[reach_player[i], hand[reach_player[i]], i]

         So game dynamic is as follows:
         reaching node i from its parent: S[reach_player[i], : , i]
         reaching node i by following path pj for j=1,...,d:
         S[reach_player[i], : ,
    strategy_base: this is the main input
    """
    def __init__(self, game, strategy_base=None):
        self.game = game

        self.number_of_hands = self.game.number_of_hands
        self.number_of_nodes = self.game.public_tree.number_of_nodes
        if strategy_base is None:
            strategy_base = np.ones((2, self.number_of_hands, self.number_of_nodes))
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
        self.reach_player = [1 - (self.depth_of_node[i] % 2) for i in self.game.node]
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

# ----------------------------------- STRATEGY AND VALUES INDUCED BY STRATEGY ---------------------------------------- #

    def uniform_strategy(self):
        S = np.ones((2, self.number_of_hands, self.number_of_nodes))
        for _decision_node in self.decision_node:
            childs = self.game.public_state[_decision_node].children
            for child in childs:
                S[self.reach_player[_decision_node], :, child:child+1] = np.full((
                    self.number_of_hands, 1), 1/len(childs))
        return S

    def update_strategy_base_to(self, action_prob_function):
        S = np.ones((2, self.number_of_hands, self.number_of_nodes))
        for i in range(self.number_of_hands):
            for _decision_node in self.decision_node:
                childs = self.game.public_state[_decision_node].children
                for child in childs:
                    S[self.reach_player[_decision_node], i, child:child + 1] = action_prob_function(i, child)
        return S


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

    SK = Strategy(K)
    GK = Strategy(G)
    SK.strategy_base = SK.uniform_strategy()

