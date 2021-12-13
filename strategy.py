# ---------------------------------------------- STATE OF THIS CODE -------------------------------------------------- #
# STATUS OF CURRENTS CODES: works perfectly, possible speed improvement
#
# FUTURE PLAN: 1) possibly combining valuation and reach prob calculation( they seem to have common computation)
#              2)
# DOCUMENTATION: 25%
#
# LAST UPDATE OF THI BOX: dec 13 00:15 - after long commit of adding cf reach and value methods
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
from betting_games import BettingGame


class Strategy:
    """ Provide tools to analysis and study strategies of given betting game.
    Our Representation of Strategy: 2*number_of_hands*number_of_nodes np array, call it S. Then  S[position,hand,i] is
    chance of moving to  node i from its parent holding given hand by the player with given position.actually position
    is equivalent to player, and S[0,:,:] is strategy of op and S[1,:,:] is strategy of ip.
     The player who is to act at the parent of i is called reach_player of i, since we reach current node by his move.
     note that if players hand:[op_hand, ip_hand]. Then chance of moving to node i, from its parent is:
         For reach_player, which is player who is act at parent of i:  S[reach_player[i], hand[reach_player[i]], i]
         For other player, which is player who is act at i:            S[turn[i], hand[turn[i]], i] = 1
     Strategy is stored in self.strategy_base

    Attributes: all the structural properties of strategy which depend only on game, and not how players play are stored
        at attributes and all attributes are constant for given game, exept self.strategy_base that can and will change
    strategy_base: this is the main input

    Methods:
        Main methods: they all depend on strategy_base, which is supposed to be current strategy of player and they
        change when self.strategy_base changes. These methods provide reach probabilities and player reach probabilities
         of different public state and information state and world states
    """

    def __init__(self, game, strategy_base=None):
        self.game = game

        self.number_of_hands = self.game.number_of_hands
        self.number_of_nodes = self.game.public_tree.number_of_nodes

        # This present current given strategy and it is the only attributes that changes
        if strategy_base is None:
            strategy_base = np.ones((2, self.number_of_hands, self.number_of_nodes))
        self.strategy_base = strategy_base
        self.initial_strategy = self.strategy_base.copy()
        self.strategy_sum = np.zeros((2, self.number_of_hands, self.number_of_nodes))
        self.iteration = 0
        self.cumulative_regret = np.zeros((2, self.number_of_hands, self.number_of_nodes))
        self.node = self.game.node
        self.decision_node = self.game.decision_node
        self.decision_node_children = [self.game.public_state[i].children for i in self.decision_node]
        self.is_decision_node = [not self.game.public_state[i].is_terminal for i in self.game.node]
        self.check_decision_branch = np.array([node for node in self.decision_node
                                               if self.game.public_state[node].first_played_action == 'Check'])
        self.bet_decision_branch = np.array([node for node in self.decision_node
                                             if self.game.public_state[node].first_played_action == 'Bet'])
        self.start_with_check = [self.game.public_state[node].first_played_action == 'Check' for node in self.game.node]
        self.op_turn_nodes = np.array([node for node in self.game.node if self.game.public_state[node].to_move == 0])
        self.ip_turn_nodes = np.array([node for node in self.game.node if self.game.public_state[node].to_move == 1])
        self.depth_of_node = self.game.depth_of_node
        self.turn = [self.depth_of_node[i] % 2 for i in self.game.node]
        self.reach_player = [1 - (self.depth_of_node[i] % 2) for i in self.game.node]
        self.parent = self.game.parent

        self.terminal_values_all_nodes = self.game.terminal_values_all_node.copy()

        self.chance_reach_prob = self.game.deck_matrix()
# ---------------------------- MAIN METHODS: REACH PROBABILITIES OF GIVEN STRATEGY  ---------------------------------- #

    # even cols are op cols
    def check_branch_strategy(self, position):
        """ return strategy columns of decision nodes in check branch of game.
        First column corresponds to op first column in strategy_base: 1
        even indexed columns are for op """
        return np.hstack((self.strategy_base[position, :, 1:2],
                          self.strategy_base[position, :, 4:self.check_decision_branch[-1] + 1:6]))

    # odd cols are op cols
    def bet_branch_strategy(self, position):
        """ return strategy columns of decision nodes in bet branch of game.
        First column corresponds to ip first column: 2
        odd indexed columns are op  """
        return np.hstack((self.strategy_base[position, :, 2:3],
                          self.strategy_base[position, :, 7:self.bet_decision_branch[-1] + 1:6]))

    def player_reach_probs_of_check_decision_branch_indo_nodes(self, position):
        """ return reach probability columns of decision nodes in check branch of game. First column corresponds to op
        first decision node  """
        return np.cumprod(self.check_branch_strategy(position), axis=1)

    def player_reach_probs_of_bet_decision_branch_info_nodes(self, position):
        return np.cumprod(self.bet_branch_strategy(position), axis=1)

    def player_reach_probs_of_info_node(self, node, position):
        """ returns (self.number_of_hands)*1 numpy array where row i corresponds to info node(hand=i, node)"""
        if node == 0:
            PRN = np.ones((self.number_of_hands, 1))
        elif self.start_with_check[node]:
            if self.is_decision_node[node]:
                return self.player_reach_probs_of_check_decision_branch_indo_nodes(position)[:,
                       self.depth_of_node[node] - 1:self.depth_of_node[node]]
            else:
                parent = self.parent[node]
                PRN = self.strategy_base[position, :,
                       node:node + 1] * self.player_reach_probs_of_check_decision_branch_indo_nodes(
                    position)[:, self.depth_of_node[parent] - 1:self.depth_of_node[parent]]

        else:
            if self.is_decision_node[node]:
                PRN = self.player_reach_probs_of_bet_decision_branch_info_nodes(position)[:,
                       self.depth_of_node[node] - 1:self.depth_of_node[node]]
            else:
                parent = self.parent[node]
                PRN = self.strategy_base[position, :, node:node + 1] \
                       * self.player_reach_probs_of_bet_decision_branch_info_nodes(position)[:,
                         self.depth_of_node[parent] - 1:self.depth_of_node[parent]]
        return PRN

    def players_reach_probs_of_info_nodes_table(self):
        """ returns 2*(self.number_of_hands)*(self.number_of_nodes) numpy array """
        PR = np.ones((2, self.number_of_hands, self.number_of_nodes))
        for i in range(2):
            for node in self.node[1:]:
                PR[i, :, node:node + 1] = self.player_reach_probs_of_info_node(node, i)
        return PR

    def cf_reach_probs_of_world_nodes_table(self, position):
        """ returns (self.number_of_hands)*(self.number_of_hands)*(self.number_of_nodes) numpy array """

        if position == 0:
            reach_of_info_nodes_table = self.players_reach_probs_of_info_nodes_table()[1:2, :, :]
            OR = np.broadcast_to(reach_of_info_nodes_table,
                                 (self.number_of_hands, self.number_of_hands, self.number_of_nodes))
        else:
            reach_of_info_nodes_table = self.players_reach_probs_of_info_nodes_table()[0:1, :, :].reshape(3, 1, 9)
            OR = np.broadcast_to(reach_of_info_nodes_table,
                                 (self.number_of_hands, self.number_of_hands, self.number_of_nodes))
        return OR

    def reach_prob_of_world_node(self, node, hands):
        """ returns a single real number, which is reach probability of both players( multiplied, no chance probs) """
        return self.player_reach_probs_of_info_node(node, 0)[hands[0]] * self.player_reach_probs_of_info_node(node, 1)[hands[1]]

    def reach_prob__of_world_node_with_chance(self, node, hands):
        """ returns a single real number, which is real total reach probability of given world node """
        return self.player_reach_probs_of_info_node(node, 0)[hands[0]
               ] * self.player_reach_probs_of_info_node(node, 1)[hands[1]] * self.chance_reach_prob[hands[0], hands[1]]

    def reach_probs_of_world_nodes_table(self):
        """ returns (self.number_of_hands)*(self.number_of_hands)*(self.number_of_nodes) numpy array """
        R = np.ones(( self.number_of_hands, self.number_of_hands, self.number_of_nodes))
        for op_hand in range(self.number_of_hands):
            for ip_hand in range(self.number_of_hands):
                for node in self.game.node[1:]:
                    R[ op_hand, ip_hand, node,] = self.reach_prob_of_world_node(node, [op_hand, ip_hand])
        return R

    def reach_probs_of_world_nodes_with_chance_table(self):
        """ returns (self.number_of_hands)*(self.number_of_hands)*(self.number_of_nodes) numpy array """
        R = np.ones((self.number_of_hands, self.number_of_hands, self.number_of_nodes))
        R[:, :, 0] = self.chance_reach_prob.copy()
        for op_hand in range(self.number_of_hands):
            for ip_hand in range(self.number_of_hands):
                for node in self.game.node[1:]:
                    R[op_hand, ip_hand, node] = self.reach_prob__of_world_node_with_chance(node, [op_hand, ip_hand])
        return R

    # this ignore your hand! and it is only true for games with independent card dealing from deck
#   def opponents_reach_probs_table_ignorant(self):
#       OR = np.ones((2, self.number_of_hands, self.number_of_nodes))
#       for i in range(2):
#           for node in self.node[1:]:
#               OR[i, :, node:node + 1] = self.player_reach_probs(node, 1-i)*self.chance_reach_prob[:, 0:1]
#       return OR

# ---------------------------------- MAIN METHODS: EVALUATION OF GIVEN STRATEGY -------------------------------------- #
    """ whenever v is cf_value_world_nodes_table or cf_regrets_of_public_node_from_world_values or ...
      
          position info node values can be computed by np.sum(v[:,:,node], axis=1-position)
          op     info node values can be computed by np.sum(v[:,:,node], axis=1         )
          ip     info node values can be computed by np.sum(v[:,:,node], axis=0         )
          also if you drop the node will give a number_of_hands*number_of_nodes table where
          each column is info node values of corresponding node=column_index
          each row corresponds to hand in info node      
    """
    def values_of_world_nodes_table(self):
        """ returns (self.number_of_hands)*(self.number_of_hands)*(self.number_of_nodes) numpy array """

        number_of_hands = self.number_of_hands
        world_state_values = self.terminal_values_all_nodes.copy()
        given_strategy = self.strategy_base[0, :, :] * self.strategy_base[1, :, :]
        # world_state_value[ , , t.parent] += world_state_value[ , , t]**strategy[ , t]
        nonzero_nodes = self.node[1:]
        reverse_node_list = nonzero_nodes[::-1]

        for current_node in reverse_node_list:
            current_player = self.turn[current_node]
            parent_node = self.parent[current_node]
            parent_node_player = 1 - current_player

            # if parent_node player is op we multiply rows of value matrix
            if parent_node_player == 0:
                world_state_values[:, :, parent_node] += (
                        world_state_values[:, :, current_node] * (
                    given_strategy[:, current_node].reshape(number_of_hands, 1)))

            # else if parent_node player is ip we multiply cols of value matrix
            elif parent_node_player == 1:
                world_state_values[:, :, parent_node] += (
                        world_state_values[:, :, current_node] * given_strategy[:, current_node])

        return world_state_values

    def cf_value_world_nodes_table(self, position):
        """ returns (self.number_of_hands)*(self.number_of_hands)*(self.number_of_nodes) numpy array """
        return self.cf_reach_probs_of_world_nodes_table(position) * self.values_of_world_nodes_table()

    def cf_values_of_info_nodes_table(self, position):
        """ returns (self.number_of_hands)*(self.number_of_nodes) numpy array """
        return np.sum(self.cf_value_world_nodes_table(position)[:, :, :], axis=1 - position)

    # free of chance players probs
    def cf_value_of_info_node(self,  hand, node, player=None):
        """ returns a single real number, which is cf value of given info node, for the given player"""
        if player is None:
            player = self.turn[node]
        if player == 0:
            return np.sum(
                self.values_of_world_nodes_table()[hand, :, node][:, np.newaxis] * self.player_reach_probs_of_info_node(node, 1))
        elif player == 1:
            return np.sum(
                self.values_of_world_nodes_table()[:, hand, node][:, np.newaxis] * self.player_reach_probs_of_info_node(node, 0))

    def cf_values_of_info_nodes_of_decision_player_table(self):
        """ returns (self.number_of_hands)*(self.number_of_nodes) numpy array """
        cfv_info = np.zeros((self.number_of_hands, self.number_of_nodes))
        for j in self.node:
            p = self.turn[j]
            cfv_info[:, j:j+1] = np.sum(self.cf_value_world_nodes_table(p)[:, :, j], axis=1-p)[:, np.newaxis]
        return cfv_info

    def cf_regret_of_info_node(self, hand, node, child):
        """ returns a single real number, which is cf value of given info node, for the given player"""
        position = self.turn[node]
        return self.cf_value_of_info_node(hand, child, position)-self.cf_value_of_info_node(hand, node, position)

    def cf_regrets_of_of_info_nodes_table(self, position):
        """ returns (self.number_of_hands)*(self.number_of_nodes) numpy array """
        cfr_t = PR = np.zeros((self.number_of_hands, self.number_of_nodes))
        cfv_t = self.cf_values_of_info_nodes_table(position).copy()
        nn = self.node[1:]
        for j in nn:
            cfr_t[:, j:j+1] = cfv_t[:, j:j+1]-cfv_t[:, self.parent[j]:self.parent[j]+1]
        return cfr_t

    def cf_regrets_of_public_node_from_world_values(self, child):
        node = self.parent[child]
        p = self.turn[node]
        cf_r = (self.values_of_world_nodes_table()[:, :, child]-self.values_of_world_nodes_table()[:, :, node]
             )*self.cf_reach_probs_of_world_nodes_table(p)[:, :, node]
        return cf_r

# -------------------------------------------------------------------------------------------------------------------- #

    def update_cumulative_regrets(self):
        self.cumulative_regret[0, :, :] += self.cf_regrets_of_of_info_nodes_table(0)
        self.cumulative_regret[1, :, :] += self.cf_regrets_of_of_info_nodes_table(1)

    def update_strategy(self):
        cr = self.cumulative_regret.copy()
        cr_positive = np.where(cr >=0 , cr, 0)
        for index, d_node in enumerate(self.decision_node):
            turn = self.turn[d_node]
            children = self.decision_node_children[index]
            l = len(children)
            sum_r = np.sum(cr_positive[:, children[0]:children[0]+l], axis=0)
            for child in self.decision_node_children[index]:
                self.strategy_base[turn, :, child] = cr_positive[child]/sum_r

    def updated_strategy(self):
        strat = np.ones((2, self.number_of_hands, self.number_of_nodes))
        cr = self.cumulative_regret.copy()
        cr_positive = np.where(cr >= 0, cr, 0)
        for index, d_node in enumerate(self.decision_node):
            turn = self.turn[d_node]
            children = self.decision_node_children[index]
            n_children = len(children)
            sum_r = np.sum(cr_positive[turn, :, children[0]:children[0] + n_children], axis=1)
            for child in self.decision_node_children[index]:
                strat[turn, :, child] = cr_positive[turn, :, child] / sum_r
        return strat






# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------- STRATEGY INITIALIZING TOOLS AND SPECIFIC STRATEGIES ---------------------------------- #

    def uniform_strategy(self):
        S = np.ones((2, self.number_of_hands, self.number_of_nodes))
        for _decision_node in self.decision_node:
            childs = self.game.public_state[_decision_node].children
            for child in childs:
                S[self.turn[_decision_node], :, child:child + 1] = np.full((
                    self.number_of_hands, 1), 1 / len(childs))
        return S

    def update_strategy_base_to(self, action_prob_function):
        S = np.ones((2, self.number_of_hands, self.number_of_nodes))
        for i in range(self.number_of_hands):
            for _decision_node in self.decision_node:
                childs = self.game.public_state[_decision_node].children
                for child in childs:
                    S[self.turn[_decision_node], i, child:child + 1] = action_prob_function(i, child)
        return S


# ------------------------------- INITIALIZING STRATEGIC BETTING GAMES WITH USUAL SIZES ------------------------------ #

# Start a Game
if __name__ == '__main__':
    #    J = 0;
    #    Q = 1;
    #    K = 2
    #    KUHN_BETTING_GAME = BettingGame(bet_size=0.5, max_number_of_bets=2,
    #                                    deck={J: 1, Q: 1, K: 1}, deal_from_deck_with_substitution=False)
    #
    #    KK = KUHN_BETTING_GAME
    #    max_n = 12
    #    G = BettingGame(bet_size=1, max_number_of_bets=max_n,
    #                    deck={i: 1 for i in range(5)}, deal_from_deck_with_substitution=True)
    #
    #    SK = Strategy(KK)
    #    GK = Strategy(G)
    #    SK.strategy_base = SK.uniform_strategy()
    #    GK.strategy_base = GK.uniform_strategy()
    #
    #    # Testing
    #    test_start = np.ones((2, 3, 9))
    #    test_start[0, :, :] = np.array([[1, 0.5, 0.5, 1, 1, 1, 1, 0.5, 0.5], [1, 0.9, 0.1, 1, 1, 1, 1, 0.7, 0.3],
    #                                    [1, 0.05, 0.95, 1, 1, 1, 1, 0.1, 0.9]])
    #    test_start[1, :, :] = np.array([[1, 1, 1, 0.6, 0.4, 0.2, 0.8, 1, 1], [1, 1, 1, 0.3, 0.7, 0.35, 0.65, 1, 1],
    #                                    [1, 1, 1, 0.1, 0.9, 0.05, 0.95, 1, 1]])
    #
    #    TS = Strategy(KK, strategy_base=test_start)
    #    V_TS = TS.values_of_world_nodes_table()
    #    sr = TS.reach_probs_of_world_nodes_table()
    #    sr_cf = TS.cf_reach_probs_table()

    KKK = BettingGame(bet_size=0.5, max_number_of_bets=2,
                                    deck={0: 1, 1: 1, 2: 1}, deal_from_deck_with_substitution=False)
    ttest_start = np.ones((2, 3, 9))
    ttest_start[0, :, :] = np.array([[1, 0.5, 0.5, 1, 1, 1, 1, 0.5, 0.5], [1, 0.9, 0.1, 1, 1, 1, 1, 0.7, 0.3],
                                    [1, 0.05, 0.95, 1, 1, 1, 1, 0.1, 0.9]])
    ttest_start[1, :, :] = np.array([[1, 1, 1, 0.6, 0.4, 0.2, 0.8, 1, 1], [1, 1, 1, 0.3, 0.7, 0.35, 0.65, 1, 1],
                                    [1, 1, 1, 0.1, 0.9, 0.05, 0.95, 1, 1]])
    TTS = Strategy(KKK, strategy_base=ttest_start)
    V = TTS.values_of_world_nodes_table()
    R = TTS.cf_reach_probs_of_world_nodes_table(1)
    cfv = TTS.cf_value_world_nodes_table(1)
    cfv_inf = TTS.cf_values_of_info_nodes_of_decision_player_table()

