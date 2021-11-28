import numpy as np
import copy
import general_utilities as gu
import betting_games as bg
from scipy.sparse import csr_matrix, lil_matrix
from numpy.random import rand

# Matrix Rep Of Strats: (3n+3)(3n+3) matrix S:  S[i,j] shows chance of moving from i to j
# properties of S:
# Each decision row S[i]: sums up to 1 -- has at most 3 nonzero element corresponding to i;s children
# all non-decision row are 0
# Pure strat have matrices with just 0,1. Exactly one 1 in each decision row.

G = bg.BettingGame(12)
K = bg.BettingGame(bet_size=0.5, max_number_of_bets=2,
                   deck={0: 1, 1: 1, 2: 1}, deal_from_deck_with_substitution=False)


class StrategyMatrix:
    def __init__(self, game: bg.BettingGame):
        self.game = game
        self.n = len(self.game.nodes)
        self.csr_row_indices = self._csr_row_indices()
        self.csr_col_indices = self.game.nodes
        self.start = csr_matrix(np.array([[int(i == 0) for i in self.game.nodes]]))
        self.fold_all = csr_matrix((self.fold_all_csr_data(), (self.csr_row_indices, self.csr_col_indices)),
                                   shape=(39, 39))
        self.diagonal_terminal = np.diag([self.game.public_state[i].is_terminal for i in self.game.nodes])
        self.fold_all_stable = csr_matrix(self.fold_all.todense() + self.diagonal_terminal)

    def _csr_row_indices(self):
        indices = [0]
        counter = 0
        for decision_node in self.game.public_tree.decision_nodes:
            for j in self.game.public_state[decision_node].children:
                indices.append(decision_node)
                counter += 1
        return np.array(indices)

    def fold_all_csr_data(self):
        data = np.zeros(len(self.game.nodes))
        for decision_node in self.game.public_tree.decision_nodes[1:]:
            data[list(self.csr_row_indices).index(decision_node)] = 1
        data[1] = 1
        return np.array(data)


# ------------------------------------ Initializing Strategies and Analyzing them ------------------------------------ #


def strategy_applier(strat_matrix):
    def multiplier(current):
        return current.dot(strat_matrix)

    return multiplier


# ------------------------------------ Initializing Strategies and Analyzing them ------------------------------------ #


SG = StrategyMatrix(G)
fr = SG.csr_row_indices
fc = SG.csr_col_indices
fd = SG.fold_all
fds = SG.fold_all_stable
fad = SG.fold_all_csr_data()
start = SG.start
s1 = start.dot(fd)

fd_applier = strategy_applier(fd)
fds_applier = strategy_applier(fds)
c1 = fd_applier(start)
d1 = c1.todense()
c2 = fd_applier(c1)
d2 = c2.todense()
c3 = fd_applier(c2)
d3 = c3.todense()

fds_applier = strategy_applier(fds)
cs1 = fds_applier(start)
ds1 = cs1.todense()
cs2 = fds_applier(cs1)
ds2 = cs2.todense()
cs3 = fds_applier(cs2)
ds3 = cs3.todense()
cs4 = fds_applier(cs3)
ds4 = cs3.todense()



