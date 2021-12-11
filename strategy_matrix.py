# THIS APPROACH SEEMS TO NOT WORK....PAUSED FOR NOW
import numpy as np
import copy
import time
import general_utilities as gu
import betting_games as bg
from scipy.sparse import csr_matrix, lil_matrix, identity
from scipy.sparse.linalg import spsolve
from numpy.random import rand


# Transition Matrix Of Strats: (3n+3)(3n+3) matrix S:  S[i,j] shows chance of moving from i to j
# properties of S:
# Each decision row S[i]: sums up to 1 -- has at most 3 nonzero element corresponding to i;s children
# all non-decision row are 0
# Pure strat have matrices with just 0,1. Exactly one 1 in each decision row.
# Stable transition matrices have 1 on terminal diagonal elements which makes them stay at terminal states for ever!


# 3n+3 nodes, (3n+3)^2 TransitionMatrix or StrategyMatrix which all are determined by strategy or  strategy vector
class StrategyMatrix:
    def __init__(self, game: bg.BettingGame):
        self.game = game
        self.dim = len(self.game.node)

        self.csr_row_indices = self._csr_row_indices()
        self.csr_col_indices = self.game.node
        self.start = csr_matrix(np.array([[int(i == 0) for i in self.game.node]]))
        self.diagonal_terminal = np.diag([int(self.game.public_state[i].is_terminal) for i in self.game.node])

# -------------------------------------- transition matrix of some basic strategies:---------------------------------- #

        self.fold_all = csr_matrix((self.fold_all_csr_data(), (self.csr_row_indices, self.csr_col_indices)),
                                   shape=(self.dim, self.dim))
        self.fold_all_stable = csr_matrix(self.fold_all.todense() + self.diagonal_terminal)

        #  tran
        self.uniform = csr_matrix((self.uniform_csr_data(), (self.csr_row_indices, self.csr_col_indices)),
                                  shape=(self.dim, self.dim))

# -------------------------------------------------------------------------------------------------------------------- #

    # returns sparse matrix that row i of it shows probabilistic state at time i (use .todense() to see it)
    # if using stable trans_mat after max depth it doesnt change and each terminal element of fix row is reach
    # probability of that state and reach probability of decision state is equal to sum of reach prob of all of its
    # terminal descendant
    def transition_seq(self, trans_mat):
        predicted_states = [self.start]
        for i in range(self.dim):
            predicted_states.append(predicted_states[i].dot(trans_mat))
        return predicted_states

    def _csr_row_indices(self):
        indices = [0]
        counter = 0
        for decision_node in self.game.public_tree.decision_nodes:
            for j in self.game.public_state[decision_node].children:
                indices.append(decision_node)
                counter += 1
        return np.array(indices)

    def to_stable(self, csr_mat):
        return csr_matrix(csr_mat.todense() + self.diagonal_terminal)

    def fold_all_csr_data(self):
        data = np.zeros(self.dim)
        for decision_node in self.game.public_tree.decision_nodes[1:]:
            data[list(self.csr_row_indices).index(decision_node)] = 1
        data[1] = 1
        return np.array(data)

    def uniform_csr_data(self):
        data = np.zeros(self.dim)
        for i in self.game.node:
            if 1 <= i <= 4:
                data[i] = 1 / 2
            elif self.dim - 4 <= i < self.dim:
                data[i] = 1 / 2
            elif 4 < i < self.dim - 4:
                data[i] = 1 / 3
        return data


# ------------------------------------ Functions that help working on Strategies  ------------------------------------ #


def strategy_applier(strat_matrix):
    def multiplier(current):
        return current.dot(strat_matrix)

    return multiplier


# ------------------------------------ Initializing Strategies and Analyzing them ------------------------------------ #
t = {0: time.perf_counter()}
# Two famous betting game as an example:

G = bg.BettingGame(12)
K = bg.BettingGame(bet_size=0.5, max_number_of_bets=2,
                   deck={0: 1, 1: 1, 2: 1}, deal_from_deck_with_substitution=False)

SK = StrategyMatrix(K)
SG = StrategyMatrix(G)

Pstates2 = SK.game.public_state

t[1] = time.perf_counter()
TSKSU = SK.transition_seq(SK.to_stable(SK.uniform))
t[2] = time.perf_counter()

TSGSU = SG.transition_seq(SG.to_stable(SG.uniform))
t[3] = time.perf_counter()

sg100 = StrategyMatrix(bg.BettingGame(100))
t[4] = time.perf_counter()

sg100_dense_uniform_trans = sg100.uniform.todense()
t[5] = time.perf_counter()

# ts100su = sg100.transition_seq(sg100.to_stable(sg100.uniform))

# ------------------------ checking basic functionality of transition matrices and csr_matrix ------------------------ #
# fr = SG.csr_row_indices
# fc = SG.csr_col_indices
# fd = SG.fold_all
# fds = SG.fold_all_stable
# fad = SG.fold_all_csr_data()
# start = SG.start
# s1 = start.dot(fd)
#
# fd_applier = strategy_applier(fd)
# fds_applier = strategy_applier(fds)
# c1 = fd_applier(start)
# d1 = c1.todense()
# c2 = fd_applier(c1)
# d2 = c2.todense()
# c3 = fd_applier(c2)
# d3 = c3.todense()
#
# fds_applier = strategy_applier(fds)
# cs1 = fds_applier(start)
# ds1 = cs1.todense()
# cs2 = fds_applier(cs1)
# ds2 = cs2.todense()
# cs3 = fds_applier(cs2)
# ds3 = cs3.todense()
# cs4 = fds_applier(cs3)
# ds4 = cs3.todense()
# ----------------------------- Seems that everything checked above worked as expected ------------------------------- #

# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------- Ideas That Did Not Work----------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------- Solving xA=x is not useful, singular A-I-------------------------------------- #
## Does solving xA=x helps to find final invariant distribution over states?
## vA=v --> A^tv^t=v^t-->(A^t-I)v^t=0
# At = csr_matrix.getH(A)
# adjusted_At = At - identity(sg100.dim)
# b = np.zeros(sg100.dim)
# bt = csr_matrix.getH(b)
# x = spsolve(A, bt)
# A = sg100.to_stable(sg100.uniform)
# v = ts100su[99]
# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------- to_super_stable instead of to_stable ----------------------------------------- #
# trans_mat_K_super_stable = SK.transition_seq(SK.to_super_stable(SK.uniform))
#  def to_super_stable(self, csr_mat):
    #  return csr_matrix(csr_mat.todense() + identity(self.dim))
# -------------------------------------------------------------------------------------------------------------------- #

