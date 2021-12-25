"""
Basics components of counterfactual regret minimization algorithms: definition and implementation

    Strategy
    Reach Probabilities
    Utilities( or values)
    Counterfactual Utilities( or values)
    Regret

    Note that everything we need can be computed from reach probability and utilities.
    So for given strategy we compute reach probabilities and utilities.
    Strategies are defined as:
    Strat as list of  willingness probs of coming to public node k. Note that each public node k  has only one parent
    which is accessible by G.public_state[k].parent
"""
import copy
import time
import numpy as np
from betting_games import BettingGame
from strategy import Strategy

np.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=450, suppress=None, nanstr=None,

                    infstr=None, formatter=None, sign=None, floatmode=None, legacy=None)

# -------------------------------- INITIALIZING EXAMPLE OF STRATEGIC BETTING GAMES ----------------------------------- #

KUHN_BETTING_GAME = BettingGame(bet_size=0.5, max_number_of_bets=2,
                                deck={0: 1, 1: 1, 2: 1}, deal_from_deck_with_substitution=False)
K = copy.deepcopy(KUHN_BETTING_GAME)

K_true = BettingGame(bet_size=0.5, max_number_of_bets=2,
                                deck={0: 1, 1: 1, 2: 1}, deal_from_deck_with_substitution=True)

K_b1 = BettingGame(bet_size=1, max_number_of_bets=2,
                                deck={0: 1, 1: 1, 2: 1}, deal_from_deck_with_substitution=False)

K_d6 = BettingGame(bet_size=0.5, max_number_of_bets=2,
                                deck={i: 1 for i in range(6)}, deal_from_deck_with_substitution=False)

K_b1_max4 = BettingGame(bet_size=1, max_number_of_bets=4,
                                deck={0: 1, 1: 1, 2: 1}, deal_from_deck_with_substitution=False)

K_b4 = BettingGame(bet_size=4, max_number_of_bets=2,
                                deck={0: 1, 1: 1, 2: 1}, deal_from_deck_with_substitution=False)

K_d12 = BettingGame(bet_size=0.5, max_number_of_bets=2,
                                deck={i: 1 for i in range(12)}, deal_from_deck_with_substitution=False)

K_d100_1 = BettingGame(max_number_of_bets=2,
                                deck={i: 1 for i in range(100)},bet_size=1, deal_from_deck_with_substitution=False)

G = BettingGame(4, deck={i: 1 for i in range(8)})




class Vanilla_Cfr:
    def __init__(self, game, strategy=None, number_of_iteration=10000, ):
        self.game = game
        self.strategy = Strategy(self.game, strategy)
        self.strategy.strategy_base = self.strategy.uniform_strategy()
        self.number_of_iteration = number_of_iteration

        self.speed_per_1000_iter = self.strategy.run_base_cfr(self.number_of_iteration)
        self.nash_eq = self.strategy.average_strategy()
        self.nash_eq_strategic_analysis = Strategy(self.game, self.nash_eq)
        self.game_nash_value_of_world_nodes = self.nash_eq_strategic_analysis.values_of_world_nodes_table()
        self.game_nash_value = np.sum(self.game_nash_value_of_world_nodes[:, :, 0])
        self.average_regret_of_world_nodes = self.strategy.cumulative_regret/self.strategy.iteration
        self.number_of_iteration = self.strategy.iteration


#K_d100 = BettingGame( 2, {i: 1 for i in range(100)}, 0.5, False)
#vk100 = Vanilla_Cfr(K_d100, strategy=None, number_of_iteration=5000000)
#sk100 = vk100.strategy
#np.savez('V_2_100_50_0', sk100.cumulative_regret, sk100.cumulative_strategy, vk100.nash_eq)

#K_d100_1 = BettingGame( 2, {i: 1 for i in range(100)}, 1, False)
#vk100_1 = Vanilla_Cfr(K_d100_1, strategy=None, number_of_iteration=5000000)
#sk100_1 = vk100.strategy
#np.savez('V_2_100_100_0', sk100_1.cumulative_regret, sk100_1.cumulative_strategy, vk100_1.nash_eq)

#K_max4 =  BettingGame(0.5, 4, {0: 1, 1: 1, 2: 1}, False)
#vkmax4 = Vanilla_Cfr(K_max4, strategy=None, number_of_iteration=2000000)
#skmax4 = vkmax4.strategy
#np.savez('V_4_100_50_0', skmax4.cumulative_regret, skmax4.cumulative_strategy, vkmax4.nash_eq)

#K_max6 =  BettingGame( 6, {0: 1, 1: 1, 2: 1}, 0.5, False)
#vkmax6 = Vanilla_Cfr(K_max6, strategy=None, number_of_iteration=2000000)
#skmax6 = vkmax6.strategy
#np.savez('V_6_100_50_0', skmax6.cumulative_regret, skmax6.cumulative_strategy, vkmax6.nash_eq)

K_max8 =  BettingGame( 8, {0: 1, 1: 1, 2: 1}, 0.5, False)
vkmax8 = Vanilla_Cfr(K_max8, strategy=None, number_of_iteration=2000000)
skmax8 = vkmax8.strategy
np.savez('V_8_100_50_0', skmax8.cumulative_regret, skmax8.cumulative_strategy, vkmax8.nash_eq)





# vk100_1 = Vanilla_Cfr(K_d100_1, strategy=None, number_of_iteration=1000000)
# np.savez('B_test_100', sk100.cumulative_regret, sk100.cumulative_strategy, vk100.nash_eq)
# npztest = np.load('B_test_100.npz')


#t0 = time.perf_counter()
#cfrs = [Vanilla_Cfr(game) for game in games_list]
#cfrs2 = [Vanilla_Cfr(game, None, number_of_iteration=20000) for game in games_list]
#t1 = time.perf_counter()
#dur = t1-t0
#
#cfrs_number_of_iteration = [v.number_of_iteration for v in cfrs]
#cfrs_game_values = [v.game_nash_value/6 for v in cfrs]


#_kb_4= Vanilla_Cfr(K_b1_max4, strategy=None, number_of_iteration=100000)
#_kb_4_1000000= Vanilla_Cfr(K_b1_max4, strategy=None, number_of_iteration=1000000)
#_k_d6= Vanilla_Cfr(K_d6, strategy=None, number_of_iteration=100000)
#n = v_k_d6
#_k_d12= Vanilla_Cfr(K_d12, strategy=None, number_of_iteration=100000)
#12=v_k_d12
#k12 = Vanilla_Cfr(K_d12, strategy=None, number_of_iteration=100000)

#k12_2 = Vanilla_Cfr(K_d12, strategy=None, number_of_iteration=2000000)

# games_list = [K, K_true, K_b1, K_d6, K_max4, K_b1_max4, K_b4]





