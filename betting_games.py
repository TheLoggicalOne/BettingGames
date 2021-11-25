""" Here we can create Betting Games with different MaxNumberOfBets and BetSizes
Betting Game Is a simplified model for one street of betting in poker.
For example Kuhn poker is BettingGame(BetSize=0.5, MaxNumberOfRaises=2). For more information about betting game see the
project documentation or the article.

"""
import numpy as np
import public_nodes_tree as public_nodes_tree
from collections import namedtuple
from utilities import START_NODE


class BettingGame:
    """ Creates a betting game with following specification(inputs):

    MaxNumberOfBets: int, Total number of bets allowed. Call count as one bet and each Raise count as 2 bets.(since
        technically a raise is a call and then a bet)
    BetSize: float, showing size of bet in relation to pot. This is fix in game and has default value of 1.
    Deck: iterable, Contains different possible hands for player, equivalent to deck of cards in poker. Player hands
        will be drawn from Deck randomly using GetHand or Chance Node actions.
    DealFromDeckWithSubstitution: Boolean

    Application and Usage:
    self.tree : PublicTree.Tree object, gives the public tree of the game and let you access everything about public
        nodes and their tree structure through methods and attrs available in PublicTree.Tree class.
    self.PublicNodesAsTuple: list, each public nodes as namedtuple is one element of this list , in order of their
        BF ID (BreathFirst Tree Creation ID). This light and fast representation of public nodes is actually a reference
        to PublicTree.Tree.CreateNamedTuple. For simplicity we consider the node with ID equal to i as node i, also
        action with id equal to j as action j
        Let S=self.PublicNodesAsTuple. Then S[i] is a namedtuple representation of public node i. The fields of this
        namedtuple let us access almost every information about the node:
            S[i].ID
            S[i].node: numpy.ndarray rep of node i
            S[i].actions: ordered list of actions available in node i (just action id, not the np.ndarray rep of them)
            S[i].children: ordered list of children of node i (just child id, not the np.ndarray rep of them)
            S[i].parent: parent of node i
            S[i].turn: whose player move is at node i
            S[i].is_terminal
            S[i].first_played_action: The first of OP action in the StartNode has value of 'Check' or 'Bet'
            S[i].last_played_action: The last played action that has led us to current node.
    self.Play: numpy.ndarray, this np array makes moving from one state (public node) to its child easy and fast.
        i.e: self.play[i,j] shows the result(ID of resulting node) of applying action j at node i
    self.CommonHistory: 2D-array, self.CommonHistory[i][j] is a list containing all common nodes in the paths from
        StartNode to current nodes.
    self.InfoSet: create the raw namedtuple to be used as info set
        """

    def __init__(self, max_number_of_bets, deck, bet_size=1, deal_from_deck_with_substitution=True):
        self.max_number_of_bets = max_number_of_bets
        self.bet_size = bet_size
        self.deck = deck
        self.deal_from_deck_with_substitution = deal_from_deck_with_substitution

        self.public_tree = public_nodes_tree.PublicTree(START_NODE, self.max_number_of_bets)
        self.nodes = self.public_tree.nodes
        self.public_state = self.public_tree.create_PublicState_list()
        self.next_node = self.public_tree.create_next_node_table()

        self.InfoNode = namedtuple('InfoNode'+str(self.max_number_of_bets), ['node', 'hand'])
        self.InfoState = namedtuple('InfoState' + str(self.max_number_of_bets), ['public_state', 'hand'])
        self.WorldNode = namedtuple('WorldNode'+str(self.max_number_of_bets), ['node', 'op_hand', 'ip_hand'])
        self.WorldState = namedtuple('State' + str(self.max_number_of_bets),
                                     ['public_state', 'op_hand', 'ip_hand'])

        self.first_common_ancestors = np.array([[self.public_tree.common_ancestors(i, j)[-1] for i in self.nodes]
                                               for j in self.nodes])

    def info_node(self, hand):
        return [self.InfoNode(node=node, hand=hand) for node in self.nodes]

    def world_node(self, op_hand, ip_hand):
        return [self.WorldNode(node=node, op_hand=op_hand, ip_hand=ip_hand) for node in self.nodes]


#class OPPureStrategy:
#    def __init__(self, Player, MaxTargetNumberOfBets, LastActionVsHigherThanTarget):
#        pass


# ------------------------------------ INITIALIZING BETTING GAMES WITH USUAL SIZES ----------------------------------- #


# Start a Game with MaxNumberOfRaise = n
if __name__ == '__main__':
    m = 12
    d = list(range(100))
    G = BettingGame(m, d)
    T = G.public_tree
    Play = G.next_node
    Pstate = G.public_state
    State12 = G.WorldState
    x = State12(public_state=10, op_hand=95, ip_hand=70)
    H = [G.public_tree.common_ancestors_table[i][i] for i in G.public_tree.nodes]
    AH = [G.public_tree.history_of_node(i) for i in G.public_tree.nodes]
#test
print('Testing Local VCS')
