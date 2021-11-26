""" This module can be seen as Betting Game API. Here  we can create Betting Games with different specification
parameters. Betting Game Is a simplified model for one street of betting in poker. For more information about betting
game see the ReadMe.md or the article. For general definition and description of the game see ReadMe-Game Description,
for our specific representation of the game that is also used in this API see ReadMe-Game Representation
"""
import numpy as np
import public_nodes_tree as public_nodes_tree
from collections import namedtuple
from utilities import START_NODE


class BettingGame:
    """ Creates a betting game object with following specification parameters (inputs):

    max_number_of_bets: int, Total number of bets allowed. Call count as one bet and each Raise count as 2 bets.(since
        technically a raise is a call and then a bet)

    bet_size: float, showing size of bet in relation to pot. This is fix in game and has default value of 1.

    deck: a dictionary {a_i:x_i for i in range(1, m)} where dictionary keys a_i, for i=1,2, ...,m show different
    possible cards(or hands) in the deck, and each dictionary  value `x_i` shows how many of card `a_i` we have in the
    deck.
    # TODO add second deck to inputs
    deal_from_deck_with_substitution: Boolean


    Application and Usage:

    self.public_tree : public_nodes_tree.PublicTree object, gives the public tree of the game and let you access
        everything about public states and their tree structure through methods and attrs available in
        public_nodes_tree.PublicTree class. For more info see public_nodes_tree.PublicTree documentation.

    self.public_state: list, each public state as namedtuple is one element of this list , in order of their
        creation in breath_first tree. This light and fast representation of public states is actually a reference
        to public_node_tree.public_tree.create_PublicState_list()
        Let S=self.public_state, Then S[i] is public node i representation as namedtuple public_nodes_tree.PublicState()
        Fields of this namedtuple let us access almost every information about the node:
            S[i].node: int, equal to i, public node number or name
            S[i].np_node: numpy.ndarray rep of node i
            S[i].actions: ordered list of actions available in node i
            S[i].children: ordered list of children of node i
            S[i].parent: parent of node i
            S[i].to_move: 0 or 1, showing whose player move is at node i
            S[i].is_terminal: boolean
            S[i].first_played_action: 'Check' or 'Bet', showing first action of in the StartNode
            S[i].last_played_action: 'Check' or 'Call' or 'Bet' or 'Raise', showing The last played action that has led
             us to current node.

    self.next_node: numpy.ndarray, this np array makes moving from one state (public node) to its child easy and fast.
        i.e: self.next_node[i,j] shows the resulting node of taking action j at node i

    self.first_common_ancestors: numpy 2D-array, self.first_common_ancestors[i][j] is a list containing all common nodes
        in the paths from StartNode to current nodes.

    self.InfoState: create the raw namedtuple with fields public_states and hand, to create namedtuple representation of
        InfoStates of game
    self.InfoNode: create the raw namedtuple with fields node and hand, to create namedtuple representation of
        InfoNodes of game
    self.WorldNode: create the raw namedtuple with fields node and op_hand and ip_hand to create namedtuple
        representation of WorldNodes of game
    self.WorldState: create the raw namedtuple with fields public_states and op_hand and ip_hand to create namedtuple
        representation of WorldStates of game
        """

    def __init__(self, max_number_of_bets,
                 deck={i: 1 for i in range(100)},
                 bet_size=1,
                 deal_from_deck_with_substitution=True):
        self.max_number_of_bets = max_number_of_bets
        self.bet_size = bet_size
        self.deck = deck
        self.deal_from_deck_with_substitution = deal_from_deck_with_substitution

        self.public_tree = public_nodes_tree.PublicTree(START_NODE, self.max_number_of_bets)
        self.nodes = self.public_tree.nodes
        self.public_state = self.public_tree.create_PublicState_list()
        self.next_node = self.public_tree.create_next_node_table()

        self.InfoNode = namedtuple('InfoNode' + str(self.max_number_of_bets), ['node', 'hand'])
        self.InfoState = namedtuple('InfoState' + str(self.max_number_of_bets), ['public_state', 'hand'])
        self.WorldNode = namedtuple('WorldNode' + str(self.max_number_of_bets), ['node', 'op_hand', 'ip_hand'])
        self.WorldState = namedtuple('State' + str(self.max_number_of_bets),
                                     ['public_state', 'op_hand', 'ip_hand'])

        self.first_common_ancestors = np.array([[self.public_tree.common_ancestors(i, j)[-1] for i in self.nodes]
                                                for j in self.nodes])

    def info_node(self, hand):
        """ Creates a list, i'th element of list is self.InfoNode(node=i, hand=hand), namedtuple representation of info
            node corresponding to public node i holding the given hand. """

        return [self.InfoNode(node=node, hand=hand) for node in self.nodes]

    def info_state(self, hand):
        """ Creates a list, i'th element of list is self.InfoState(public_state=self.public_state[i], hand=hand),
            namedtuple representation of info
            node corresponding to public node i holding the given hand """
        return [self.InfoState(public_state=self.public_state[node], hand=hand)
                for node in self.nodes]

    def world_node(self, op_hand, ip_hand):
        """ Creates a list, i'th element of list is self.WorldNode(node=node, op_hand=op_hand, ip_hand=ip_hand),
            namedtuple representation of world node corresponding to public node i, players holding the given hands """
        return [self.WorldNode(node=node, op_hand=op_hand, ip_hand=ip_hand) for node in self.nodes]

    def world_state(self, op_hand, ip_hand):
        """ Creates a list, i'th element of list is
            self.WorldState(public_state=self.public_state[i], op_hand=op_hand, ip_hand=ip_hand),
            namedtuple representation of world state corresponding to public state i holding the given hands """
        return [self.WorldState(public_state=self.public_state[node], op_hand=op_hand, ip_hand=ip_hand)
                for node in self.nodes]


# class OPPureStrategy:
#    def __init__(self, Player, MaxTargetNumberOfBets, LastActionVsHigherThanTarget):
#        pass


# ------------------------------------ INITIALIZING BETTING GAMES WITH USUAL SIZES ----------------------------------- #


# Start a Game
if __name__ == '__main__':
    max_number_of_bets = 12
    G = BettingGame(max_number_of_bets)
    T = G.public_tree
    Play = G.next_node
    Pstate = G.public_state
    State12 = G.WorldState
    x = State12(public_state=10, op_hand=95, ip_hand=70)
    H = [G.public_tree.common_ancestors_table[i][i] for i in G.public_tree.nodes]
    AH = [G.public_tree.history_of_node(i) for i in G.public_tree.nodes]
    decision_states = [G.public_state[i] for i in G.public_tree.decision_nodes]


