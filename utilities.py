import numpy as np

# Constants
FOLD = 'Fold';  CHECK = 'Check';  CALL = 'Call';  BET = 'Bet';  RAISE = 'Raise'
TERMINAL = 'Terminal';  DECISION = 'Decision';  START = 'Start'
# Check or Fold = 0 --- Bet or Call = 1 --- Raise = 2
IP_ACTIONS = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [0, 2, 0, 1]])
OP_ACTIONS = np.array([[0, 0, 1, 1], [1, 0, 0, 1], [2, 0, 0, 1]])
ALL_ACTIONS = np.array([[[0, 0, 1, 1], [1, 0, 0, 1], [2, 0, 0, 1]], [[0, 0, 1, 1], [0, 1, 0, 1], [0, 2, 0, 1]]])
# ALL_ACTIONS[i][j] is:
# j'th action of i'th player for i= 0,1 (OP, IP) and j = 0, 1, 2 (Check or Fold, Bet or Call, Raise)
ACTION_NAME_TO_ID = dict({'Fold': 0, 'Check': 0, 'Call': 1, 'Bet': 1, 'Raise': 2})

START_NODE = np.array([0, 0, 0, 0])

# RBC_COUNTER.dot(node) returns Total number of 2*Raises+Bet+Call (for both players)
RBC_COUNTER = np.array([1, 1, 0, 0])


def create_hands_dealings(OPDeck, IPDeck, substitution=True):
    cards_dealings = []
    if substitution:
        cards_dealings = [(i, j) for i in OPDeck for j in IPDeck]
    else:
        for i in OPDeck:
            for j in IPDeck:
                if j != i:
                    cards_dealings.append((i, j))
    return cards_dealings


