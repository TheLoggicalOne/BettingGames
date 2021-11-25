# Betting Games
Betting Game is a two player zero-sum imperfect information game. It can be seen as a simplified model for one street of
heads up betting in poker. Betting Games can be seen as a general 
case of 'Kuhn Poker':
Actually BettingGame(BetSize=0.5, max_number_of_bets=2, deck = {J, Q, K}, deal_from_deck_with_substitution=True) is 
exactly Kuhn Poker. 

## Betting Game description in conventional poker term:
Each player put 1 **'chip'** in the **'pot'**, which is unit of money in the game.

Each player is dealt a **'hand'** from the **'deck'** with specified given **'distribution'** which is usually uniform. 

Then player go to one round of betting, similar to one street of betting in poker. With two important difference:
First there is only one bet size allowed. (Note that sizes are in relation to pot size). Second there is a limit on max 
number of bets possible. (with bet size being fix, this is equivalent to having a limit on the **'stack'** of players). 

**'deck'** in this game is set of numbers: {1, 2, ..., m}. Players deck might be different, it might be equal but 
physically 

**'dealing cards'**: is this each player draw his hand randomly according to given distribution which is usually uniform
**'Showdown'**: player holding the hand with higher number wins


## Our Model:
We consider the extensive form representation of this game. 

#### node:
By node we actually mean 'public state'  which is actually same public representation of decision nodes in 
BettingGames(also many other games like poker and its variants). Note that public representation of decision points 
means representation of decision points based on information that is available publicly to anyone.


## Possible approaches to modeling and analyzing the Betting Game:
 we are working on G = BettingGame(n) and T is its BF tree....Also let say we have m possible different hand
 this game has 3n+3 nodes.
 n+2 of them are decision nodes. First 2 and last 2 decision node has 2 choices. n-2 other ones have 3 choices
 For now let say n=12:
 Strategy can be seen as:
 1)Pure Strategies(PS): For each possible holding hand we should decide:
 As OP: What do you want your last raise make the 2*R+B ? ( assuming opponent is coming!) What is your action in case
 Opponent passed your 2*R+B? call or fold?
 Therefore, for each possible holding hand for OP there [(n/2)+1]*2=n+2 choices. For n=12: 14 -- For n=4: 6 -- For n=6:8
 As IP: Facing Bet: Go for Max 2R =< n/2. Call or Fold in case opponent passed your Max 2*R
 As IP: Facing Check: Go for Max 2R+B =< n/2 . all or Fold in case opponent passed your Max 2*R+B
 Therefore for each possible holding hand for IP there [(n/2)+1]*2=n+2 choices. For n=12: 14 -- For n=4: 6 -- For n=6:8
 So Total Number Of Pure Strategies for each position  Is: (n+2)^m
 For n=2,m=10: 10^6 -- For n=4,m=10: 6*10^7 -- For n=6,m=10: 10^9
 2) Mixed Strategies: Probabilistic weighted combo of Pure Strategies
 Important question: Each Pure Strategy ps: (Pstate, Hand) ---> Action. i.e:
 3) Behavioral Strategy: On each Istate = (Pstate, Hand) determines distribution over possible actions
 Finally, How should we define Strats pure or behavioral or mixed?
 We could construct them as a tree of table, and work with them. Which one is better?
 One idea is to consider behavioral strats as tree to fully use tree structure of the game
 Can we do everything in terms of matrix and np arrays?
 C(5,3)=10 -- C(8,3)=56 -- C(10,3)=120 -- C(20,3)=1140 --
 Monte Carlo Tree Search:
 See your action as Fold/Check, Bet/Call, Raise cut offs

