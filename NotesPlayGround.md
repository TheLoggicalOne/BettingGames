 # TO BE COMPLETED
 # Just a Play Ground
 Consider ` G = betting_game(n)` .Let `T` be is its Breath First tree....Also let say we have `m` possible different  
 hand. This game:  
 
**3n+3 nodes**  

 **n+2 decision nodes**:  
 First 2 and last 2 decision nodes has 2 choices. n-2 other ones have 3 choices  
 
Also, for each player define 'Max Number Of Bet Target', **MNBT** as:  `MNBT = 2R+B`  

For now let say n=12:
 Strategy can be seen as:  
#### Pure Strategies(PS): 
For each possible holding hand we should decide:

 **As OP:** What do you want your last raise make the **MNBT** ? ( assuming opponent is coming!) What is your action in   
 case opponent passed your **MNBT**? call or fold?  
 Therefore, for each possible holding hand for OP there `[(n/2)+1]*2=n+2` choices.  
 
**As IP:** Facing Bet: Go for Max `2R =< n/2`. Call or Fold in case opponent passed your Max `2*R`  
  **As IP:** Facing Check: Go for Max `2R+B =< n/2` . all or Fold in case opponent passed your Max `2*R+B`  
 Therefore for each possible holding hand for IP there [(n/2)+1]*2=n+2 choices.   
 
So Total Number Of Pure Strategies for each position  Is: `(n+2)^m `   
For n=2,m=10: 10^6 -- For n=4,m=10: 6*10^7 -- For n=6,m=10: 10^9  
Note that this is all pure strats, even crazy ones and dominated ones  
**Important Question** How can we consider non-dominated ones? 

#### Mixed Strategies:
Probabilistic weighted combo of Pure Strategies  
 Important question: Each Pure Strategy ps: (Pstate, Hand) ---> Action. i.e:

#### Behavioral Strategy: On each Istate = (Pstate, Hand) determines distribution over possible actions
 Finally, How should we define Strats pure or behavioral or mixed?
 We could construct them as a tree of table, and work with them. Which one is better?
 One idea is to consider behavioral strats as tree to fully use tree structure of the game
 Can we do everything in terms of matrix and np arrays?
 
#### Another perspective to Strats
**See your action as Fold/Check, Bet/Call, Raise cut-offs at each public nodes**
C(5,3)=10 -- C(8,3)=56 -- C(10,3)=120 -- C(20,3)=1140 --
 Monte Carlo Tree Search:
 See your action as Fold/Check, Bet/Call, Raise cut-offs
 
##### Information Nodes and Information States
Number Of **Public Nodes** is `n+2`  
Number of **Information States** is `(n+2)*m`  
Number of **World States** is `(n+2)*m^2`  




