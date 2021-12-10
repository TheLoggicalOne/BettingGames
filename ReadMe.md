# Betting Game: Optimal Strategies and Its Applications 
The goal of this project is to solve Betting Game and use this game and its solution and optimal strategies to model and  
analyze other games and problems, specifically poker games. More specifically this project will provide:
- [**Game description and representation**](#game-description-and-representation)
- [**API for the Betting Game**](betting_games.py)      
- [**Efforts To Solve The Betting Game**](#efforts-to-solve-the-betting-game)
  - **Counterfactual regret minimization efficient implementations for Betting Games**
- **Applying Betting Game in other problems and games**  

This is undergoing project with undergoing research behind it. The Table of Contents and Road Map will be dynamically  
updated.

## Game description and representation
### Game description
Betting Game is a two player zero-sum imperfect information game. It can be seen as a simplified model for one street of  
heads up betting in poker. Betting Games can be seen as a general case of 
[**Kuhn Poker**](https://en.wikipedia.org/wiki/Kuhn_poker). Actually Kuhn Poker is exactly  
equivalent to following game:  
```python
J=1; Q=2; K=3
BettingGame(bet_size=0.5, max_number_of_bets=2, deck={J:1, Q:1, K:1}, deal_from_deck_with_substitution=False)
```  
where `J=1, Q=2, K=3` are respectively equivalent to *'Jack'*, *'Queen'* and *'King'* in Kuhn Poker.  
### Betting Game description in conventional poker term:
In this game, similar to other variants of poker, players bet on strength of their hand, if they agree on same amount of  
bet, they go to showdown, which they show their hand and the stronger(higher) hand will win all the money. If one  
player does not agree to increase of bet by other player, he will lose all the bets that he has put in the pot so far in  
that round.    
We divide game procedure into 4 steps(note that players only make decision at step 3, which is betting round).

1. **Posting Blinds(putting forced bets in the put):**  Each player put 1 **'chip'** in the **'pot'**. **chip** is unit of  
   money in the game.    

   
2. **Dealing Hands:** Each player is dealt a **'hand'** from the **'deck'** with specified given **'distribution'**  
   which is usually uniform distribution. Note that deck of Betting Game is totally different from poker, for more  
   information see [Dealing Cards and Deck](#dealing-cards-and-deck)
   

3. **One Round Of Betting:** players go to one round of betting, which is similar to one street of betting in poker  
   , with two important differences:  
   First there is only one bet size allowed. (Note that sizes are in relation to pot size).  
   Second there is a limit on max number of bets allowed. (with bet size being fix, this condition is equivalent to having  
   a limit on the **'stack'** of players). 


4. **Showdown**: player cards(or hands) are just number, at showdown player with higher number will win the whole pot.  
   if both player have the same hand(equal number), they will split the pot equally.

#### Dealing Cards and Deck
**'deck'** in this game is usually a **'hyper set'** of numbers, which can be presented by a dictionary in python:  
`deck = {a_i:x_i for i in range(1, m)}`   
where dictionary keys `a_i, for i=1,2, ...,m` show different possible cards(or hands) in the deck, and each dictionary  
value `x_i` shows how many of card `a_i` we have in the deck.   
Players might draw from the same common deck of cards ,  which is equivalent to  
`BettingGame.deal_from_deck_with_substitution=False`

Each player might have their own deck, but their deck are equal sets, this case is equivalent to:  
BettingGame.deal_from_deck_with_substitution=True  
Or they might have different decks.

Although we usually consider player drawing their hand from decks of the form `{a_i:x_i for i in range(1,m)}` , the  
mathematical definition of BettingGames deck is more general, and it presents both 'deck' and **'dealing cards'** at   
same time using a joint distribution:  
`(op_hand, ip_hand)` is drawn from given arbitrary joint distribution `P`, which is called dealing distribution or dd.  
op_hand = out of position player hand (first player)  
ip_hand = in position player hand (second player)

#### Betting Game contrast with No limit Texas Holdem:
The Betting Game Can be seen as No flop Poker with just one round of betting and one round of cards dealing from its  
specific deck of cards. For more specific contrast we explain differences with most popular version of poker, No Limit  
Texas Holdem or NL for simplicity:  
In NL game has 4 streets, in each street there is some cards or **'community cards'** dealing and after that there is  
one round of betting. In Betting Games, we only have one round of cards dealing from specific deck and just one round of  
betting.  
In NL, in the start of game one player put half chip('Small Blind), and other player put one chip(Big Blind), in Betting  
Game both player start by putting same amount of forced bet into the pot.

### Game Representation
To Be Completed ...

# Efforts To Solve The Betting Game
