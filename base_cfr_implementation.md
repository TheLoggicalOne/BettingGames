## What we have so far:
### strategy:  
`S[position,hand,node]`:  
`S[0, op_hand, node]` and `S[1, ip_hand, node]`

### world state reach probabilities:
`R[op_hand, ip_hand, node]` for any combination of op, ip, chance

### world state values:
`V[op_hand, ip_hand, node]`

## What else we need:
Cumulative counterfactual regret at each decision node,for each action  
in that Tnode for to_move position.  
`cumulative_cf_regret[info_state, action]`

### How to compute cumulative_cfr?
- compute cf_regret at each iteration:  
`cf_regret[info_state, action] =cf_value[info_state&action]-cf_value[info_state]`


- compute cf_value of info_state:  
`cf_value[info_state]=sum(p(oppo_hand)*V[{player_hand, oppo_hand}, info_state.ndoe] for all oppo hand)`

- or compute cf_regret directly:  
`cf_regret[info_state, action]= sum over following for every possible oppo_hand:`  
`p(oppo_hand)*(V[{player_hand, oppo_hand}, info_state.ndoe]-V[{player_hand, oppo_hand}, info_state&action])`

let `cfv=cf_value_world_nodes_table(op)`  
each op info state=(op_hand, node) required info are in `cfv[op_hand, :, node]`  
op just doesn't know which ip hand he is facing, but the numbers are ip reach  
prob adjusted, meaning they are  
`V=values_of_world_nodes_table()[op_hand, :, node]`
multiplied by:  
`R=cf_reach_probs_of_world_nodes_table(op)[op_hand,:,node]`  

**So we have**:  
`cf_regret[op_hand, node, child] = cfv[op_hand, :, child]-cfv[op_hand, :, node]`  
`= (V[op_hand, :, child]-V[op_hand, :, node])*R[op_hand,:,node]  `

**Can we Also say**:
`cf_regret[op_hand, node, child] = cfv[op_hand, :, child]-cfv[op_hand, :, node]`  
`= (V[op_hand, :, child]-V[op_hand, :, node])*R[op_hand,:,node]  `