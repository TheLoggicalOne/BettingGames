
""" Can Each Strat just be a list of probs?
 Strat as list of  willingness probs of coming to public node k. Note that each public node k  has only one parent
 which is accessible by G.public_state[k]
 Strat List could be initialized using tree or using or public_node_tree.PublicTree.children_of_nodes or...?
 Anyway after initialization the can be updated using tree.
 Is there a way of updating the start list by matrices and in numpy?
 PublicTree.Tree.ActionIDs
 Best Rep Possible?:
 let d=deck size. Let S be a k*k*x np.array where A=S[i,j,:] shows the state where op_hand=i and ip_hand=j
 Now let A be 1-d array of length that A[k] is the willingness probs of coming to public node k.
 On the other hand if we are at node r with hand i and we want to know chance of playing given action a according to
 our strat, it is S[i,j, G.PublicNodeAsTuple[r].children[a]] """
import numpy as np
import copy