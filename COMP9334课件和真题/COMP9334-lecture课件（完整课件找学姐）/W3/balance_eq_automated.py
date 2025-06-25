#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMP9334 Week 3B 

To derive the state balance equations for the database server example 
automatically
"""

import numpy as np 

# Recall that the states are 
# (#jobs at CPU, #jobs at fast disk, #jobs at slow disk)

# Specify all the states as a list of 3-tuples
states = [(2,0,0),
          (1,1,0),
          (1,0,1),
          (0,2,0),
          (0,1,1),
          (0,0,2)]

# Each transition in state can be characterised by a 3-tuple
# For example, if a job finishes at the fast disk and moves to the CPU, then
# it can be represented by the tuple
#           (1,-1,0) 
# where: 
#   The 1 in the first position means a job has moved to the CPU,
#   and 
#   -1 in the second position means a job has left the fast disk
mv_fast_to_cpu = (1,-1,0)
mv_slow_to_cpu = (1,0,-1)
mv_cpu_to_fast = (-1,1,0)
mv_cpu_to_slow = (-1,0,1)

# The corresponding transition rate 
rate_fast_to_cpu = 4
rate_slow_to_cpu = 2
rate_cpu_to_fast = 3
rate_cpu_to_slow = 3

# The number of states 
num_states = len(states)

# Initialise the R matrix which contains the equations for state balance
R = np.zeros((num_states,num_states))

# Loop through all the state pairs to find the elements of the R matrix
for i in range(num_states):         # i is the row index 
    for j in range(num_states):     # j is the column index 
        if i != j:
            # The states 
            state_i = states[i]
            state_j = states[j]
            
            # Determine the transition by subtracting 
            # state_j from state_i 
            change = tuple(np.subtract(state_i, state_j))
            
            # Determine the transition
            if change == mv_fast_to_cpu:
                R[i,j] += rate_fast_to_cpu
                R[j,j] -= rate_fast_to_cpu
            elif change == mv_slow_to_cpu:
                R[i,j] += rate_slow_to_cpu
                R[j,j] -= rate_slow_to_cpu
            elif change == mv_cpu_to_fast:
                R[i,j] += rate_cpu_to_fast
                R[j,j] -= rate_cpu_to_fast
            elif change == mv_cpu_to_slow:
                R[i,j] += rate_cpu_to_slow
                R[j,j] -= rate_cpu_to_slow
            # All other changes are invalid 
            
            
print('The state balance equation matrix \n',R)                