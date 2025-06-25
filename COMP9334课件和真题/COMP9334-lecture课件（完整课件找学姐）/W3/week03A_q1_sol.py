#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMP9334, Revision Problems Week 3A, Question 1

Solution 

Chun Tung Chou, CSE, UNSW
"""

# import 
import numpy as np
import matplotlib.pyplot as plt 

# ## Functions
def cal_MM1_resp_time(lamb,mu):
    # This function calculates the response time T of an M/M/1 
    # queue with 
    #   lamb = mean arrival rate  
    #   mu = mean service rate 
    
    return 1 / (mu-lamb)



def cal_MMM_resp_time(lamb,mu,m):
    # This function calculates the response time T of an M/M/m 
    # queue with 
    #   lamb = mean arrival rate  
    #   mu = mean service rate 
    #   m = number of servers
   
    # utilisation
    rho = lamb/mu/m
    
    # form a list with (m rho)^k / k! for k = 0,1,...,m
    x = [1]        # k = 0 
    for k in range(1,m+1):
        x.append(x[-1]*m*rho/k) # next element = previous element * m * rho / k 
    
    # The waiting time expression
    C_num = x[-1]  # Last element in x 
    C_den = (1-rho)*sum(x[:-1]) + x[-1]
    C = C_num/C_den
        
    return (1/mu)*(1+ C/m/(1-rho))


# %% Loop through different arrival rates 

# Use a range of arrival rate 
#   Starting from 0.1 and ends at 9
lambda_start = 0.1
lambda_stop = 9.0
lambda_step = 0.1
array_lambda = np.arange(lambda_start,lambda_stop+lambda_step/2,lambda_step)
len_array_lambda = len(array_lambda)

# The service rate 
mu = 1/0.1

# Create zero arrays to store the results of response time calculation
T_mm1 = np.zeros((len_array_lambda,))
T_CPU2times = np.zeros((len_array_lambda,)) 
T_2queues = np.zeros((len_array_lambda,))
T_1queue = np.zeros((len_array_lambda,))

# For each arrival rate, calculate the response time 
for i in range(len_array_lambda):
    # Arrival rate 
    lamb = array_lambda[i]
    
    # M/M/1    
    T_mm1[i] = cal_MM1_resp_time(lamb,mu)
    #
    # Alternative 1 - a server 2 times faster
    T_CPU2times[i] = cal_MM1_resp_time(lamb,2*mu)
    #
    # Alternative 2 - Two servers, each with a queue
    T_2queues[i] = cal_MM1_resp_time(lamb/2,mu)
    #
    # Alternative 3 - Two servers, 1 queue
    T_1queue[i] = cal_MMM_resp_time(lamb,mu,2)

# Plot the results
plt.plot(array_lambda,T_mm1,label='original system')
plt.plot(array_lambda,T_CPU2times,label='CPU - 2 times faster') 
plt.plot(array_lambda,T_2queues,label='Two servers, 2 queues')
plt.plot(array_lambda,T_1queue,label='Two servers, 1 queue')
plt.legend()
plt.grid()
plt.xlabel('Arrival rate')
plt.ylabel('Response time')
plt.savefig('prob_q1.png')