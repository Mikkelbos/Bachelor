import numpy as np
import pandas as pd
import mestim as M
from tabulate import tabulate
from numpy import linalg as la
from scipy.optimize import minimize
from numpy import random
import time
from scipy.stats import genextreme
import mestim as M


def s_j(alpha, beta, x_j, p_j):
    return alpha*p_j + x_j*beta 

def exp_delta(alpha, beta, x_j, p_j):
    delta_list = []
    exp_delta = []
    for i in range(len(p_j)):
        delta_list.append(s_j(p_j[i], alpha, beta, x_j[i]))
        
    for i in range (len(delta_list)):
        exp_delta.append(np.exp(delta_list[i]))
    return np.array(exp_delta)

def ccp(alpha, beta, x_j, p_j):
    ccp_list = []
    exp_delta = exp_delta(alpha, beta, x_j, p_j)
    sum_exp = np.sum(exp_delta)
    
    for i in range(len(exp_delta)):
        ccp_list.append(exp_delta[i]/sum_exp)
    return np.array(ccp_list)

def probability_ratio(ccp, index, columns): #index = alternative j, columns = alternative i
    probability_ratio_matrix = pd.DataFrame(index = index, columns = columns)
    for i in range(len(ccp)):
        for j in range(len(ccp)):
            probability_ratio_matrix.iloc[i,j] = ccp[i]/ccp[j]
    return probability_ratio_matrix

def marginal_effects(ccp, index, columns, coefficients):
    marginal_effects = pd.DataFrame(index = index, columns = columns)
    for i in range(len(ccp)):
        for j in range(len(coefficients)-1):
            marginal_effects.iloc[i,j] = coefficients[j+1]*ccp[i]*(1-ccp[j])
    return marginal_effects

def cross_marginal_effects(ccp, index, columns, coefficients):
    cross_marginal_effects = np.zeros((len(ccp), len(ccp), len(coefficients)-1))
    for i in range(len(ccp)):
        for j in range(len(ccp)):
            for k in range(len(coefficients)-1):
                cross_marginal_effects[i,j, k] = -coefficients*ccp[i]*ccp[j]
    return cross_marginal_effects

   
   
   
   
   
   
