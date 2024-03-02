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


#def s_j(alpha, beta, x_j, p_j):
#    return alpha*p_j + x_j*beta 

def exp_delta(alpha, beta, X, p_j):
    s_j = []
    exp_s_j = []
    for i in range(len(p_j)):
        s = alpha*p_j[i] + X[i:i+1,2:]@beta[2:].reshape(-1,1)
        s_j.append(s)
        
    for i in range (len(s_j)):
        exp_s_j.append(np.exp(s_j[i]))
    return np.array(exp_s_j)

def ccp(alpha, beta, x_j, p_j):
    ccp_list = [] 
    exp_delta_list = exp_delta(alpha, beta, x_j, p_j)
    sum_exp = np.sum(exp_delta_list)
    
    for i in range(len(exp_delta_list)):
        ccp_list.append(exp_delta_list[i]/sum_exp) 
    print(np.sum(ccp_list))
    return np.array(ccp_list)



def probability_ratio(ccp, index, columns): #index = alternative j, columns = alternative i
    probability_ratio_matrix = pd.DataFrame(index = index, columns = columns)
    for i in range(len(ccp)):
        for j in range(len(ccp)):
            probability_ratio_matrix.iloc[i,j] = ccp[i]/ccp[j]
    #print(f'probability_ratio_matrix: \n{probability_ratio_matrix}')
    return probability_ratio_matrix

def marginal_effects(ccp, index, columns, coefficients):
    marginal_effects = pd.DataFrame(index = index, columns = columns)
    for i in range(len(ccp)):
        for j in range(len(coefficients)-1):
            marginal_effects.iloc[i,j] = coefficients[j+1]*ccp[i]*(1-ccp[j])
    #print(f'marginal_effects: \n{marginal_effects}')
    return marginal_effects

def cross_marginal_effects(ccp, index, columns, coefficients):
    cross_marginal_effects = np.zeros((len(ccp), len(ccp), len(coefficients)-1))
    for i in range(len(ccp)):
        for j in range(len(ccp)):
            for k in range(len(coefficients)-1):
                cross_marginal_effects[i,j, k] = -coefficients[k+1]*ccp[i]*ccp[j]

    return cross_marginal_effects

def elasticity (ccp, index, columns, coefficients, X):
    elasticity = pd.DataFrame(index = index, columns = columns)
    X = X[:,1:]
    for i in range(len(ccp)):
        for j in range(len(coefficients)-1):
            elasticity.iloc[i,j] = ((coefficients[j+1])*X[i]*(1 - ccp[i]))
    print(elasticity)
    return elasticity

def cross_elasticity(ccp, index, columns, coefficients, X):
    cross_elasticity = np.zeros((len(ccp), len(ccp), len(coefficients)-1))
    X = X[:,1:] #remove the constant
    for i in range(len(ccp)):
        for j in range(len(ccp)):
            for k in range(len(coefficients)-1):
                cross_elasticity[i,j, k] = -coefficients[k+1]*X[i]*(ccp[j])
    print(cross_elasticity)
    return cross_elasticity
    
   
   
   
   
   
