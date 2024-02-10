import numpy as np
import pandas as pd
import mestim as M
from tabulate import tabulate
from numpy import linalg as la
from scipy.optimize import minimize
from numpy import random
import time
from scipy.stats import genextreme

def estimation(Qfun, theta0, deriv=0, cov_type ='sandwich', parnames='', output=False): 

    tic = time.perf_counter() 

    # Q: Sample objective function to minimize (e.g. sample average of negative log-livelihood)
    Q     = lambda theta:  Qfun(theta, out='Q')

    # dQ: Derivative of sample objective function wrt parameters theta (function returns size K array)
    dQ = None
    if deriv>0: # use user-supplied 1 order derivatives
        dQ = lambda theta:  Qfun(theta, out='dQ')

    hess = None
    if deriv>1: # use user-supplied 2 order derivatives
        hess  =  lambda theta:  Qfun(theta, out='H')
        res=minimize(fun=Q, jac=dQ, hess=hess, x0=theta0, method='trust-ncg')
        res.hess_inv=la.inv(res.hess)
    else: # use bfgs
        res=minimize(fun=Q, jac=dQ, x0=theta0, method='bfgs')
    
    theta_hat=np.array(res.x).reshape(-1,1)
    
    toc = time.perf_counter()

    # variance co-variance matrix
    s_i=Qfun(theta_hat, out='s_i')

    cov = avar(s_i, res.hess_inv, cov_type)
    se = np.sqrt(np.diag(cov)).reshape(-1, 1)

    # collect output
    names =   ['parnames', 'theta_hat', 'se',  't-values',  'cov', 'Q', 'time', 's_i']
    results = [parnames, theta_hat, se, theta_hat/se, cov, Q(theta_hat), toc - tic, s_i]

    res.update(dict(zip(names, results)))

    if output: 
        if res.parnames: 
            table=({k:res[k] for k in ['parnames', 'theta_hat', 'se', 't-values', 'jac']})
        else: 
            table=({k:res[k] for k in ['theta_hat', 'se', 't-values', 'jac']})
        print(tabulate(table, headers="keys",floatfmt="10.5f"))
        print('')
        print(res.message)
        print('Objective function:', res['Q'])
        print ('Iteration info: %d iterations, %d evaluations of objective, and %d evaluations of gradients' 
            % (res.nit,res.nfev, res.njev))
        print(f"Elapsed time: {res['time']:0.4f} seconds")

    return res

def avar(s_i, Ainv, cov_type ='sandwich'):
    n, K=s_i.shape
    B=s_i.T@ s_i/n 
    if cov_type=='Ainv':        return Ainv/n;         
    if cov_type=='Binv':        return la.inv(B)/n
    if cov_type=='sandwich':    return Ainv @ B @ Ainv/n


def sim_data(N, J, theta) -> tuple:
    k = theta.size
    
    x = random.normal(size=(N, J, k)) + np.linspace(3,5,J).reshape(1,J, 1)
    v = utiliy(theta, x)
    e = genextreme.ppf(random.uniform(size=(N, J)), c=0)
    u = v + e # utility
    
    # Find which choice that maximizes utility.
    y = u.argmax(axis=1)
    
    label = ['y', 'x']
    d=dict(zip(label, [y, x]))
    return d

def clogit(y, x, cov_type='Ainv',theta0=None, deriv=0, quiet=False): 
	# Objective function and derivatives for 
    N, J, K, palt, xalt, xvars = labels(x)
    Qfun     = lambda theta, out:  Q_clogit(theta, y, x, out)

    if theta0 is None: 
    	theta0=np.zeros((K,1)) 

    res = estimation(Qfun, theta0, deriv, cov_type, parnames=xvars)
    # v, p, dv = Qfun(res.theta_hat, out='predict')
    res.update(dict(zip(['yvar', 'xvars', 'N','K', 'n'], ['y', xvars, N, K, N])))

    if quiet==False:
        print('Conditional logit')
        print('Initial log-likelihood', -Qfun(theta0, 'Q'))
        print('Initial gradient\n', -Qfun(theta0, 'dQ'))
        print_output(res)

    return res

def Q_clogit(theta, y, x, out='Q'):
    v = utiliy(theta, x) 	# Deterministic component of utility
    ll_i=logccp(v, y)
    q_i= - ll_i   
    if out=='Q':
    	return np.mean(q_i)
    dv=x
    p=ccp(v)
    if out=='predict':  return v, p, dv         # Return predicted values
    N, J, K = dv.shape
    idx=y[:,] + J*np.arange(0, N)
    dvj=dv.reshape(N*J, K)[idx,:] 	# pick choice specific values corresponding to y 

    s_i = dvj -  np.sum(p.reshape(N,J,1)*dv, axis=1)
    g = -np.mean(s_i, axis=0)
    
    if out=='s_i': return s_i                     # Return s_i: NxK array with scores
    if out=='dQ':  return g;  # Return dQ: array of size K derivative of sample objective function

def utiliy(theta, x):
	N, J, K=x.shape
	u = x @ theta
	return u.reshape(N,J) 

def logsum(v, sigma=1): 
	# Expected max over iid extreme value shocks with scale parameter sigma
	# Logsum is reentered around maximum to obtain numerical stability (avoids overflow, but accepts underflow)
	max_v = v.max(axis=1).reshape(-1, 1)
	return max_v + sigma*np.log(np.sum(np.exp((v-max_v)/sigma), 1)).reshape(-1, 1)
	
def logccp(v, y=None, sigma=1):
    # Log of conditional choice probabilities 
    # If y = None return logccp corresponding to all choices
    # if y is Nx1 vector of choice indexes, return likelihood
    ev = logsum(v, sigma) 	# Expected utility (always larger than V)
    if y is not None:  		
    	N, J=v.shape
    	idx=y[:,] + J*np.arange(0, N)
    	v=v.reshape(N*J, 1)[idx] 	# pick choice specific values corresponding to y 
    return (v - ev)/sigma

def ccp(v, y=None, sigma=1):
    # Conditional choice probabilities
    return np.exp(logccp(v, y, sigma))

def labels(x):
	# labels and dimensions for plotting
	N, J, K = x.shape
	palt=['p' + str(i)  for i in range(J)]; 
	xalt=['alt' + str(i)  for i in range(J)]; 
	xvars=['var' + str(i)  for i in range(K)]; 
	return N, J, K, palt, xalt, xvars

def APE_var(theta, x, m=0, quiet=False):
	# matrix of partial derivatives with respect of a change in attribute m
    N, J, K, palt, xalt, xvars = labels(x)
    p=ccp(utiliy(theta, x))
    E=np.empty((J,J))
    for j in range(J):
        for k in range(J):
            E[k, j]=np.mean(p[:,j]*theta[m]*(1*(j==k)-p[:,k]), axis=0)
    if not quiet: 
        print('\nAPE wrt change in', xvars[m])
        print(tabulate(np.c_[xalt, E], headers=palt,floatfmt="10.5f"))
    return E

def Ematrix_var(theta, x, m=0, quiet=False):
	# matrix of elasticities with respect ot a change in attribute m
	N, J, K, palt, xalt, xvars = labels(x)
	p=ccp(utiliy(theta, x))
	E=np.empty((J,J))
	for j in range(J):
	    for k in range(J):
	        E[k, j]=np.mean(x[:,k,m]*theta[m]*(1*(j==k)-p[:,k]), axis=0)
	if not quiet: 
	    print('\nElasticity wrt change in', xvars[m])
	    print(tabulate(np.c_[xalt, E], headers=palt,floatfmt="10.5f"))
	return E

def Ematrix_own(theta, x, quiet=False):
    # Own elasticity: % change in prob of alternative j wrt % change in attribute of same alternative j
    # done for for each variable in x  
    N, J, K, palt, xalt, xvars = labels(x)
    p=ccp(utiliy(theta, x))
    E_own=np.empty((J, K))
    for iJ in range(J):
        for iK in range(K):
            E_own[iJ, iK]=np.mean(x[:,iJ,iK]*theta[iK]*(1-p[:,iJ]), axis=0)
    if not quiet: 
        print('\nOwn elasticity')
        print(tabulate(np.c_[xalt, E_own], headers=xvars,floatfmt="10.5f"))
    return E_own

def Ematrix_cross(theta, x, quiet=False):
    # Cross elasticity:  % change in prob of alternative j wrt % change in attribute of other alternative k ne j
    # done for each variable in x  
    N, J, K, palt, xalt, xvars = labels(x)
    p=ccp(utiliy(theta, x))
    E_cross=np.empty((J, K))
    for iJ in range(J):
        for iK in range(K):
            E_cross[iJ, iK]=np.mean(x[:,iJ,iK]*theta[iK]*(-p[:,iJ]), axis=0)
    if not quiet: 
        print('\nCross-Elasticity')
        print(tabulate(np.c_[xalt, E_cross], headers=xvars,floatfmt="10.5f"))

def print_output(res, cols=['parnames','theta_hat', 'se', 't-values', 'jac']): 
    print('Dep. var. :', res['yvar'], '\n') 

    table=({k:res[k] for k in cols})
    print(tabulate(table, headers="keys",floatfmt="10.5f"))
    # print('\n# of groups:      :', res['n'])
    print('# of observations :', res['N'])
    print('# log-likelihood. :', - res['Q']*res['n'], '\n')
    print ('Iteration info: %d iterations, %d evaluations of objective, and %d evaluations of gradients' 
        % (res.nit,res.nfev, res.njev))
    print(f"Elapsed time: {res['time']:0.4f} seconds")
    print('')