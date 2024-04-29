import statsmodels.api as sm
import numpy as np
from numpy import linalg as la
import pandas as pd

def exp_delta(alpha, beta, X, p_j):
    share_j = []
    exp_share_j = []
    for j in range(len(p_j)):
        s = alpha*p_j[j] + X[j:j+1,2:]@beta[2:].reshape(-1,1)
        share_j.append(s)
    print(f'share_j: {len(share_j)}')
        
    for j in range (len(share_j)):
        exp_share_j.append(np.exp(share_j[j]))
    #print(f'exp_share_j: {exp_share_j[:10]}, sum: {np.sum(exp_share_j)}')
    return exp_share_j

def ccp(alpha, beta, X, p_j):
    ccp_list = [] 
    exp_delta_list = exp_delta(alpha, beta, X, p_j)
    sum_exp = np.sum(exp_delta_list)

    for i in range(len(exp_delta_list)):
        ccp_list.append(exp_delta_list[i]/sum_exp) 
    #print(f'choice probability sum: {np.sum(ccp_list)} \n 3 highest probability: {np.sort(ccp_list, axis=0)[-3:]}')
    print(f' choice probability sum: {np.sum(ccp_list)} \n ccp:{ccp_list[:11]}')
    return ccp_list


#Rows = model labels, columns = model labels for NxN matrix
def probability_ratio(ccp, model_labels): #index = alternative j, columns = alternative i
    probability_ratio_matrix = pd.DataFrame(index = model_labels, columns = model_labels)
    for i in range(len(ccp)):
        for j in range(len(ccp)):
            probability_ratio_matrix.iloc[i,j] = ccp[i]/ccp[j]
    #print(f'probability_ratio_matrix: \n{probability_ratio_matrix}')
    return probability_ratio_matrix

def marginal_effects(ccp, model_labels, coefficients_labels, coefficients):
    marginal_effects = pd.DataFrame(index = model_labels, columns = coefficients_labels)
    for i in range(len(ccp)):
        for j in range(len(coefficients)-1):
            marginal_effects.iloc[i,j] = coefficients[j+1]*ccp[i]*(1-ccp[j])
    #print(f'marginal_effects: \n{marginal_effects}')
    return marginal_effects

def cross_marginal_effects(ccp, coefficients):
    cross_marginal_effects = np.zeros((len(ccp), len(ccp), len(coefficients)-1))
    for i in range(len(ccp)):
        for j in range(len(ccp)):
            for k in range(len(coefficients)-1):
                cross_marginal_effects[i,j, k] = -coefficients[k+1]*ccp[i]*ccp[j]
    #print(f'cross_marginal_effects: \n {cross_marginal_effects}')
    print(cross_marginal_effects.shape)
    return cross_marginal_effects

def elasticity(ccp, model_labels, coefficients_labels, coefficients, X):
    elasticity = pd.DataFrame(index = model_labels, columns = coefficients_labels)
    #print(elasticity)
    X = X[:,1:].reshape(1, -1)
  
    for i in range(len(model_labels)):
        for j in range(len(coefficients)-1):
            elasticity.iloc[i,j] = ((coefficients[j+1])*X[:,j:j+1]*(1 - ccp[i]))
    #print(elasticity)
    print(f'elasticity shape: \n{elasticity.shape}')
    return elasticity

def print_cross_elasticity(cross_elasticity, model_labels):
    print(cross_elasticity.shape[1])
    for k in range(cross_elasticity.shape[1]):
        print(f'Change in : {model_labels[k]} \n {cross_elasticity[:,k:k+1,:]}')

def cross_elasticity(ccp, coefficients, X, model_labels):
    cross_elasticity = np.zeros((len(ccp), len(ccp), len(coefficients)-1))
    X = X[:,1:] #remove the constant
    for k in range(len(coefficients)-1):
        for i in range(len(ccp)):
            for j in range(len(ccp)):
                cross_elasticity[i,j, k] = -coefficients[k+1]*X[1,k]*(ccp[j])
    print_cross_elasticity(cross_elasticity, model_labels)
    #return cross_elasticity
    

def est_OLS(y, X, xnames):
    model = sm.OLS(y, X)
    results = model.fit()
    #print(results.summary(xname= xnames, yname='Market share'))
    return results
   
   
def logit(alpha, beta, X, p_j, model_labels, coefficients_labels, coefficients):
    ccp_list = ccp(alpha, beta, X, p_j)
    probability_ratio_matrix = probability_ratio(ccp_list, model_labels, model_labels)
    marginal_effects_matrix = marginal_effects(ccp_list, model_labels, coefficients_labels, coefficients)
    cross_marginal_effects_matrix = cross_marginal_effects(ccp_list, coefficients)
    elasticity_matrix = elasticity(ccp_list, model_labels, coefficients_labels, coefficients, X)
    cross_elasticity_matrix = cross_elasticity(ccp_list, beta, X, model_labels)

    #return ccp_list, probability_ratio_matrix#, marginal_effects_matrix, cross_marginal_effects_matrix, elasticity_matrix, cross_elasticity_matrix

def BLP(dataframe, instrument):
    # Sum the attribute of other models
    def sum_attribute(row):
        # Filter the DataFrame excluding the current model
        data = dataframe[(dataframe['Year'] == row['Year']) & (dataframe['Model'] != row['Model'])]
                
        #Sum
        sum = data[instrument].sum()


        return pd.Series({'sum': sum})

    # Apply the function to each row in the DataFrame
    new_columns = dataframe.apply(sum_attribute, axis=1)

    # Add the new columns to the DataFrame
    dataframe[instrument+'_sum'] = new_columns['sum']

    return dataframe


#Instrument generation
def create_instrument_sum(df1, instrument):
    df1[instrument+'_instrument'] = 0
    for index, row in df1.iterrows():
        current_model = row['Model']
        sum_except_current = df1[df1['Model'] != current_model][instrument].sum()
        df1.at[index, instrument+'_instrument'] = sum_except_current


def create_instrument_localsum(df, instrument, factor):
    df[instrument + '_instrument_localdiff'] = 0
    std_dev = df[instrument].std()
    std_dev = std_dev*factor

    for index, row in df.iterrows():
        current_model_instrument = row[instrument]
        sum_except_current = df[(df['Model'] != row['Model']) & (df['Year'] == row['Year']) &
                    ((df[instrument] > current_model_instrument + std_dev) |
                     (df[instrument] < current_model_instrument - std_dev))][instrument].sum()
        df.at[index, instrument + '_instrument_localdiff'] = sum_except_current #- current_model_instrument hvis det skal være som i 2xHao

    return df

#Manipuler data hvor market share er 0
def straf_0ms(df):
    for i in range(len(df)):
        if df['Market share'][i] == 0:
            df['Price'][i] = 10_000_000
            #df['HP'][i] = -df['HP'][i]
            #df['Chargetime'][i] = -df['Chargetime'][i]

    return df

