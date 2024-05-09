import statsmodels.api as sm
import numpy as np
from numpy import linalg as la
import pandas as pd


'''
def exp_delta(alpha, beta, X, p_j):
    share_j = []
    exp_share_j = []
    for j in range(len(p_j)):
        s = alpha*p_j[j] + X[j:j+1,2:]@beta[2:].reshape(-1,1) #Slicer hhv konstant og pris væk
        share_j.append(s)
    print(f'share_j: {len(share_j)}')
        
    for j in range (len(share_j)):
        exp_share_j.append(np.exp(share_j[j]))
    #print(f'exp_share_j: {exp_share_j[:10]}, sum: {np.sum(exp_share_j)}') #Sanity check
    return exp_share_j

def ccp(alpha, beta, X, p_j):
    ccp_list = [] 
    exp_delta_list = exp_delta(alpha, beta, X, p_j)
    sum_exp = np.sum(exp_delta_list)

    for i in range(len(exp_delta_list)):
        ccp_list.append(exp_delta_list[i]/sum_exp) 
    print(f' choice probability sum: {np.sum(ccp_list)} \n ccp:{ccp_list[:11]}')
    return ccp_list
'''

def ccp(alpha, beta, dataset, X):
    ccp_list = []  # Initialize a list to store CCP arrays for each year

    # Group the dataset by year
    grouped_data = dataset.groupby('Year')

    for year, data_year in grouped_data:
        X_year = data_year[X.columns]
        p_j_year = data_year['Price'].values
        
        # Utility
        utility_year = alpha * p_j_year + np.dot(X_year, beta)

        # CCP
        ccp_year = np.exp(utility_year) / np.sum(np.exp(utility_year))

        # Reshape the CCPs to a column vector
        ccp_reshaped = ccp_year.reshape(-1, 1)

        # Append
        ccp_list.append(ccp_reshaped)

    # Stack the CCP arrays vertically
    ccp_array = np.vstack(ccp_list)

    return ccp_array



#Rows = model labels, columns = model labels for NxN matrix
def probability_ratio(dataset): #index = alternative j, columns = alternative i
    ccp = dataset['CCP']
    model_labels = dataset['Model_year']
    probability_ratio_matrix = pd.DataFrame(index = model_labels, columns = model_labels)
    for i in range(len(ccp)):
        for j in range(len(ccp)):
            probability_ratio_matrix.iloc[i,j] = ccp[i]/ccp[j]
    #print(f'probability_ratio_matrix: \n{probability_ratio_matrix}')
    return probability_ratio_matrix

def marginal_effects(dataset, estimation): #,coefficients_labels, coefficients):
    ccp = dataset['CCP']
    model_labels = dataset['Model_year']
    
    coefficients_labels = estimation.params.index
    coefficients = estimation.params
    marginal_effects = pd.DataFrame(index = model_labels, columns = coefficients_labels)
    
    for i in range(len(ccp)):
        for j in range(len(coefficients)):
            marginal_effects.iloc[i,j] = coefficients[j]*ccp[i]*(1-ccp[i]) #dv/dz*P_i*(1-P_i)
    
    #print(f'marginal_effects: \n{marginal_effects}')
    
    return marginal_effects

def cross_marginal_effects(dataset, estimation):
    
    ccp = dataset['CCP']
    coefficients = estimation.params
    
    cross_marginal_effects = np.zeros((len(ccp), len(ccp), len(coefficients)-1))
    for i in range(len(ccp)):
        for j in range(len(ccp)):
            for k in range(len(coefficients)):
                cross_marginal_effects[i,j, k] = -coefficients[k]*ccp[i]*ccp[j] #-dv/dz*P_i*P_j
    #print(f'cross_marginal_effects: \n {cross_marginal_effects}')
    print(cross_marginal_effects.shape)
    return cross_marginal_effects

def elasticity(ccp, model_labels, coefficients_labels, coefficients, X):
    elasticity = pd.DataFrame(index = model_labels, columns = coefficients_labels)
    #print(elasticity)
    #X = np.array(X)
    #X = X.reshape(1, -1)
  
    for i in range(len(model_labels)):
        for j in range(len(coefficients)):
            elasticity.iloc[i,j] = ((coefficients[j])*X[:,j:j+1]*(1 - ccp[i])) # < 0 * 1 * 0<ccp<1 = 
            #elasticity.iloc[i,j] = ((coefficients[j])*X[j:j+1]*(1 - ccp[i])) # < 0 * 1 * 0<ccp<1 = 
    print(f'elasticity shape: \n{elasticity.shape}')
    print(elasticity)
    return elasticity

def print_cross_elasticity(cross_elasticity, model_labels):
    print(cross_elasticity.shape)
    for k in range(cross_elasticity.shape[1]):
        print(f'Change in : {model_labels[k]} \n {cross_elasticity[:,k:k+1,:]}')

#def cross_elasticity(dataset, coefficients, X, model_labels):
'''def cross_elasticity(dataset, estimation):
    
    ccp = dataset['CCP']
    coefficients = estimation.params
    X = dataset[estimation.index]
    model_labels = dataset['Model_year']
    
    #cross_elasticity = np.zeros((len(ccp), len(ccp), len(coefficients))) #række, colonne, højde
        
    for k in range(len(coefficients)):
        for i in range(len(ccp)):
            for j in range(len(ccp)):
                cross_elasticity_table[i,j, k] = -coefficients[k]*X[i,k]*(ccp[j]) 
                
    print_cross_elasticity(cross_elasticity, model_labels)
    #return cross_elasticity'''
    
def cross_elasticity(dataset, estimation):
    
    ccp = dataset['CCP']
    coefficients = estimation.params
    X = dataset[estimation.params.index]
    model_labels = dataset['Model_year']
    
    cross_elasticity_table = pd.MultiIndex.from_product([model_labels, model_labels, X])
        
    for k in range(len(coefficients)):
        for i in range(len(ccp)):
            for j in range(len(ccp)):
                cross_elasticity_table[i, j, k] = -coefficients[k] * X.iloc[i, k] * ccp.iloc[j]
                
    #print_cross_elasticity(cross_elasticity_table, model_labels)
    return cross_elasticity_table




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
    df[instrument + '_instrument_localsum'] = 0
    std_dev = df[instrument].std()
    std_dev = std_dev*factor

    for index, row in df.iterrows():
        current_model_instrument = row[instrument]
        sum_except_current = df[(df['Model'] != row['Model']) & (df['Year'] == row['Year']) &
                    ((df[instrument] > current_model_instrument + std_dev) |
                     (df[instrument] < current_model_instrument - std_dev))][instrument].sum()
        df.at[index, instrument + '_instrument_localsum'] = sum_except_current 

    return df

#Manipuler data hvor market share er 0
def straf_0ms(df):
    for i in range(len(df)):
        if df['Market share'][i] == 0:
            df['Price'][i] = 10_000_000
            #df['HP'][i] = -df['HP'][i]
            #df['Chargetime'][i] = -df['Chargetime'][i]

    return df



#Cost

def cost(p_j, ccp, alpha):
    cost = np.zeros(len(p_j))
    for i in range(len(p_j)):
        #cost[i] = p_j[i] + (s[i]/alpha) 
        cost[i] = p_j[i] + (ccp[i]/(alpha*ccp[i]*(1-ccp[i]))) #alpha(-)*ccp(+)*(1-ccp)(+) = noget negativt
    return cost