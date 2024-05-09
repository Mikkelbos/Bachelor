import statsmodels.api as sm
import numpy as np
from numpy import linalg as la
import pandas as pd

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


def probability_ratio(dataset, year):
    
    year_data = dataset[dataset['Year'] == year]

    # DataFrame for the specific year
    ccp_values = year_data['CCP']
    model_labels = year_data['Model']
    probability_ratio_matrix = pd.DataFrame(index=model_labels, columns=model_labels)

    # Calculate the probability ratio
    for i in range(len(ccp_values)):
        for j in range(len(ccp_values)):
            probability_ratio_matrix.iloc[i, j] = ccp_values.iloc[i] / ccp_values.iloc[j]

    return probability_ratio_matrix


def marginal_effects(dataset, estimation):
    ccp = dataset['CCP']
    model_labels = dataset['Model_year']
    
    coefficients_labels = estimation.params.index
    coefficients = estimation.params
    marginal_effects = pd.DataFrame(index = model_labels, columns = coefficients_labels)
    
    for i in range(len(ccp)):
        for j in range(len(coefficients)):
            marginal_effects.iloc[i,j] = coefficients[j]*ccp[i]*(1-ccp[i]) #dv/dz*P_i*(1-P_i)
    
    return marginal_effects

def cross_marginal_effects(dataset, estimation):
    
    ccp = dataset['CCP']
    model_labels = dataset['Model_year']
    model = dataset['Model']
    
    coefficients_labels = estimation.params.index 
    coefficients = estimation.params
    
    marginal_effects_dict = {}
    
    for k, coef_label in enumerate(coefficients_labels):
        marginal_effects = np.zeros((len(ccp), len(ccp)))
        for i in range(len(ccp)):
            for j in range(len(ccp)):
                marginal_effects[i, j] = -coefficients[k] * ccp[i] * ccp[j]
                
        marginal_effects_flat = marginal_effects.flatten()
        
        marginal_effects_dict[coef_label] = marginal_effects_flat
    
    marginal_effects_df = pd.DataFrame(marginal_effects_dict, index=pd.MultiIndex.from_product([model, model_labels]))
    
    return marginal_effects_df


def elasticity(dataset, estimation):
    ccp = dataset['CCP']
    model_labels = dataset['Model_year']
    X = dataset[estimation.params.index]
    
    coefficients_labels = estimation.params.index
    coefficients = estimation.params
    elasticity = pd.DataFrame(index = model_labels, columns = coefficients_labels)
    
    for i in range(len(ccp)):
        for j in range(len(coefficients)):
            elasticity.iloc[i,j] = ((coefficients[j])*X.iloc[i,j]*(1 - ccp[i]))
    
    return elasticity

#def print_cross_elasticity(cross_elasticity, model_labels):
    #print(cross_elasticity.shape)
    #for k in range(cross_elasticity.shape[1]):
    #    print(f'Change in : {model_labels[k]} \n {cross_elasticity[:,k:k+1,:]}')

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
    dataframe[instrument+'_BLP'] = new_columns['sum']

    return dataframe


def GH(df, instrument, factor): #create instrument local sum
    df[instrument + '_GH'] = 0
    std_dev = df[instrument].std()
    std_dev = std_dev*factor

    for index, row in df.iterrows():
        current_model_instrument = row[instrument]
        sum_except_current = df[(df['Model'] != row['Model']) & (df['Year'] == row['Year']) &
                    ((df[instrument] > current_model_instrument + std_dev) |
                     (df[instrument] < current_model_instrument - std_dev))][instrument].sum()
        df.at[index, instrument + '_GH'] = sum_except_current 

    return df

'''
#Manipuler data hvor market share er 0
def straf_0ms(df):
    for i in range(len(df)):
        if df['Market share'][i] == 0:
            df['Price'][i] = 10_000_000
            #df['HP'][i] = -df['HP'][i]
            #df['Chargetime'][i] = -df['Chargetime'][i]

    return df
'''



def cost_original(dataset, alpha):
    s = dataset['CCP']
    p_j = dataset['Price']
    cost = np.zeros(len(p_j))
    for i in range(len(p_j)):
        cost[i] = p_j[i] + (alpha/s[i])
        #Hvis cost stadig er højere end pris, så ændre "-" til "+" og køre igen
    return cost

#Cost

def cost(p_j, ccp, alpha):
    cost = np.zeros(len(p_j))
    for i in range(len(p_j)):
        #cost[i] = p_j[i] + (s[i]/alpha) 
        cost[i] = p_j[i] + (ccp[i]/(alpha*ccp[i]*(1-ccp[i]))) #alpha(-)*ccp(+)*(1-ccp)(+) = noget negativt
    return cost
