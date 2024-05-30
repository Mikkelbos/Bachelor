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

def utility(alpha, beta, dataset, X):
    utility_list = []  # Initialize a list to store CCP arrays for each year

    # Group the dataset by year
    grouped_data = dataset.groupby('Year')

    for year, data_year in grouped_data:
        X_year = data_year[X.columns]
        p_j_year = data_year['Price'].values
        
        # Utility
        utility_year = alpha * p_j_year + np.dot(X_year, beta)

        # Reshape the CCPs to a column vector
        utility_reshaped = utility_year.reshape(-1, 1)

        # Append
        utility_list.append(utility_reshaped)

    # Stack the CCP arrays vertically
    ccp_array = np.vstack(utility_list)

    return ccp_array

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
    
    year_data = dataset[dataset['Year'] == year].reset_index(drop=True)

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
   
    dataset = dataset.reset_index(drop=True)
    ccp = dataset['CCP']
    model_labels = dataset['Model_year']
    model = dataset['Model']
    year = dataset['Year'] 
    
    coefficients_labels = estimation.params.index 
    coefficients = estimation.params
    
    marginal_effects_dict = {}
    
    for k, coef_label in enumerate(coefficients_labels):
        marginal_effects = np.zeros((len(ccp), len(ccp)))
        for i in range(len(ccp)):
            for j in range(len(ccp)):
                # Check if the observations are in the same year
                if year[i] == year[j]:
                    marginal_effects[i, j] = -coefficients[k] * ccp[i] * ccp[j]
                else:
                    marginal_effects[i, j] = np.nan  # Set to NaN if not in the same year
                    
        marginal_effects_flat = marginal_effects.flatten()
        
        marginal_effects_dict[coef_label] = marginal_effects_flat
    
    marginal_effects_df = pd.DataFrame(marginal_effects_dict, index=pd.MultiIndex.from_product([model, model_labels]))
    
    # Drop NaN values from the DataFrame
    marginal_effects_df = marginal_effects_df.dropna()
    
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

'''def print_cross_elasticity(cross_elasticity, model_labels):
    #print(cross_elasticity.shape)
    #for k in range(cross_elasticity.shape[1]):
    #    print(f'Change in : {model_labels[k]} \n {cross_elasticity[:,k:k+1,:]}')
'''
def cross_elasticity(dataset, estimation, parameter):
    models = dataset['Model_year']
    cross_elasticity_table = pd.DataFrame(index=models, columns=models)
    ccp = dataset['CCP']
    coefficients = estimation.params[parameter]
    X = dataset[parameter]

    for i in range(len(ccp)):
        for j in range(len(ccp)):
            cross_elasticity_value = -coefficients * X[j] * ccp[j]
            cross_elasticity_table.iloc[i, j] = round(cross_elasticity_value, 5)

    return cross_elasticity_table
            
    #print_cross_elasticity(cross_elasticity, model_labels)
    return cross_elasticity
    
'''def cross_elasticity(dataset, estimation):
    
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
    return cross_elasticity_table'''

def cross_elasticity_1(dataset, estimation):
    
    ccp = dataset['CCP']
    coefficients = estimation.params
    X = dataset[estimation.params.index]
    model_labels = dataset['Model_year']
    
    cross_elasticity_table = pd.DataFrame(index=pd.MultiIndex.from_product([model_labels, model_labels, X.columns]),
                                           columns=['Cross_Elasticity'])
        
    for k in range(len(coefficients)):
        #print(f'Current coefficient: {estimation.params.index[k]}')
        for i in range(len(ccp)):
            for j in range(len(ccp)):
                cross_elasticity_table.loc[(model_labels[i], model_labels[j], X.columns[k]), 'Cross_Elasticity'] = -coefficients[k] * X.iloc[j, k] * ccp.iloc[j] 
                
    return cross_elasticity_table

# Example usage:
# cross_elasticity_table = cross_elasticity(logit_data, OLS)
# print(cross_elasticity_table)

   

def BLP(dataset, instrument):
    # Sum the attribute of other models
    def sum_attribute(row):
        data = dataset[(dataset['Year'] == row['Year']) & (dataset['Model'] != row['Model'])]
                
        #Sum
        sum = data[instrument].sum()


        return pd.Series({'sum': sum})

    new_columns = dataset.apply(sum_attribute, axis=1)

    dataset[instrument+'_BLP'] = new_columns['sum']

    return dataset


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

#Treat each models as a firm
def cost(dataset, alpha):
    ccp = dataset['CCP']
    p_j = dataset['Price']
    cost = np.zeros(len(p_j))
    for i in range(len(p_j)):
        cost[i] = p_j[i] + (ccp[i]/alpha)
    return cost



#Costs for firms

def cost_firm(dataset, alpha):
    dataset['firm_cost'] = 0
    
    for firm in dataset['Manufacturer'].unique():
        for year in dataset['Year'].unique():
            dataset_firm_year = dataset[(dataset['Manufacturer'] == firm) & (dataset['Year'] == year)]
            
            total_share = dataset_firm_year['CCP'].sum()
            
            for car in dataset_firm_year.index:
                dataset.at[car, 'firm_cost'] = dataset.at[car, 'Price'] + (total_share / alpha)
                
    return dataset


def cost_OLD(dataset, alpha):
    ccp = dataset['CCP']
    p_j = dataset['Price']
    cost_OLD = np.zeros(len(p_j))
    for i in range(len(p_j)):
        cost_OLD[i] = p_j[i] + (ccp[i]/(alpha*ccp[i]*(1-ccp[i])))
    return cost_OLD

def markup(data):
    #data['markup%'] = (data['Price'] - data['firm_cost_OLD'])/data['firm_cosst_OLD']*100 
    data['markup%'] = (data['Price'] - data['firm_cost'])/data['firm_cost']*100
    return data

def price_est(data, alpha):
    ccp = data['CCP']
    
    for i in range(len(data)):
        data['Price_est'][i] = data['cost_firm'][i] + (ccp[i]/(alpha*ccp[i]*(1-ccp[i])))
    return data




def cost_firm_OLD(dataset, alpha):
    dataset['firm_cost_OLD'] = 0
    
    for firm in dataset['Manufacturer'].unique():
        for year in dataset['Year'].unique():
            dataset_firm_year = dataset[(dataset['Manufacturer'] == firm) & (dataset['Year'] == year)]
            
            ccp = dataset_firm_year['CCP'].sum()
            
            for car in dataset_firm_year.index:
                dataset.at[car, 'firm_cost_OLD'] = dataset.at[car, 'Price'] + (ccp/(alpha*ccp*(1-ccp)))
                
    return dataset



# Update car price in Iterative Best Response
def set_car_price(x, p:float, j:int) -> None:
    x.loc[j, 'Price'] = p

def IBR(data, model):
    iterated_prices = []
    #price_y = data[data['Model'] == model]['Price']
    eta = data[data['Model'] == model]['markup%']
    cost = data[data['Model'] == model]['firm_cost']
    #print(price_y, eta)

    price = 0 #Initial guess
    iterated_prices.append(price)
    for i in range(0,5):
        iterated_prices[i] = cost + eta
        iterated_prices.append(iterated_prices[i])
                
    return iterated_prices.shape