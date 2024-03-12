import pandas as pd
import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt
import scipy.stats as st
import statsmodels.api as sm
import seaborn as sns


def get_coefficients_OLS(data):
    Nobs=data['ID'].count()
    data['const']=np.ones((Nobs,1))
    data = data[data['Market share'] != 0]    

    data = data.copy()

    data = pd.get_dummies(data, columns=['Segment'], drop_first=True)
    data = pd.get_dummies(data, columns=['Year'], drop_first=True)
    data['China'] = (data['Country'] == 'CH').astype(int)

    data['log_market_share'] = np.log(data['Market share'])

    y = data['log_market_share']
    x = data[['const', 'Range', 'Price', 'HP', 'Chargetime']]
    dummies = data[['Segment_B', 'Segment_C', 'Segment_D', 'Segment_E', 'Segment_F', 'Segment_M', 'Segment_J',
                    'Year_2014', 'Year_2015', 'Year_2016', 'Year_2017', 'Year_2018', 'Year_2019', 'Year_2020', 'Year_2021', 'Year_2022', 'Year_2023',
                    'China']]
    X = pd.concat([x, dummies], axis=1)

    OLS_model = sm.OLS(y, X)
    OLS_result = OLS_model.fit()
    coefficients = OLS_result.params

    return coefficients