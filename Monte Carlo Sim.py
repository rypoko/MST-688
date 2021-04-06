# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 14:06:22 2021

@author: ryanp
"""

import pandas as pd
import pandas_montecarlo
import numpy as np
import scipy
import seaborn as sns
import math
import random
import matplotlib.pyplot as plt  
from scipy.stats import norm

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.model_selection import train_test_split

data = r'C:\Users\ryanp\OneDrive\Documents\NIU\MST 688 Data Science Applications\HW\Git\Monte_Carlo_Simulation_Loan_Status\loan_timing.csv'

df = pd.read_csv(data)
df.head()
df.describe()

#%% Split data
df_default = df[df['days from origination to chargeoff'] > 0] #default loans
df_current = df[np.isnan(df['days from origination to chargeoff'])] #current loans
df_default.describe()

#df_default['default_log'] = np.log(df_default['days from origination to chargeoff'])
x = df_default.iloc[:, 1].values
y = df_default.iloc[:,2].values
#%% histogram
ax1 = df_default.plot.hist(bins=100, alpha=0.5, title="Loans in default")
ax2 = df_current.plot.hist(bins=100, alpha=0.5, title="Current loans")
df_default['default_log'].plot.hist(bins=100, alpha=0.5, title="Log of loans in default")
#%% scatter plot
ax3 = df.plot.scatter(x='days since origination', y='days from origination to chargeoff',c='green', s=2, title="Days to charge off v days since origination")

#%% Normal distribution
ax4 = df_default.plot.scatter(x='days since origination', y='days from origination to chargeoff',c='red', s=2, title="Defaulted loans: days to chargeoff v days since origination")
ax5 = df_default.plot.hist(x='days since origination', y='days from origination to chargeoff',bins=100, alpha=0.5, title="Number of defaults vs days since origination")
ax6 = plt.plot(norm.pdf(df_default['days from origination to chargeoff'][:]))
#%%T-test P-value
a = df_default
b = df_current
t_test = scipy.stats.ttest_ind(a, b, axis=0, equal_var=True)

#%%
category = ['days since origination', 'days from origination to chargeoff']
encoder = LabelEncoder()
for i in category:
    df_default[i] = encoder.fit_transform(df_default[i])
    df_default.dtypes
    
#%% modeling
def class_model(model, data, predictors, outcome):
    model.fit(data[predictors].values, data[outcome].values) #fit model
    predictions = model.predict(data[predictors]) #make predictions
    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print("Accuracy: %s" % "{0:.3%}".format(accuracy))
    
    kf = KFold(data.shape[0], n_folds = 5)
    error = []
    for train , test in kf:
        train_predictors = (data[predictors].iloc[train,:])
        train_target = data[outcome].iloc[train]
        model.fit(train_predictors, train_target)
    
    error.append(model.score(data[predictors].iloc[test,:],
                             data[outcome].iloc[test]))
    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

#%%logistic regression
outcome_var = 'days from origination to chargeoff'
model = LogisticRegression()
predictor_var = 'days since origination'
data2 = df_default.values.reshape(-1,1).astype(int)
class_model(model, data2, predictor_var, outcome_var)

#%% linear regression
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=0)
x_train= x_train.reshape(-1, 1)
y_train= y_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)




