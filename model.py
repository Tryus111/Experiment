#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:10:18 2023

@author: user
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score

raw = pd.read_csv('data_pre.csv')
df_model = raw.copy() 
df_model['DATE'] = pd.to_datetime(df_model['DATE'], format = '%Y %m %d')

type(df_model.DATE[0])

x1 = df_model.drop(df_model[['DATE', 'TS']], axis = 1)
y = df_model['TS']

x = sm.add_constant(x1)
result = sm.OLS(y, x).fit()
result.summary()

x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size = 0.1, random_state = 365)

# Linear Regression
from sklearn.linear_model import Lasso, LinearRegression
lm = LinearRegression() 
lm.fit(x_train, y_train)

cross_val_score(lm, x_train, y_train, scoring = 'neg_mean_absolute_error', cv = 3)

# Lasso Regression
lm_L = Lasso(0.1)
lm_L.fit(x_train, y_train)

cross_val_score(lm_L, x_train, y_train, scoring = 'neg_mean_absolute_error', cv = 3)

alpha = []
error = []

for i in range(1, 100):
    alpha.append(i/100)
    lml = Lasso(i/100)
    error.append(np.mean(cross_val_score(lm_L, x_train, y_train, scoring = 'neg_mean_absolute_error', cv = 3)))
    
plt.plot(alpha, error)

err = tuple(zip(alpha, error))
df_err = pd.DataFrame(err, columns = ['alpha', 'error'])

# Random Forest 
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(x_train, y_train)

cross_val_score(rf, x_train, y_train, scoring = 'neg_mean_absolute_error', cv = 3)

# Tuning
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators': range(10, 300, 10), 'criterion': ('absolute_error', 'poisson', 'friedman_mse', 'squared_error'), 'max_features': ('auto', 'sqrt', 'log2')}
gs = GridSearchCV(rf, parameters, scoring = 'neg_mean_absolute_error', cv = 3)
gs.fit(x_train, y_train)

gs.best_score_
