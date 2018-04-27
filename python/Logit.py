#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 21:58:09 2018

@author: fredshone
"""

import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import seaborn as sns
from statsmodels.graphics.api import abline_plot

data_filename = 'data.csv'
data = pd.read_csv(data_filename, index_col = 'ward_code')

##NOTE Consider removing Welsh Harp Ward (Borough of Brent), due to outlier speed change (>10km/hr)
data = data.drop('E05000103')

#add binomial results
data['bus_travel'] = data['mode_share'] * data['2013_population']
data['other_travel'] = data['2013_population'] - data['bus_travel']

# set up data

y = data[['mode_share']]
y_bi = data[['bus_travel', 'other_travel']]

#X = data.drop(['mode_share', 'ward_name', 'Borough', 'Zone'], axis=1)

X = data[['2011_ethnic_black']]
#add constant
X['constant'] = 1




##LOGIT MODEL
#logit_mod = sm.Logit(y, X)
#logit_results = logit_mod.fit()
#print(logit_results.summary())
#
##fig, ax = plt.subplots()
##fig = sm.graphics.plot_fit(logit_results, 0, ax=ax)
##ax.set_ylabel("Murder Rate")
##ax.set_xlabel("Poverty Level")
##ax.set_title("Linear Regression")

###############
#BINOMIAL

glm_binom = sm.GLM(y_bi, X, family=sm.families.Binomial())
binomial_results = glm_binom.fit()
print(binomial_results.summary())

#nobs = binomial_results.nobs
##y_temp = y_bi[['bus_travel']]/y_bi.sum(1)
#yhat = binomial_results.mu
#
#fig, ax = plt.subplots(figsize=(8, 8))
#ax.scatter(yhat, y)
#line_fit = sm.OLS(y, sm.add_constant(yhat, prepend=True)).fit()
#abline_plot(model_results=line_fit, ax=ax)
#
#
#ax.set_title('Model Fit Plot')
#ax.set_ylabel('Observed values')
#ax.set_xlabel('Fitted values')

###############
#LINEAR
#X["constant"] = 1
linear_mod = sm.OLS(y, X)
linear_results = linear_mod.fit()
print(linear_results.summary())

#fig, ax = plt.subplots()
#fig = sm.graphics.plot_fit(linear_results, 0, ax=ax)
#ax.set_ylabel("Murder Rate")
#ax.set_xlabel("Poverty Level")
#ax.set_title("Linear Regression")




# Probit regression. Probit analysis will produce results similar logistic regression. 
#The choice of probit versus logit depends largely on individual preferences.

# OLS regression. When used with a binary response variable, this model is known as a 
#linear probability model and can be used as a way to describe conditional probabilities. 
#However, the errors (i.e., residuals) from the linear probability model violate the 
#homoskedasticity and normality of errors assumptions of OLS regression, resulting in 
#invalid standard errors and hypothesis tests. For a more thorough discussion of these and 
#other problems with the linear probability model, see Long (1997, p. 38-40).

# Two-group discriminant function analysis. A multivariate method for dichotomous outcome 
#variables.

# Hotelling’s T2. The 0/1 outcome is turned into the grouping variable, and the former 
#predictors are turned into outcome variables. This will produce an overall test of significance 
#but will not give individual coefficients for each variable, and it is unclear the extent to 
#which each "predictor" is adjusted for the impact of the other "predictors."
