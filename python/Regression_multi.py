#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 16:00:48 2018

@author: fredshone
"""

import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import seaborn as sns
from statsmodels.graphics.api import abline_plot
from scipy import stats
import numpy as np
from statsmodels.sandbox.regression.predstd import wls_prediction_std

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

#DROP
#X = data.drop(['mode_share', 'bus_travel', 'other_travel', 'ward_name', 'Borough', 'Zone', 'area_km', '2013_population', '2011_cars_own_density'], axis=1)
X = data.drop(['PTAL_AI2015','ward_name','Borough','Zone','mode_share','bus_travel', 'other_travel'], axis = 1)
#add constant
X['constant'] = 1

###############
#GLM BINOMIAL LOGIT

glm_binom = sm.GLM(y_bi, X, family=sm.families.Binomial())
model_1_binomial_results = glm_binom.fit()
print(model_1_binomial_results.summary())

#record results
model_1_results = data
model_1_results['y_model'] = model_1_binomial_results.mu
model_1_results['error'] = model_1_results['mode_share'] - model_1_results['y_model']
model_1_results['pearsons'] = stats.zscore(model_1_binomial_results.resid_pearson)

#Plot Yhat vs Y
nobs = model_1_binomial_results.nobs
y_in = model_1_binomial_results.model.endog
yhat = model_1_binomial_results.mu

output_filename = 'Model_1_Y_modelled'
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, ylabel='Observed Values', xlabel='Fitted Values')
ax.scatter(yhat, y_in)
y_vs_yhat = sm.OLS(y_in, sm.add_constant(yhat, prepend=True)).fit()
fig = abline_plot(model_results=y_vs_yhat, ax=ax)
plt.savefig(output_filename)

#PEARSONS
#http://data.princeton.edu/wws509/notes/c3s8.html
            
output_filename = 'Model_1_Pearsons'
fig = plt.figure(figsize=(12,8))
sns.pairplot(x_vars=['y_model'], y_vars=['pearsons'], data=model_1_results, hue='Zone', size=5)
plt.plot([0.0, 0.3],[0.0, 0.0], 'k-');
plt.plot([0, 0.3],[2, 2], 'r--');
plt.plot([0, 0.3],[-2, -2], 'r--');
plt.savefig(output_filename)

#Standardised deviance residuals
resid = model_1_binomial_results.resid_deviance
resid_std = stats.zscore(resid) 
kde_resid = sm.nonparametric.KDEUnivariate(resid_std)
kde_resid.fit()

output_filename = 'Model_1_standardized_dev_res'
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, title="Standardized Deviance Residuals")
ax.hist(resid_std, bins=25, normed=True);
ax.plot(kde_resid.support, kde_resid.density, 'r');
plt.savefig(output_filename)

#QQ Plot
output_filename = 'Model_1_QQ'
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
fig = sm.graphics.qqplot(resid, line='r', ax=ax)
plt.savefig(output_filename)

###############
#LINEAR

linear_mod = sm.OLS(y, X)
model_1_linear_results = linear_mod.fit()
print(model_1_linear_results.summary())

###############
#CORRELATION
corrX = X.drop(['constant'], axis = 1).corr()

output_filename = 'Model_1_Correlation_Matrix'

# Generate a mask for the upper triangle
mask = np.zeros_like(corrX, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 12))
#plt.margins(0.1)
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.15)
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corrX, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig(output_filename)

################
#Extract outliers

mask = np.array((model_1_results[['error']] > 0.1) | (model_1_results[['error']] < -0.1))
model_1_outliers = model_1_results[mask]

mask = np.array((model_1_results[['pearsons']] > 2) | (model_1_results[['pearsons']] < -2))
model_1_outliers_pearsons = model_1_results[mask]

######################################################
#MODEL 2

#DROP 2011_ethnic_white 
X = X.drop(['2013_population','pop_16to64_perc','2011_ethnic_mixed', 
              'weekday_boardings_allday', 'pop_0to15_perc'], axis = 1)

###############
#GLM BINOMIAL LOGIT

glm_binom = sm.GLM(y_bi, X, family=sm.families.Binomial())
model_2_binomial_results = glm_binom.fit()
print(model_2_binomial_results.summary())

#record results
model_2_results = data
model_2_results['y_model'] = model_2_binomial_results.mu
model_2_results['error'] = model_1_results['mode_share'] - model_1_results['y_model']
model_2_results['pearsons'] = stats.zscore(model_2_binomial_results.resid_pearson)

#Plot Yhat vs Y
nobs = model_2_binomial_results.nobs
y_in = model_2_binomial_results.model.endog
yhat = model_2_binomial_results.mu

output_filename = 'Model_2_Y_modelled'
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, ylabel='Observed Values', xlabel='Fitted Values')
ax.scatter(yhat, y_in)
y_vs_yhat = sm.OLS(y_in, sm.add_constant(yhat, prepend=True)).fit()
fig = abline_plot(model_results=y_vs_yhat, ax=ax)
plt.savefig(output_filename)

#PEARSONS
#http://data.princeton.edu/wws509/notes/c3s8.html
            
output_filename = 'Model_2_Pearsons'
fig = plt.figure(figsize=(12,8))
sns.pairplot(x_vars=['y_model'], y_vars=['pearsons'], data=model_2_results, hue='Zone', size=5)
plt.plot([0.0, 0.3],[0.0, 0.0], 'k-');
plt.plot([0, 0.3],[2, 2], 'r--');
plt.plot([0, 0.3],[-2, -2], 'r--');
plt.savefig(output_filename)

#Standardised deviance residuals
resid = model_2_binomial_results.resid_deviance
resid_std = stats.zscore(resid) 
kde_resid = sm.nonparametric.KDEUnivariate(resid_std)
kde_resid.fit()

output_filename = 'Model_2_standardized_dev_res'
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, title="Standardized Deviance Residuals")
ax.hist(resid_std, bins=25, normed=True);
ax.plot(kde_resid.support, kde_resid.density, 'r');
plt.savefig(output_filename)

#QQ Plot
output_filename = 'Model_2_QQ'
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
fig = sm.graphics.qqplot(resid, line='r', ax=ax)
plt.savefig(output_filename)

###############
#LINEAR

linear_mod = sm.OLS(y, X)
model_2_linear_results = linear_mod.fit()
print(model_2_linear_results.summary())

###############
#CORRELATION
corrX = X.drop(['constant'], axis = 1).corr()

output_filename = 'Model_2_Correlation_Matrix'
# Generate a mask for the upper triangle
mask = np.zeros_like(corrX, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 12))
#plt.margins(0.1)
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.15)
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corrX, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig(output_filename)

################
#Extract outliers

mask = np.array((model_2_results[['error']] > 0.1) | (model_2_results[['error']] < -0.1))
model_2_outliers = model_1_results[mask]

mask = np.array((model_2_results[['pearsons']] > 2) | (model_2_results[['pearsons']] < -2))
model_2_outliers_pearsons = model_1_results[mask]

######################################################
#MODEL 3

#DROP 2011_ethnic_white 
X = X.drop(['PTAL_tram2015','speed_weekday_allday','change_weekday_allday', 
              'area_km', '2011_cars_own_density'], axis = 1)

###############
#GLM BINOMIAL LOGIT

glm_binom = sm.GLM(y_bi, X, family=sm.families.Binomial())
model_3_binomial_results = glm_binom.fit()
print(model_3_binomial_results.summary())

#record results
model_3_results = data
model_3_results['y_model'] = model_3_binomial_results.mu
model_3_results['error'] = model_3_results['mode_share'] - model_3_results['y_model']
model_3_results['pearsons'] = stats.zscore(model_3_binomial_results.resid_pearson)

#Plot Yhat vs Y
nobs = model_3_binomial_results.nobs
y_in = model_3_binomial_results.model.endog
yhat = model_3_binomial_results.mu

output_filename = 'Model_3_Y_modelled'
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, ylabel='Observed Values', xlabel='Fitted Values')
ax.scatter(yhat, y_in)
y_vs_yhat = sm.OLS(y_in, sm.add_constant(yhat, prepend=True)).fit()
fig = abline_plot(model_results=y_vs_yhat, ax=ax)
plt.savefig(output_filename)

#PEARSONS
#http://data.princeton.edu/wws509/notes/c3s8.html
            
output_filename = 'Model_3_Pearsons'
fig = plt.figure(figsize=(12,8))
sns.pairplot(x_vars=['y_model'], y_vars=['pearsons'], data=model_3_results, hue='Zone', size=5)
plt.plot([0.0, 0.3],[0.0, 0.0], 'k-');
plt.plot([0, 0.3],[2, 2], 'r--');
plt.plot([0, 0.3],[-2, -2], 'r--');
plt.savefig(output_filename)

#Standardised deviance residuals
resid = model_3_binomial_results.resid_deviance
resid_std = stats.zscore(resid) 
kde_resid = sm.nonparametric.KDEUnivariate(resid_std)
kde_resid.fit()

output_filename = 'Model_3_standardized_dev_res'
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, title="Standardized Deviance Residuals")
ax.hist(resid_std, bins=25, normed=True);
ax.plot(kde_resid.support, kde_resid.density, 'r');
plt.savefig(output_filename)

#QQ Plot
output_filename = 'Model_3_QQ'
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
fig = sm.graphics.qqplot(resid, line='r', ax=ax)
plt.savefig(output_filename)

###############
#LINEAR

linear_mod = sm.OLS(y, X)
model_3_linear_results = linear_mod.fit()
print(model_3_linear_results.summary())

###############
#CORRELATION
corrX = X.drop(['constant'], axis = 1).corr()

output_filename = 'Model_3_Correlation_Matrix'
# Generate a mask for the upper triangle
mask = np.zeros_like(corrX, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 12))
#plt.margins(0.1)
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.15)
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corrX, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig(output_filename)

################
#Extract outliers

mask = np.array((model_3_results[['error']] > 0.1) | (model_3_results[['error']] < -0.1))
model_3_outliers = model_3_results[mask]

mask = np.array((model_3_results[['pearsons']] > 2) | (model_3_results[['pearsons']] < -2))
model_3_outliers_pearsons = model_3_results[mask]

################################################################
#MODEL 4

#DROP 2011_ethnic_white 
X = X.drop(['speed_weekday_IN','change_weekday_IN'], axis = 1)

###############
#GLM BINOMIAL LOGIT

glm_binom = sm.GLM(y_bi, X, family=sm.families.Binomial())
model_4_binomial_results = glm_binom.fit()
print(model_4_binomial_results.summary())

#record results
model_4_results = data
model_4_results['y_model'] = model_4_binomial_results.mu
model_4_results['error'] = model_4_results['mode_share'] - model_4_results['y_model']
model_4_results['pearsons'] = stats.zscore(model_4_binomial_results.resid_pearson)

model_4_results['tag'] = "normal_performer"
model_4_results['tag'][model_4_results['pearsons'] < -2] = 'under_performer'
model_4_results['tag'][model_4_results['pearsons'] > 2] = 'over_performer'

output_filename = 'XX_model_4_results_Summary_Heatmap'
f, ax = plt.subplots(figsize=(10, 4))

plt.subplots_adjust(bottom=0.2)
plt.subplots_adjust(left=0.2)
model_4_outlier_summary = model_4_results.groupby('tag').mean().sort_values(by = 'mode_share', ascending = False)
normalized_borough_summary=(model_4_outlier_summary-model_4_outlier_summary.min())/(model_4_outlier_summary.max()-model_4_outlier_summary.min())
sns.heatmap(normalized_borough_summary, center=0, cmap = cmap, square=True, linewidths=0.5, cbar_kws={"shrink": .5})

plt.savefig(output_filename)


#Plot Yhat vs Y
nobs = model_4_binomial_results.nobs
y_in = model_4_binomial_results.model.endog
yhat = model_4_binomial_results.mu

output_filename = 'Model_4_Y_modelled'
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, ylabel='Observed Values', xlabel='Fitted Values')
ax.scatter(yhat, y_in)
y_vs_yhat = sm.OLS(y_in, sm.add_constant(yhat, prepend=True)).fit()
fig = abline_plot(model_results=y_vs_yhat, ax=ax)
plt.savefig(output_filename)

#PEARSONS
#http://data.princeton.edu/wws509/notes/c3s8.html
            
output_filename = 'Model_4_Pearsons'
fig = plt.figure(figsize=(12,8))
sns.pairplot(x_vars=['y_model'], y_vars=['pearsons'], data=model_4_results, hue='Zone', size=5)
plt.plot([0.0, 0.3],[0.0, 0.0], 'k-');
plt.plot([0, 0.3],[2, 2], 'r--');
plt.plot([0, 0.3],[-2, -2], 'r--');
plt.savefig(output_filename)

#Standardised deviance residuals
resid = model_4_binomial_results.resid_deviance
resid_std = stats.zscore(resid) 
kde_resid = sm.nonparametric.KDEUnivariate(resid_std)
kde_resid.fit()

output_filename = 'Model_4_standardized_dev_res'
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, title="Standardized Deviance Residuals")
ax.hist(resid_std, bins=25, normed=True);
ax.plot(kde_resid.support, kde_resid.density, 'r');
plt.savefig(output_filename)

#QQ Plot
output_filename = 'Model_4_QQ'
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
fig = sm.graphics.qqplot(resid, line='r', ax=ax)
plt.savefig(output_filename)

###############
#LINEAR

linear_mod = sm.OLS(y, X)
model_4_linear_results = linear_mod.fit()
print(model_4_linear_results.summary())

###############
#CORRELATION
corrX = X.drop(['constant'], axis = 1).corr()

output_filename = 'Model_4_Correlation_Matrix'
# Generate a mask for the upper triangle
mask = np.zeros_like(corrX, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 12))
#plt.margins(0.1)
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.15)
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corrX, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig(output_filename)

################
#Extract outliers

mask = np.array((model_4_results[['error']] > 0.1) | (model_4_results[['error']] < -0.1))
model_4_outliers = model_4_results[mask]

mask = np.array((model_4_results[['pearsons']] > 2) | (model_4_results[['pearsons']] < -2))
model_4_outliers_pearsons = model_4_results[mask]




######################################################
#MODEL 5

#DROP 2011_ethnic_white 
X = X.drop(['2011_ethnic_white','speed_weekday_AM', 'pop_65plus_perc',
            '2014_med_house_price'], axis = 1)
X = X.drop(['PTAL_bus2015'], axis = 1)
#add constant
###############
#GLM BINOMIAL LOGIT

glm_binom = sm.GLM(y_bi, X, family=sm.families.Binomial())
model_5_binomial_results = glm_binom.fit()
print(model_5_binomial_results.summary())

#record results
model_5_results = data
model_5_results['y_model'] = model_5_binomial_results.mu
model_5_results['error'] = model_5_results['mode_share'] - model_5_results['y_model']
model_5_results['pearsons'] = stats.zscore(model_5_binomial_results.resid_pearson)

#Plot Yhat vs Y
nobs = model_5_binomial_results.nobs
y_in = model_5_binomial_results.model.endog
yhat = model_5_binomial_results.mu

output_filename = 'Model_5_Y_modelled'
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, ylabel='Observed Values', xlabel='Fitted Values')
ax.scatter(yhat, y_in)
y_vs_yhat = sm.OLS(y_in, sm.add_constant(yhat, prepend=True)).fit()
fig = abline_plot(model_results=y_vs_yhat, ax=ax)
plt.savefig(output_filename)

#PEARSONS
#http://data.princeton.edu/wws509/notes/c3s8.html
            
output_filename = 'Model_5_Pearsons'
fig = plt.figure(figsize=(12,8))
sns.pairplot(x_vars=['y_model'], y_vars=['pearsons'], data=model_5_results, hue='Zone', size=5)
plt.plot([0.0, 0.3],[0.0, 0.0], 'k-');
plt.plot([0, 0.3],[2, 2], 'r--');
plt.plot([0, 0.3],[-2, -2], 'r--');
plt.savefig(output_filename)

#Standardised deviance residuals
resid = model_5_binomial_results.resid_deviance
resid_std = stats.zscore(resid) 
kde_resid = sm.nonparametric.KDEUnivariate(resid_std)
kde_resid.fit()

output_filename = 'Model_5_standardized_dev_res'
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, title="Standardized Deviance Residuals")
ax.hist(resid_std, bins=25, normed=True);
ax.plot(kde_resid.support, kde_resid.density, 'r');
plt.savefig(output_filename)

#QQ Plot
output_filename = 'Model_5_QQ'
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
fig = sm.graphics.qqplot(resid, line='r', ax=ax)
plt.savefig(output_filename)

###############
#LINEAR

linear_mod = sm.OLS(y, X)
model_5_linear_results = linear_mod.fit()
print(model_5_linear_results.summary())

###############
#CORRELATION
corrX = X.drop(['constant'], axis = 1).corr()

output_filename = 'Model_5_Correlation_Matrix'
# Generate a mask for the upper triangle
mask = np.zeros_like(corrX, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 12))
#plt.margins(0.1)
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.15)
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corrX, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig(output_filename)

################
#Extract outliers

mask = np.array((model_5_results[['error']] > 0.1) | (model_5_results[['error']] < -0.1))
model_5_outliers = model_5_results[mask]

mask = np.array((model_5_results[['pearsons']] > 2) | (model_5_results[['pearsons']] < -2))
model_5_outliers_pearsons = model_5_results[mask]

######################################################
#MODEL 6

#DROP 2011_ethnic_white 
X = X.drop(['2013_pop_density'], axis = 1)
X = X.drop(['2014_mean_income'], axis = 1)
X = X.drop(['PTAL_rail2015'], axis = 1)

###############
#GLM BINOMIAL LOGIT

glm_binom = sm.GLM(y_bi, X, family=sm.families.Binomial())
model_6_binomial_results = glm_binom.fit()
print(model_6_binomial_results.summary())

#record results
model_6_results = data
model_6_results['y_model'] = model_6_binomial_results.mu
model_6_results['error'] = model_6_results['mode_share'] - model_6_results['y_model']
model_6_results['pearsons'] = stats.zscore(model_6_binomial_results.resid_pearson)

#Plot Yhat vs Y
nobs = model_6_binomial_results.nobs
y_in = model_6_binomial_results.model.endog
yhat = model_6_binomial_results.mu

output_filename = 'Model_6_Y_modelled'
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, ylabel='Observed Values', xlabel='Fitted Values')
ax.scatter(yhat, y_in)
y_vs_yhat = sm.OLS(y_in, sm.add_constant(yhat, prepend=True)).fit()
fig = abline_plot(model_results=y_vs_yhat, ax=ax)
plt.savefig(output_filename)

#PEARSONS
#http://data.princeton.edu/wws509/notes/c3s8.html
            
output_filename = 'Model_6_Pearsons'
fig = plt.figure(figsize=(12,8))
sns.pairplot(x_vars=['y_model'], y_vars=['pearsons'], data=model_6_results, hue='Zone', size=5)
plt.plot([0.0, 0.3],[0.0, 0.0], 'k-');
plt.plot([0, 0.3],[2, 2], 'r--');
plt.plot([0, 0.3],[-2, -2], 'r--');
plt.savefig(output_filename)

#Standardised deviance residuals
resid = model_6_binomial_results.resid_deviance
resid_std = stats.zscore(resid) 
kde_resid = sm.nonparametric.KDEUnivariate(resid_std)
kde_resid.fit()

output_filename = 'Model_6_standardized_dev_res'
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, title="Standardized Deviance Residuals")
ax.hist(resid_std, bins=25, normed=True);
ax.plot(kde_resid.support, kde_resid.density, 'r');
plt.savefig(output_filename)

#QQ Plot
output_filename = 'Model_6_QQ'
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
fig = sm.graphics.qqplot(resid, line='r', ax=ax)
plt.savefig(output_filename)

###############
#LINEAR

linear_mod = sm.OLS(y, X)
model_6_linear_results = linear_mod.fit()
print(model_6_linear_results.summary())

###############
#CORRELATION
corrX = X.drop(['constant'], axis = 1).corr()

output_filename = 'Model_6_Correlation_Matrix'
# Generate a mask for the upper triangle
mask = np.zeros_like(corrX, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 12))
#plt.margins(0.1)
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.15)
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corrX, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig(output_filename)

################
#Extract outliers

mask = np.array((model_6_results[['error']] > 0.1) | (model_6_results[['error']] < -0.1))
model_6_outliers = model_6_results[mask]

mask = np.array((model_6_results[['pearsons']] > 2) | (model_6_results[['pearsons']] < -2))
model_6_outliers_pearsons = model_6_results[mask]

######################################################
#MODEL 7

#DROP 2011_ethnic_white 
X = X.drop(['weekend_boardings_ratio'], axis = 1)
X = X.drop(['2011_cars_per_household'], axis = 1)
X = X.drop(['PTAL_LUL2015'], axis = 1)


###############
#GLM BINOMIAL LOGIT

glm_binom = sm.GLM(y_bi, X, family=sm.families.Binomial())
model_7_binomial_results = glm_binom.fit()
print(model_7_binomial_results.summary())

#record results
model_7_results = data
model_7_results['y_model'] = model_7_binomial_results.mu
model_7_results['error'] = model_7_results['mode_share'] - model_7_results['y_model']
model_7_results['pearsons'] = stats.zscore(model_7_binomial_results.resid_pearson)

#Plot Yhat vs Y
nobs = model_7_binomial_results.nobs
y_in = model_7_binomial_results.model.endog
yhat = model_7_binomial_results.mu

output_filename = 'Model_7_Y_modelled'
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, ylabel='Observed Values', xlabel='Fitted Values')
ax.scatter(yhat, y_in)
y_vs_yhat = sm.OLS(y_in, sm.add_constant(yhat, prepend=True)).fit()
fig = abline_plot(model_results=y_vs_yhat, ax=ax)
plt.savefig(output_filename)

#PEARSONS
#http://data.princeton.edu/wws509/notes/c3s8.html
            
output_filename = 'Model_7_Pearsons'
fig = plt.figure(figsize=(12,8))
sns.pairplot(x_vars=['y_model'], y_vars=['pearsons'], data=model_7_results, hue='Zone', size=5)
plt.plot([0.0, 0.3],[0.0, 0.0], 'k-');
plt.plot([0, 0.3],[2, 2], 'r--');
plt.plot([0, 0.3],[-2, -2], 'r--');
plt.savefig(output_filename)

#Standardised deviance residuals
resid = model_7_binomial_results.resid_deviance
resid_std = stats.zscore(resid) 
kde_resid = sm.nonparametric.KDEUnivariate(resid_std)
kde_resid.fit()

output_filename = 'Model_7_standardized_dev_res'
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, title="Standardized Deviance Residuals")
ax.hist(resid_std, bins=25, normed=True);
ax.plot(kde_resid.support, kde_resid.density, 'r');
plt.savefig(output_filename)

#QQ Plot
output_filename = 'Model_7_QQ'
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
fig = sm.graphics.qqplot(resid, line='r', ax=ax)
plt.savefig(output_filename)

###############
#LINEAR

linear_mod = sm.OLS(y, X)
model_7_linear_results = linear_mod.fit()
print(model_7_linear_results.summary())

###############
#CORRELATION
corrX = X.drop(['constant'], axis = 1).corr()

output_filename = 'Model_7_Correlation_Matrix'
# Generate a mask for the upper triangle
mask = np.zeros_like(corrX, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 12))
#plt.margins(0.1)
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.15)
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corrX, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig(output_filename)

################
#Extract outliers

mask = np.array((model_7_results[['error']] > 0.1) | (model_7_results[['error']] < -0.1))
model_7_outliers = model_7_results[mask]

mask = np.array((model_7_results[['pearsons']] > 2) | (model_7_results[['pearsons']] < -2))
model_7_outliers_pearsons = model_7_results[mask]

######################################################
#MODEL 8

#DROP 2011_ethnic_white 
X = X.drop(['change_weekday_AM'], axis = 1)
X = X.drop(['alightings_boardings_ratio'], axis = 1)
X = X.drop(['2011_ethnic_asian'], axis = 1)


###############
#GLM BINOMIAL LOGIT

glm_binom = sm.GLM(y_bi, X, family=sm.families.Binomial())
model_8_binomial_results = glm_binom.fit()
print(model_8_binomial_results.summary())

#record results
model_8_results = data
model_8_results['y_model'] = model_8_binomial_results.mu
model_8_results['error'] = model_8_results['mode_share'] - model_8_results['y_model']
model_8_results['pearsons'] = stats.zscore(model_8_binomial_results.resid_pearson)

#Plot Yhat vs Y
nobs = model_8_binomial_results.nobs
y_in = model_8_binomial_results.model.endog
yhat = model_8_binomial_results.mu

output_filename = 'Model_8_Y_modelled'
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, ylabel='Observed Values', xlabel='Fitted Values')
ax.scatter(yhat, y_in)
y_vs_yhat = sm.OLS(y_in, sm.add_constant(yhat, prepend=True)).fit()
fig = abline_plot(model_results=y_vs_yhat, ax=ax)
plt.savefig(output_filename)

#PEARSONS
#http://data.princeton.edu/wws509/notes/c3s8.html
            
output_filename = 'Model_8_Pearsons'
fig = plt.figure(figsize=(12,8))
sns.pairplot(x_vars=['y_model'], y_vars=['pearsons'], data=model_8_results, hue='Zone', size=5)
plt.plot([0.0, 0.3],[0.0, 0.0], 'k-');
plt.plot([0, 0.3],[2, 2], 'r--');
plt.plot([0, 0.3],[-2, -2], 'r--');
plt.savefig(output_filename)

#Standardised deviance residuals
resid = model_8_binomial_results.resid_deviance
resid_std = stats.zscore(resid) 
kde_resid = sm.nonparametric.KDEUnivariate(resid_std)
kde_resid.fit()

output_filename = 'Model_8_standardized_dev_res'
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, title="Standardized Deviance Residuals")
ax.hist(resid_std, bins=25, normed=True);
ax.plot(kde_resid.support, kde_resid.density, 'r');
plt.savefig(output_filename)

#QQ Plot
output_filename = 'Model_8_QQ'
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
fig = sm.graphics.qqplot(resid, line='r', ax=ax)
plt.savefig(output_filename)

###############
#LINEAR

linear_mod = sm.OLS(y, X)
model_8_linear_results = linear_mod.fit()
print(model_8_linear_results.summary())

###############
#CORRELATION
corrX = X.drop(['constant'], axis = 1).corr()

output_filename = 'Model_8_Correlation_Matrix'
# Generate a mask for the upper triangle
mask = np.zeros_like(corrX, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 12))
#plt.margins(0.1)
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.15)
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corrX, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig(output_filename)

################
#Extract outliers

mask = np.array((model_8_results[['error']] > 0.1) | (model_8_results[['error']] < -0.1))
model_8_outliers = model_8_results[mask]

mask = np.array((model_8_results[['pearsons']] > 2) | (model_8_results[['pearsons']] < -2))
model_8_outliers_pearsons = model_8_results[mask]


##########################################
#Outlier data


