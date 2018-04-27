# This program performs a linear regression

import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import pandas as pd

data_filename = 'coursework_1_data_2017_v2_input.csv'

all_data = pd.read_csv(data_filename, index_col = 'local_authority_area')

##################################################
# Remove outliers (Rutland & City of London )
all_data = all_data[(all_data['09_10_KSI_change']<1) & (all_data['09_10_KSI_change']>-2)]

##################################################

# Investigate if Total spend is a predictor of change in KSIs
# Simple cases with total budget:

output_filename = 'Model_A_change in KSIs vs total budget.png'

# Inputs

x_in = all_data[['Total']]
y_in = all_data[['09_10_KSI_change']]
#safety_budget = all_data[['Safety_Total']]
#norm_2008_KSIs = all_data[['2008_KSI_pop']]

data = pd.concat([y_in, x_in], axis=1)
data.columns = ['y_in', 'x_in']

# create a fitted model in one line
lm = smf.ols(formula='y_in ~ x_in', data=data).fit()
# print the coefficients
print(lm.params)
print(lm.pvalues)
print(lm.summary())

data['yhat'] = lm.fittedvalues
data['resid'] = lm.resid

fig, ax = plt.subplots(1,2,  figsize=(8,4))
ax[0].scatter(data['x_in'], data['y_in'], s=8)
ax[0].plot(data['x_in'], data['yhat'], c='red', linewidth=1)

ax[0].set_title('Change in KSIs vs Total Spend', fontsize=10)
ax[0].set_xlabel('Total Spend per head', fontsize=8)
ax[0].set_ylabel('2010 change in KSIs per 10,000 pop.', fontsize=8)

ax[1].scatter(data['yhat'], data['resid'], c='black', s=5)

ax[1].set_title('Residuals Plot', fontsize=10)

plt.savefig(output_filename)

# print the coefficients
plt.figure(figsize=(10,6))
plt.text(0.01, 0.05, str(lm.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig('Model_A_output.png')

##################################################

# Investigate if Safety spend is a predictor of change in KSIs
# Simple cases with safety budget:

output_filename = 'Model_B_change in KSIs vs safety budget.png'

# Inputs

x_in = all_data[['Safety_Total']]
y_in = all_data[['09_10_KSI_change']]
#safety_budget = all_data[['Safety_Total']]
#norm_2008_KSIs = all_data[['2008_KSI_pop']]

data = pd.concat([y_in, x_in], axis=1)
data.columns = ['y_in', 'x_in']

# create a fitted model in one line
lm = smf.ols(formula='y_in ~ x_in', data=data).fit()
# print the coefficients
print(lm.params)
print(lm.pvalues)
print(lm.summary())

data['yhat'] = lm.fittedvalues
data['resid'] = lm.resid

fig, ax = plt.subplots(1,2,  figsize=(8,4))
ax[0].scatter(data['x_in'], data['y_in'], s=8)
ax[0].plot(data['x_in'], data['yhat'], c='red', linewidth=1)

ax[0].set_title('Change in KSIs vs Safety Spend', fontsize=10)
ax[0].set_xlabel('Safety Spend per head', fontsize=8)
ax[0].set_ylabel('2010 change in KSIs per 10,000 pop.', fontsize=8)

ax[1].scatter(data['yhat'], data['resid'], c='black', s=5)

ax[1].set_title('Residuals Plot', fontsize=10)

plt.savefig(output_filename)

# print the coefficients
plt.figure(figsize=(10,6))
plt.text(0.01, 0.05, str(lm.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig('Model_B_output.png')

##################################################

# Investigate if previous years KSIs is a predictor of budget
# Simple cases with total budget compared to 2008 KSIs per head:

output_filename = 'Model_F_Total budget vs previous KSIs.png'

# Inputs

y_in = all_data[['Total']]
x_in = all_data[['2008_KSI_pop']]
#safety_budget = all_data[['Safety_Total']]
#norm_2008_KSIs = all_data[['2008_KSI_pop']]

data = pd.concat([y_in, x_in], axis=1)
data.columns = ['y_in', 'x_in']

# create a fitted model in one line
lm = smf.ols(formula='y_in ~ x_in', data=data).fit()
# print the coefficients
print(lm.params)
print(lm.pvalues)
print(lm.summary())

data['yhat'] = lm.fittedvalues
data['resid'] = lm.resid

fig, ax = plt.subplots(1,2,  figsize=(8,4))
ax[0].scatter(data['x_in'], data['y_in'], s=8)
ax[0].plot(data['x_in'], data['yhat'], c='red', linewidth=1)

ax[0].set_title('Spend vs subsequent budget', fontsize=10)
ax[0].set_xlabel('Previous KSIs per 10,000 pop.', fontsize=8)
ax[0].set_ylabel('Total spend per pop.', fontsize=8)

ax[1].scatter(data['yhat'], data['resid'], c='black', s=5)

ax[1].set_title('Residuals Plot', fontsize=10)

plt.savefig(output_filename)

# print the coefficients
plt.figure(figsize=(10,6))
plt.text(0.01, 0.05, str(lm.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig('Model_F_output.png')

##################################################

# Investigate if previous years KSIs is a predictor of budget
# Simple cases with total budget compared to 2008 KSIs per head:

output_filename = 'Model_G_Budget component vs previous KSIs.png'

# Inputs

y_in = all_data[['DDC']]
x_in = all_data[['2008_KSI_pop']]
#safety_budget = all_data[['Safety_Total']]
#norm_2008_KSIs = all_data[['2008_KSI_pop']]

data = pd.concat([y_in, x_in], axis=1)
data.columns = ['y_in', 'x_in']

# create a fitted model in one line
lm = smf.ols(formula='y_in ~ x_in', data=data).fit()
# print the coefficients
print(lm.params)
print(lm.pvalues)
print(lm.summary())

data['yhat'] = lm.fittedvalues
data['resid'] = lm.resid

fig, ax = plt.subplots(1,2,  figsize=(8,4))
ax[0].scatter(data['x_in'], data['y_in'], s=8)
ax[0].plot(data['x_in'], data['yhat'], c='red', linewidth=1)

ax[0].set_title('Spend vs subsequent budget', fontsize=10)
ax[0].set_xlabel('Previous KSIs per 10,000 pop.', fontsize=8)
ax[0].set_ylabel('Total spend per pop.', fontsize=8)

ax[1].scatter(data['yhat'], data['resid'], c='black', s=5)

ax[1].set_title('Residuals Plot', fontsize=10)

plt.savefig(output_filename)

# print the coefficients
plt.figure(figsize=(10,6))
plt.text(0.01, 0.05, str(lm.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig('Model_G_output.png')


