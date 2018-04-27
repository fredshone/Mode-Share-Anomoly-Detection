# This program performs a multiple linear regression from data stored in a csv file.

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

output_filename = 'Model_C_change in KSIs vs budget.png'

# Inputs

#CyS - Cyclist_Safety_Budget_2009
#ChS - Child_Safety_Budget_2009
#McS - Motorcycle_Safety_Budget_2009
#DDC - Drink_Drive_Campaigns_Budget_2009
#PrC - Promote_Cycling_Budget_2009
#PCS - Promote_Car_Sharing_Budget_2009

CyS_in = all_data[['CyS']]
ChS_in = all_data[['ChS']]
McS_in = all_data[['McS']]
DDC_in = all_data[['DDC']]
PrC_in = all_data[['PrC']]
PCS_in = all_data[['PCS']]

y_in = all_data[['09_10_KSI_change']]
#safety_budget = all_data[['Safety_Total']]
#norm_2008_KSIs = all_data[['2008_KSI_pop']]

data = pd.concat([y_in, CyS_in, ChS_in, McS_in, DDC_in, PrC_in, PCS_in], axis=1)
data.columns = ['y_in', 'CyS_in', 'ChS_in', 'McS_in', 'DDC_in', 'PrC_in', 'PCS_in']

# create a fitted model in one line
lm = smf.ols(formula='y_in ~ CyS_in + ChS_in + McS_in + DDC_in + PrC_in + PCS_in', data=data).fit()
# print the coefficients
print(lm.params)
print(lm.pvalues)
print(lm.summary())

data['yhat'] = lm.fittedvalues
data['resid'] = lm.resid

fig, ax = plt.subplots(1,1,  figsize=(4,4))

ax.scatter(data['yhat'], data['resid'], c='black', s=5)

ax.set_title('Residuals Plot', fontsize=10)

plt.savefig(output_filename)

# print the coefficients
plt.figure(figsize=(10,6))
plt.text(0.01, 0.05, str(lm.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig('Model_C_output.png')

##################################################

output_filename = 'MR2_Model_D_change in KSIs vs budget adjusted.png'

# create a fitted model in one line
lm = smf.ols(formula='y_in ~ CyS_in + ChS_in + McS_in + DDC_in + PCS_in', data=data).fit()
# print the coefficients
print(lm.params)
print(lm.pvalues)
print(lm.summary())

data['yhat'] = lm.fittedvalues
data['resid'] = lm.resid

fig, ax = plt.subplots(1,1,  figsize=(4,4))

ax.scatter(data['yhat'], data['resid'], c='black', s=5)

ax.set_title('Residuals Plot', fontsize=10)

plt.savefig(output_filename)

# print the coefficients
plt.figure(figsize=(10,6))
plt.text(0.01, 0.05, str(lm.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig('Model_D_output.png')

##################################################

output_filename = 'Model_E_change in KSIs vs CyS and ChS.png'

# create a fitted model in one line
lm = smf.ols(formula='y_in ~ CyS_in + ChS_in', data=data).fit()
# print the coefficients
print(lm.params)
print(lm.pvalues)
print(lm.summary())

data['yhat'] = lm.fittedvalues
data['resid'] = lm.resid

fig, ax = plt.subplots(1,1,  figsize=(4,4))

ax.scatter(data['yhat'], data['resid'], c='black', s=5)

ax.set_title('Residuals Plot', fontsize=10)

plt.savefig(output_filename)

# print the coefficients
plt.figure(figsize=(10,6))
plt.text(0.01, 0.05, str(lm.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig('Model_E_output.png')

##################################################

