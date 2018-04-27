# Creates Box Plots.


# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set style
#plt.style.use('fivethirtyeight')
#plt.style.use('ggplot')

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the title

# Header list for info:

#1. local_authority_area,
#2. 2008_KSI,
#3. 2009_KSI,
#4. 2010_KSI,
#5. 2008_pop,
#6. 2009_pop,
#7. 2010_pop,
#8. Total_Budget_2009,
#9. Cyclist_Safety_Budget_2009,
#10. Child_Safety_Budget_2009,
#11. Motorcycle_Safety_Budget_2009,
#12. Drink_Drive_Campaigns_Budget_2009,
#13. Promote_Cycling_Budget_2009,
#14. Promote_Car_Sharing_Budget_2009,
#15. local_authority_type,
#16. 2008_KSI_pop,
#17. 2009_KSI_pop,
#18. 2010_KSI_pop,
#19. KSI_warning,
#20. 2008_2009_KSI_pop_change,
#21. 2009_2010_KSI_pop_change,
#22. 2008_2010_KSI_pop_change,
#23. Total_Budget_2009_pop,
#24. Safety_Budget_2009_pop,
#25. Cyclist_Safety_Budget_2009_pop,
#26. Child_Safety_Budget_2009_pop,
#27. Motorcycle_Safety_Budget_2009_pop,
#28. Drink_Drive_Campaigns_Budget_2009_pop,
#29. Promote_Cycling_Budget_2009_pop,
#30. Promote_Car_Sharing_Budget_2009_pop

data_filename = 'coursework_1_data_2017_v2_input.csv'

# Use the next line to set figure height and width (experiment to check the scale):
#figure_width, figure_height = 4,10


# Read in data
# Need to fix to read in headers and non numeric data eg strings
all_data = pd.read_csv(data_filename, index_col = 'local_authority_area')

# RAW data, non normalised:
# ksi data in columns 2, 3, 4:
ksi_data = all_data[['2008_KSI','2009_KSI', '2010_KSI', 'local_authority_code']]
ksi_data_desc = ksi_data.describe()
print(ksi_data_desc)

#plt.figure(figsize=(figure_width,figure_height))
output_filename = '1_KSI summary'
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ksi_data.boxplot(figsize=(4,2))
ax.set_title('Distribution of local authority KSIs', fontsize = 10)
plt.savefig(output_filename)

output_filename = '2_KSI summary by local authority type'
fig = plt.figure()
ksi_data.boxplot(by = 'local_authority_code', layout=(1,3), figsize=(6,4))
plt.suptitle('Distribution of local authority KSIs by type', fontsize = 10)
plt.savefig(output_filename)

#######################################################

# pop data in columns 5, 6, 7:
#pop_data = all_data[:,4:7]

# budget data (total is in column 8):
#budget_data = all_data[:,7:14]

# data NORMALISED using population:
#norm_ksi_data = all_data[:,15:18]
norm_ksi_data = all_data[['2008_KSI_pop', '2009_KSI_pop', '2010_KSI_pop', 'local_authority_type', 'local_authority_code']]
norm_ksi_data = norm_ksi_data.drop('City of London')
norm_ksi_data_desc = norm_ksi_data.describe()
print(norm_ksi_data_desc)

grouped_norm_ksi_data = norm_ksi_data.groupby('local_authority_type')
grouped_norm_ksi_data_desc = grouped_norm_ksi_data.describe()
print(grouped_norm_ksi_data_desc)

#plt.figure(figsize=(figure_width,figure_height))
output_filename = '3_norm KSI summary'
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
norm_ksi_data.boxplot(figsize=(4,2))
ax.set_title('Distribution of local authority KSIs (per 10,000 population)', fontsize = 10)
plt.savefig(output_filename)

output_filename = '4_norm_KSI summary by local authority type'
fig = plt.figure()
norm_ksi_data.boxplot(by = 'local_authority_code', layout=(1,3), figsize=(6,4))
plt.suptitle('Distribution of local authority KSIs by type (per 10,000 population)', fontsize = 10)
plt.savefig(output_filename)

# budget data (total is in column 8):
#norm_budget_data = all_data[:,22:30]
#23. Total_Budget_2009_pop,
#24. Safety_Budget_2009_pop,
#25. Cyclist_Safety_Budget_2009_pop,
#26. Child_Safety_Budget_2009_pop,
#27. Motorcycle_Safety_Budget_2009_pop,
#28. Drink_Drive_Campaigns_Budget_2009_pop,
#29. Promote_Cycling_Budget_2009_pop,
#30. Promote_Car_Sharing_Budget_2009_pop

#norm_total_budget_data = all_data[['Total_Budget_2009_pop','Safety_Budget_2009_pop', 'Cyclist_Safety_Budget_2009_pop', 'Child_Safety_Budget_2009_pop', 'Motorcycle_Safety_Budget_2009_pop', 'Drink_Drive_Campaigns_Budget_2009_pop', 'Promote_Cycling_Budget_2009_pop','Promote_Car_Sharing_Budget_2009_pop', 'local_authority_type', 'local_authority_code']]

norm_total_budget_data = all_data[['Total', 'Safety_Total', 'local_authority_type', 'local_authority_code']]
norm_total_budget_data = norm_total_budget_data.drop('City of London')
norm_total_budget_data_desc = norm_total_budget_data.describe()
print(norm_total_budget_data_desc)

grouped_norm_total_budget_data = norm_total_budget_data.groupby('local_authority_type')
grouped_norm_total_budget_data_desc = grouped_norm_total_budget_data.describe()
print(grouped_norm_total_budget_data_desc)

#plt.figure(figsize=(figure_width,figure_height))
output_filename = '5_norm budget summary'
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
norm_total_budget_data.boxplot(figsize=(4,2))
ax.set_title('Distribution of local authority budgets per head of population', fontsize = 10)
plt.savefig(output_filename)

output_filename = '6_norm budget summary by local authority type'
fig = plt.figure()
norm_total_budget_data.boxplot(by = 'local_authority_code', figsize=(6,4))
plt.suptitle('Distribution of local authority budgets per head of population by type', fontsize = 10)
plt.savefig(output_filename)

#norm_budget_data = all_data[['Cyclist_Safety_Budget_2009_pop', 'Child_Safety_Budget_2009_pop', 'Motorcycle_Safety_Budget_2009_pop', 'Drink_Drive_Campaigns_Budget_2009_pop', 'Promote_Cycling_Budget_2009_pop','Promote_Car_Sharing_Budget_2009_pop', 'local_authority_type', 'local_authority_code']]

norm_budget_data = all_data[['CyS', 'ChS', 'McS', 'DDC', 'PrC','PCS', 'local_authority_type', 'local_authority_code']]
norm_budget_data = norm_budget_data.drop('City of London')
norm_budget_data_desc = norm_budget_data.describe()
print(norm_budget_data_desc)

grouped_norm_budget_data = norm_budget_data.groupby('local_authority_type')
grouped_norm_budget_data_desc = grouped_norm_budget_data.describe()
print(grouped_norm_budget_data_desc)

#plt.figure(figsize=(figure_width,figure_height))
output_filename = '7_norm budget summary'
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
norm_budget_data.boxplot(figsize=(4,2))
ax.set_title('Distribution of local authority budgets per head of population', fontsize = 10)
plt.savefig(output_filename)

output_filename = '8_norm budget summary by local authority type'
fig = plt.figure()
norm_budget_data.boxplot(by = 'local_authority_code', figsize=(6,8))
plt.suptitle('Distribution of local authority budgets per head of population by type', fontsize = 10)
plt.savefig(output_filename)

# normalised ksi CHANGE
#norm_ksi_change_0809_data = all_data[:,20]
#norm_ksi_change_0910_data = all_data[:,21]
#norm_ksi_change_0810_data = all_data[:,22]

norm_ksi_change_data = all_data[['08_09_KSI_change','09_10_KSI_change', '2008_2010_KSI_pop_change', 'local_authority_type', 'local_authority_code']]
norm_ksi_change_data = norm_ksi_change_data.drop('City of London')
norm_ksi_change_data_desc = norm_ksi_change_data.describe()
print(norm_ksi_data_desc)

grouped_norm_ksi_change_data = norm_ksi_change_data.groupby('local_authority_type')
grouped_norm_ksi_change_data_desc = grouped_norm_ksi_change_data.describe()
print(grouped_norm_ksi_change_data_desc)

#plt.figure(figsize=(figure_width,figure_height))
output_filename = '9_norm KSI change summary'
fig = plt.figure()
norm_ksi_change_data.boxplot(figsize=(4,2))
#fig.set_title('Distribution of change in local authority KSIs (%)', fontsize = 10)
plt.savefig(output_filename)

output_filename = '10_norm KSI change summary by local authority type'
fig = plt.figure()
norm_ksi_change_data.boxplot(by = 'local_authority_code', figsize=(6,6))
#plt.suptitle('Distribution of change in local authority KSIs by type (%)', fontsize = 10)
plt.savefig(output_filename)

###########################################################

#HISTOGRAM
output_filename = '11_norm KSI change HIST'
fig = plt.figure()
norm_ksi_change_data.hist(figsize=(6,4), bins = 15)
plt.suptitle('Distribution of change in local authority KSIs (%)', fontsize = 10)
plt.savefig(output_filename)

#############################################################
# Scatters

##############################################################

#KSI are best predicted by previous years KSIs

import seaborn as sns

all_data_noCoL = all_data.drop('City of London')

output_filename = '12a_previous KSIs as a predictor of subsequent KSIs_scatter_plot'
fig = plt.figure()
sns.pairplot(x_vars=['2008_KSI_pop'], y_vars=['2010_KSI_pop'], data=all_data_noCoL, hue='local_authority_code', size=5)
plt.suptitle('Previous KSIs as a predictor of subsequent KSIs', fontsize = 10)
plt.savefig(output_filename)

#are KSIs good predictor of subsequent change in KSIs?

output_filename = '12b_previous KSIs as a predictor of KSI change_scatter_plot'
fig = plt.figure()
sns.pairplot(x_vars=['2008_KSI_pop'], y_vars=['09_10_KSI_change'], data=all_data_noCoL, hue='local_authority_code', size=5)
plt.suptitle('Previous KSIs as a predictor of subsequent change in KSIs', fontsize = 10)
plt.savefig(output_filename)

#############################################################

#Is change in KSIs based on previous spend
# %change in 2009 KSIs vs budget

output_filename = '13_change 2009 KSIs vs budget'
fig = plt.figure()
sns.pairplot(x_vars=['Total'], y_vars=['09_10_KSI_change'], data=all_data_noCoL, hue='local_authority_code', size=5)
plt.suptitle('Change in 2009 KSIs (per 10,000 pop.) vs total budget (per pop.)', fontsize = 10)
plt.savefig(output_filename)

# %change in 2009 KSIs vs safety budget

output_filename = '13a_change 2009 KSIs vs budget'
fig = plt.figure()
sns.pairplot(x_vars=['Safety_Total'], y_vars=['09_10_KSI_change'], data=all_data_noCoL, hue='local_authority_code', size=5)
plt.suptitle('Change in 2009 KSIs (per 10,000 pop.)` vs safety budget (per pop.)', fontsize = 10)
plt.savefig(output_filename)

# Is spend based on previous KSI data
# budget vs 2008 KSIs

output_filename = '14_budget vs 2008 KSIs'
fig = plt.figure()
sns.pairplot(x_vars=['2008_KSI_pop'], y_vars=['Total'], data=all_data_noCoL, hue='local_authority_code', size=5)
plt.suptitle('total budget (per pop.) vs 2008 KSIs (per 10,000 pop.)', fontsize = 10)
plt.savefig(output_filename)

# safety budget vs 2008 KSIs

output_filename = '14a_safety budget vs 2008 KSIs'
fig = plt.figure()
sns.pairplot(x_vars=['2008_KSI_pop'], y_vars=['Safety_Total'], data=all_data_noCoL, hue='local_authority_code', size=5)
plt.suptitle('safety budget (per pop.) vs 2008 KSIs (per 10,000 pop.)', fontsize = 10)
plt.savefig(output_filename)

############################################################

# scatter matrix to predict 2010 KSIs

predict_2010_KSI_data = all_data_noCoL[['09_10_KSI_change', 'CyS', 'ChS','McS', 'DDC', 'local_authority_code']]

output_filename = '15_Budget as a predictor of subsequent change in KSIs scatter matrix'

fig = sns.pairplot(vars=['09_10_KSI_change', 'CyS', 'ChS','McS', 'DDC'], data=predict_2010_KSI_data, hue='local_authority_code', size=3)

plt.subplots_adjust(top=0.95)
fig.fig.suptitle('Budget as a predictor of subsequent change in KSIs', fontsize = 16)

fig.fig.get_children()[-1].set_bbox_to_anchor((0.95, 0.95, 0, 0))


plt.savefig(output_filename)

# scatter matrix to predict 2009 Budget NOTE that total and normalised KSIs considered

predict_2009_budget_data = all_data_noCoL[['2008_KSI', '2008_KSI_pop', 'CyS', 'ChS','McS', 'DDC', 'local_authority_code']]

output_filename = '16_KSIs as a predictor of subsequent budget scatter matrix'

fig = sns.pairplot(vars=['2008_KSI', '2008_KSI_pop', 'CyS', 'ChS','McS', 'DDC'], data=predict_2009_budget_data, hue='local_authority_code', size=3)

plt.subplots_adjust(top=0.95)
fig.fig.suptitle('KSIs as a predictor of subsequent budget', fontsize = 16)

fig.fig.get_children()[-1].set_bbox_to_anchor((0.95, 0.95, 0, 0))

plt.savefig(output_filename)

# scatter matrix to predict analyse relationship between different types of expenditure

predict_2009_budget_type_data = all_data_noCoL[['CyS', 'ChS','McS', 'DDC','PrC', 'PCS', 'local_authority_code']]

output_filename = '17_2009 budget type scatter matrix'

fig = sns.pairplot(vars=['CyS', 'ChS','McS', 'DDC', 'PrC', 'PCS'], data=predict_2009_budget_type_data, hue='local_authority_code', size=3)

plt.subplots_adjust(top=0.95)
fig.fig.suptitle('Budget type scatter matrix', fontsize = 16)

fig.fig.get_children()[-1].set_bbox_to_anchor((0.95, 0.95, 0, 0))

fig.savefig(output_filename)

# KPI BOX PLOTS
# This line creates the figure. 

#kpi_bp = ksi_data.plot.box()

# Uncomment the next three lines to set the axis limits (otherwise they will be set automatically):
#axis_min = 0.95
#axis_max = 4.05
#plt.ylim([axis_min,axis_max])

# The next lines create and save the plot:
#plt.xlim([0.75,1.25])
#plt.xticks([])
#plt.boxplot(data,manage_xticks=False)

#plt.savefig(output_filename)