#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 20:59:53 2017

@author: fredshone
"""

# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

data_filename = 'data.csv'
data = pd.read_csv(data_filename, index_col = 'ward_code')

##NOTE Consider removing Welsh Harp Ward (Borough of Brent), due to outlier speed change (>10km/hr)
data = data.drop('E05000103')

#MODE SHARE
select = 'mode_share'

mode_share_summary = data[[select]].describe()

output_filename = '1_mode_share_summary'
fig = plt.figure()
data[[select]].boxplot(figsize=(4,2))
#plt.title('Distribution of ward bus mode shares', fontsize = 10)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
plt.savefig(output_filename)

output_filename = '2_mode_share_summary_by_ward'
fig = plt.figure()
data[['Borough', select]].boxplot(by = 'Borough', figsize=(6,4))
plt.xticks(rotation='vertical')
plt.margins(0.2)
plt.subplots_adjust(bottom=0.4)
#plt.suptitle('Distribution of ward bus mode shares by borough', fontsize = 10)
plt.savefig(output_filename)

output_filename = '3_mode_share_histogram'
fig = plt.figure()
data[[select]].hist(figsize=(6,4), bins = 25)
plt.suptitle('Distribution of change in local authority KSIs (%)', fontsize = 10)
plt.savefig(output_filename)

#PTAL
select =['PTAL_bus2015', 'PTAL_LUL2015', 'PTAL_rail2015', 'PTAL_tram2015', 'PTAL_AI2015']

PTAL_summary = data[select].describe()

select =['PTAL_bus2015', 'PTAL_LUL2015', 'PTAL_rail2015', 'PTAL_tram2015']

output_filename = '4_PTAL_summary'
fig = plt.figure()
data[select].boxplot(figsize=(4,2))
#plt.title('Distribution of ward bus mode shares', fontsize = 10)
plt.savefig(output_filename)

output_filename = '5_PTAL_AI_summary_by_borough'
fig = plt.figure()
data[['Borough', 'PTAL_AI2015']].boxplot(by = 'Borough', figsize=(6,4))
plt.xticks(rotation='vertical')
plt.margins(0.2)
plt.subplots_adjust(bottom=0.4)
#plt.suptitle('Distribution of ward bus mode shares by borough', fontsize = 10)
plt.savefig(output_filename)

select =['PTAL_bus2015', 'PTAL_LUL2015', 'PTAL_rail2015', 'PTAL_tram2015', 'PTAL_AI2015']

output_filename = '6_PTAL_histogram'
fig = plt.figure()
data[select].hist(figsize=(8,8), bins = 25)
plt.suptitle('Distribution of change in local authority KSIs (%)', fontsize = 10)
plt.savefig(output_filename)

#scatter PTAL vs modeshare

output_filename = '6a_vsBusPTAL'
fig = plt.figure()
sns.pairplot(x_vars=['PTAL_bus2015'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs 2015 Bus PTAL', fontsize = 10)
plt.savefig(output_filename)

output_filename = '6b_vsLULPTAL'
fig = plt.figure()
sns.pairplot(x_vars=['PTAL_LUL2015'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs 2015 London Underground PTAL', fontsize = 10)
plt.savefig(output_filename)

output_filename = '6c_vsRailPTAL'
fig = plt.figure()
sns.pairplot(x_vars=['PTAL_rail2015'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs 2015 Rail PTAL', fontsize = 10)
plt.savefig(output_filename)

output_filename = '6d_vsTramPTAL'
fig = plt.figure()
sns.pairplot(x_vars=['PTAL_tram2015'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs 2015 Tram PTAL', fontsize = 10)
plt.savefig(output_filename)

output_filename = '6e_vsAIPTAL'
fig = plt.figure()
sns.pairplot(x_vars=['PTAL_AI2015'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs 2015 Combined PTAL (AI)', fontsize = 10)
plt.savefig(output_filename)

#BUS SPEED
select =['speed_weekday_allday', 'speed_weekday_AM','speed_weekday_IN']

Bus_speed_summary = data[select].describe()

output_filename = '7_Bus_speed_summary'
fig = plt.figure()
data[select].boxplot(figsize=(4,2))
#plt.title('Distribution of ward bus mode shares', fontsize = 10)
plt.savefig(output_filename)

#output_filename = '8_Bus_speed_summary_by_borough'
#fig = plt.figure()
#data[['Borough', 'speed_weekday_allday', 'speed_weekday_AM','speed_weekday_IN']].boxplot(by = 'Borough', figsize=(6,9))
#plt.xticks(rotation='vertical')
#plt.margins(0.2)
#plt.subplots_adjust(bottom=0.15)
##plt.suptitle('Distribution of ward bus mode shares by borough', fontsize = 10)
#plt.savefig(output_filename)

output_filename = '9_Bus_speed_histogram'
fig = plt.figure()
data[select].hist(figsize=(8,6), bins = 25)
plt.suptitle('', fontsize = 10)
plt.savefig(output_filename)

output_filename = '9a_vsspeed_weekday_allday'
fig = plt.figure()
sns.pairplot(x_vars=['speed_weekday_allday'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs Bus Speed (all day)', fontsize = 10)
plt.savefig(output_filename)

output_filename = '9b_vsspeed_weekday_AM'
fig = plt.figure()
sns.pairplot(x_vars=['speed_weekday_AM'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs Bus Speed (AM)', fontsize = 10)
plt.savefig(output_filename)

output_filename = '9c_vsspeed_weekday_IN'
fig = plt.figure()
sns.pairplot(x_vars=['speed_weekday_IN'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs Bus Speed (inter-peak)', fontsize = 10)
plt.savefig(output_filename)

#BUS SPEED CHANGE 
select =['change_weekday_allday', 'change_weekday_AM',	'change_weekday_IN']

Bus_speed_change_summary = data[select].describe()

output_filename = '10_Bus_speed_change_summary'
fig = plt.figure()
data[select].boxplot(figsize=(4,2))
#plt.title('Distribution of ward bus mode shares', fontsize = 10)
plt.savefig(output_filename)

##NOTE Consider removing Welsh Ward (Borough of Brent), due to outlier speed change (>10km/hr)

#output_filename = '8_Bus_speed_change_summary_by_borough'
#fig = plt.figure()
#data[['Borough', 'speed_weekday_allday', 'speed_weekday_AM','speed_weekday_IN']].boxplot(by = 'Borough', figsize=(6,9))
#plt.xticks(rotation='vertical')
#plt.margins(0.2)
#plt.subplots_adjust(bottom=0.15)
##plt.suptitle('Distribution of ward bus mode shares by borough', fontsize = 10)
#plt.savefig(output_filename)

output_filename = '11_Bus_speed_change_histogram'
fig = plt.figure()
data[select].hist(figsize=(8,6), bins = 25)
plt.suptitle('', fontsize = 10)
plt.savefig(output_filename)

output_filename = '11a_vs_change_weekday_allday'
fig = plt.figure()
sns.pairplot(x_vars=['change_weekday_allday'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs Bus Speed Change (all day)', fontsize = 10)
plt.savefig(output_filename)

output_filename = '11b_vs_change_weekday_allday'
fig = plt.figure()
sns.pairplot(x_vars=['change_weekday_AM'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs Bus Speed Change (AM)', fontsize = 10)
plt.savefig(output_filename)

output_filename = '11c_vs_change_weekday_allday'
fig = plt.figure()
sns.pairplot(x_vars=['change_weekday_IN'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs Bus Speed Change (inter-peak)', fontsize = 10)
plt.savefig(output_filename)

#BUS ODX DATA 
select =['weekday_boardings_allday']

Bus_ODX_boarding_summary = data[select].describe()

output_filename = '12_Bus_ODX_Boardings_summary'
fig = plt.figure()
data[select].boxplot(figsize=(4,2))
#plt.title('Distribution of ward bus mode shares', fontsize = 10)
plt.savefig(output_filename)

select =['weekend_boardings_ratio', 'alightings_boardings_ratio']

Bus_ODX_ratios_summary = data[select].describe()

output_filename = '13_Bus_ODX_ratios_summary'
fig = plt.figure()
data[select].boxplot(figsize=(4,2))
#plt.title('Distribution of ward bus mode shares', fontsize = 10)
plt.savefig(output_filename)

#output_filename = '8_Bus_speed_change_summary_by_borough'
#fig = plt.figure()
#data[['Borough', 'speed_weekday_allday', 'speed_weekday_AM','speed_weekday_IN']].boxplot(by = 'Borough', figsize=(6,9))
#plt.xticks(rotation='vertical')
#plt.margins(0.2)
#plt.subplots_adjust(bottom=0.15)
##plt.suptitle('Distribution of ward bus mode shares by borough', fontsize = 10)
#plt.savefig(output_filename)

select =['weekday_boardings_allday', 'weekend_boardings_ratio', 'alightings_boardings_ratio']

output_filename = '14_Bus_speed_change_histogram'
fig = plt.figure()
data[select].hist(figsize=(8,6), bins = 25)
plt.suptitle('', fontsize = 10)
plt.savefig(output_filename)

output_filename = '14a_vs_ODX_Boardings_allday'
fig = plt.figure()
sns.pairplot(x_vars=['weekday_boardings_allday'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs Bus Boardings (all-day)', fontsize = 10)
plt.savefig(output_filename)

output_filename = '14b_vs_ODX_Boardings_Weekend_Ratio_allday'
fig = plt.figure()
sns.pairplot(x_vars=['weekend_boardings_ratio'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs Weekend Baordings Ratio (all-day)', fontsize = 10)
plt.savefig(output_filename)

output_filename = '14c_vs_ODX_Alightings_Ratio_allday'
fig = plt.figure()
sns.pairplot(x_vars=['alightings_boardings_ratio'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs Bus Alightings Ratio (AM)', fontsize = 10)
plt.savefig(output_filename)

#WARD ATTRIBUTE DATA 

select =['2013_population', 'area_km', '2013_pop_density']

Ward_attribute_summary = data[select].describe()

#population
select =['2013_population']

output_filename = '15a_ward_pop_summary'
fig = plt.figure()
data[select].boxplot(figsize=(4,2))
#plt.title('Distribution of ward bus mode shares', fontsize = 10)
plt.savefig(output_filename)

output_filename = '16a_ward_pop_by_borough'
fig = plt.figure()
data[['Borough', '2013_population']].boxplot(by = 'Borough', figsize=(6,6))
plt.xticks(rotation='vertical')
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
#plt.suptitle('Distribution of ward bus mode shares by borough', fontsize = 10)
plt.savefig(output_filename)

output_filename = '15d_vs_2013_Population'
fig = plt.figure()
sns.pairplot(x_vars=['2013_population'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs Ward Population', fontsize = 10)
plt.savefig(output_filename)

#area
select =['area_km']

output_filename = '16d_ward_pop_summary'
fig = plt.figure()
data[select].boxplot(figsize=(4,2))
#plt.title('Distribution of ward bus mode shares', fontsize = 10)
plt.savefig(output_filename)

output_filename = '16b_ward_pop_by_borough'
fig = plt.figure()
data[['Borough', 'area_km']].boxplot(by = 'Borough', figsize=(6,6))
plt.xticks(rotation='vertical')
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
#plt.suptitle('Distribution of ward bus mode shares by borough', fontsize = 10)
plt.savefig(output_filename)

output_filename = '16e_vs_Ward_Area'
fig = plt.figure()
sns.pairplot(x_vars=['area_km'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs Ward Area', fontsize = 10)
plt.savefig(output_filename)

#density
select =['2013_pop_density']

output_filename = '15c_ward_pop_summary'
fig = plt.figure()
data[select].boxplot(figsize=(4,2))
#plt.title('Distribution of ward bus mode shares', fontsize = 10)
plt.savefig(output_filename)

output_filename = '16c_ward_pop_by_borough'
fig = plt.figure()
data[['Borough', '2013_pop_density']].boxplot(by = 'Borough', figsize=(6,6))
plt.xticks(rotation='vertical')
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
#plt.suptitle('Distribution of ward bus mode shares by borough', fontsize = 10)
plt.savefig(output_filename)

output_filename = '16f_vs_Ward_Pop_Density'
fig = plt.figure()
sns.pairplot(x_vars=['2013_pop_density'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs Ward Population Density', fontsize = 10)
plt.savefig(output_filename)


select =['2013_population', 'area_km', '2013_pop_density']

output_filename = '17_Ward_pop_histogram'
fig = plt.figure()
data[select].hist(figsize=(8,6), bins = 25)
plt.suptitle('', fontsize = 10)
plt.savefig(output_filename)

# WARD DEMOGRAPHICS

select = ['pop_0to15_perc', 'pop_16to64_perc', 'pop_65plus_perc', '2011_ethnic_white',	'2011_ethnic_mixed', '2011_ethnic_asian', '2011_ethnic_black']

Ward_age_ethnic_summary = data[select].describe()

# age
select = ['pop_0to15_perc', 'pop_65plus_perc']
output_filename = '18a_ward_age_summary'
fig, ax1 = plt.subplots()
data[select].boxplot(figsize=(4,2))
ax1.set_ylabel('%')
#plt.title('Distribution of ward bus mode shares', fontsize = 10)
plt.savefig(output_filename)

output_filename = '19a_ward_young_by_borough'
fig, ax1 = plt.subplots()
data[['Borough', 'pop_0to15_perc']].boxplot(by = 'Borough', figsize=(6,4))
ax1.set_ylabel('%')
plt.xticks(rotation='vertical')
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
#plt.suptitle('Distribution of ward bus mode shares by borough', fontsize = 10)
plt.savefig(output_filename)

output_filename = '19aa_vs_Ward_Young'
fig = plt.figure()
sns.pairplot(x_vars=['pop_0to15_perc'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs Ward Population Aged Under 16 (%)', fontsize = 10)
plt.savefig(output_filename)

output_filename = '19b_ward_mid_by_borough'
fig, ax1 = plt.subplots()
data[['Borough', 'pop_16to64_perc']].boxplot(by = 'Borough', figsize=(6,4))
ax1.set_ylabel('%')
plt.xticks(rotation='vertical')
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
#plt.suptitle('Distribution of ward bus mode shares by borough', fontsize = 10)
plt.savefig(output_filename)

output_filename = '19bb_vs_Ward_Mid'
fig = plt.figure()
sns.pairplot(x_vars=['pop_16to64_perc'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs Ward Population Aged 16 to 64 (%)', fontsize = 10)
plt.savefig(output_filename)

output_filename = '19c_ward_old_by_borough'
fig, ax1 = plt.subplots()
data[['Borough', 'pop_65plus_perc']].boxplot(by = 'Borough', figsize=(6,4))
ax1.set_ylabel('%')
plt.xticks(rotation='vertical')
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
#plt.suptitle('Distribution of ward bus mode shares by borough', fontsize = 10)
plt.savefig(output_filename)

output_filename = '19cc_vs_Ward_Old'
fig = plt.figure()
sns.pairplot(x_vars=['pop_65plus_perc'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs Ward Population Aged Over 64 (%)', fontsize = 10)
plt.savefig(output_filename)

select = ['pop_0to15_perc', 'pop_16to64_perc', 'pop_65plus_perc']
output_filename = '20_Ward_pop_histogram'
fig = plt.figure()
data[select].hist(figsize=(6,6), bins = 25)
plt.suptitle('', fontsize = 10)
plt.savefig(output_filename)

# ethnicity
select = ['2011_ethnic_white',	'2011_ethnic_mixed', '2011_ethnic_asian', '2011_ethnic_black']

output_filename = '21_ward_ethnicity_summary'
fig, ax1 = plt.subplots()
data[select].boxplot(figsize=(4,2))
ax1.set_ylabel('%')
#plt.title('Distribution of ward bus mode shares', fontsize = 10)
plt.savefig(output_filename)

output_filename = '22a_ward_white_by_borough'
fig, ax1 = plt.subplots()
data[['Borough', '2011_ethnic_white']].boxplot(by = 'Borough', figsize=(6,4))
ax1.set_ylabel('%')
plt.xticks(rotation='vertical')
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
#plt.suptitle('Distribution of ward bus mode shares by borough', fontsize = 10)
plt.savefig(output_filename)

output_filename = '22aa_vs_Ward_White'
fig = plt.figure()
sns.pairplot(x_vars=['2011_ethnic_white'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs Ward Population White Ethnicity (%)', fontsize = 10)
plt.savefig(output_filename)

output_filename = '22b_ward_mixed_by_borough'
fig, ax1 = plt.subplots()
data[['Borough', '2011_ethnic_mixed']].boxplot(by = 'Borough', figsize=(6,4))
ax1.set_ylabel('%')
plt.xticks(rotation='vertical')
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
#plt.suptitle('Distribution of ward bus mode shares by borough', fontsize = 10)
plt.savefig(output_filename)

output_filename = '22bb_vs_Ward_Mixed'
fig = plt.figure()
sns.pairplot(x_vars=['2011_ethnic_mixed'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs Ward Population Mixed Ethnicity (%)', fontsize = 10)
plt.savefig(output_filename)

output_filename = '22c_ward_asian_by_borough'
fig, ax1 = plt.subplots()
data[['Borough', '2011_ethnic_asian']].boxplot(by = 'Borough', figsize=(6,4))
ax1.set_ylabel('%')
plt.xticks(rotation='vertical')
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
#plt.suptitle('Distribution of ward bus mode shares by borough', fontsize = 10)
plt.savefig(output_filename)

output_filename = '22cc_vs_Ward_Asian'
fig = plt.figure()
sns.pairplot(x_vars=['2011_ethnic_asian'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs Ward Population Asian Ethnicity (%)', fontsize = 10)
plt.savefig(output_filename)

output_filename = '22d_ward_black_by_borough'
fig, ax1 = plt.subplots()
data[['Borough', '2011_ethnic_black']].boxplot(by = 'Borough', figsize=(6,4))
ax1.set_ylabel('%')
plt.xticks(rotation='vertical')
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
#plt.suptitle('Distribution of ward bus mode shares by borough', fontsize = 10)
plt.savefig(output_filename)

output_filename = '22dd_vs_Ward_Asian'
fig = plt.figure()
sns.pairplot(x_vars=['2011_ethnic_black'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs Ward Population Black Ethnicity (%)', fontsize = 10)
plt.savefig(output_filename)

select = ['2011_ethnic_white',	'2011_ethnic_mixed', '2011_ethnic_asian', '2011_ethnic_black']
output_filename = '23_Ward_ethnicity_histogram'
fig = plt.figure()
data[select].hist(figsize=(6,6), bins = 25)
plt.suptitle('', fontsize = 10)
plt.savefig(output_filename)

#HOUSE PRICES
select = ['2014_med_house_price']

Ward_house_price_summary = data[select].describe()

output_filename = '24_ward_med_house_price_summary'
fig, ax1 = plt.subplots()
data[select].boxplot(figsize=(4,2))
ax1.set_ylabel('%')
#plt.title('Distribution of ward bus mode shares', fontsize = 10)
plt.savefig(output_filename)

output_filename = '25_ward_med_house_price_by_borough'
fig, ax1 = plt.subplots()
data[['Borough', '2014_med_house_price']].boxplot(by = 'Borough', figsize=(6,4))
ax1.set_ylabel('%')
plt.xticks(rotation='vertical')
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
#plt.suptitle('Distribution of ward bus mode shares by borough', fontsize = 10)
plt.savefig(output_filename)

output_filename = '26_Ward_med_house_price_histogram'
fig = plt.figure()
data[select].hist(figsize=(4,4), bins = 25)
plt.xticks(rotation='vertical')
plt.suptitle('', fontsize = 10)
plt.savefig(output_filename)

output_filename = '26a_vs_Ward_Median_House_Price'
fig = plt.figure()
sns.pairplot(x_vars=['2014_med_house_price'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs Ward Median House Price', fontsize = 10)
plt.savefig(output_filename)

#INCOME
select = ['2014_mean_income']

Ward_income_summary = data[select].describe()

output_filename = '27_ward_2014_mean_income_summary'
fig, ax1 = plt.subplots()
data[select].boxplot(figsize=(4,2))
ax1.set_ylabel('%')
#plt.title('Distribution of ward bus mode shares', fontsize = 10)
plt.savefig(output_filename)

output_filename = '28_ward_2014_mean_income_by_borough'
fig, ax1 = plt.subplots()
data[['Borough', '2014_mean_income']].boxplot(by = 'Borough', figsize=(6,4))
ax1.set_ylabel('%')
plt.xticks(rotation='vertical')
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
#plt.suptitle('Distribution of ward bus mode shares by borough', fontsize = 10)
plt.savefig(output_filename)

output_filename = '29_Ward_2014_mean_income_histogram'
fig = plt.figure()
data[select].hist(figsize=(4,4), bins = 25)
plt.xticks(rotation='vertical')
plt.suptitle('', fontsize = 10)
plt.savefig(output_filename)

output_filename = '29a_vs_Ward_Mean_Income_Price'
fig = plt.figure()
sns.pairplot(x_vars=['2014_mean_income'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs Ward Mean Income', fontsize = 10)
plt.savefig(output_filename)

#CAR OWNERSHIP

select = ['2011_cars_own_density', '2011_cars_per_household']

Ward_car_ownership_summary = data[select].describe()

#cars per km2
output_filename = '30a_ward_car_ownership_density_summary'
fig, ax1 = plt.subplots()
data[['2011_cars_own_density']].boxplot(figsize=(4,2))
ax1.set_ylabel('cars/km2')
#plt.title('Distribution of ward bus mode shares', fontsize = 10)
plt.savefig(output_filename)

#cars/household
output_filename = '30b_ward_car_ownership_summary'
fig, ax1 = plt.subplots()
data[['2011_cars_per_household']].boxplot(figsize=(4,2))
ax1.set_ylabel('cars/household')
#plt.title('Distribution of ward bus mode shares', fontsize = 10)
plt.savefig(output_filename)

output_filename = '31a_ward_car_ownership_density_by_borough'
fig, ax1 = plt.subplots()
data[['Borough', '2011_cars_own_density']].boxplot(by = 'Borough', figsize=(6,4))
ax1.set_ylabel('cars/km2')
plt.xticks(rotation='vertical')
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
#plt.suptitle('Distribution of ward bus mode shares by borough', fontsize = 10)
plt.savefig(output_filename)

output_filename = '31aa_vs_Ward_Car_Ownership_Density'
fig = plt.figure()
sns.pairplot(x_vars=['2011_cars_own_density'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs Ward Car Density (Car Ownership per km)', fontsize = 10)
plt.savefig(output_filename)


output_filename = '31b_ward_car_ownership_by_borough'
fig, ax1 = plt.subplots()
data[['Borough', '2011_cars_per_household']].boxplot(by = 'Borough', figsize=(6,4))
ax1.set_ylabel('cars/household')
plt.xticks(rotation='vertical')
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
#plt.suptitle('Distribution of ward bus mode shares by borough', fontsize = 10)
plt.savefig(output_filename)

output_filename = '31bb_vs_Ward_Household_Car_Ownership'
fig = plt.figure()
sns.pairplot(x_vars=['2011_cars_per_household'], y_vars=['mode_share'], data=data, hue='Zone', size=5)
plt.suptitle('Bus Mode Share vs Ward Household Car Ownership (cars)', fontsize = 10)
plt.savefig(output_filename)

output_filename = '32_Ward_car_histogram'
fig = plt.figure()
data[select].hist(figsize=(6,4), bins = 25)
plt.xticks(rotation='vertical')
plt.suptitle('', fontsize = 10)
plt.savefig(output_filename)

select = ['2014_med_house_price',	'2014_mean_income', '2011_cars_own_density', '2011_cars_per_household']
output_filename = '33_Ward_ethnicity_histogram'
fig = plt.figure()
data[select].hist(figsize=(6,6), bins = 25)
plt.suptitle('', fontsize = 10)
plt.savefig(output_filename)

