#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 21:45:26 2018

@author: fredshone
"""

# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data_filename = 'data.csv'
data = pd.read_csv(data_filename, index_col = 'ward_code')

##NOTE Consider removing Welsh Harp Ward (Borough of Brent), due to outlier speed change (>10km/hr)
data = data.drop('E05000103')

#correlation matric from pandas library
corr = data.corr()

output_filename = '33_Correlation_Matrix'

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 12))

#plt.margins(0.1)
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.15)


# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.savefig(output_filename)


#Groups
#standardized using min max
#boroughs

#cmap = sns.color_palette("RdBu_r", 7)

output_filename = '34_Borough_Summary_Heatmap'
f, ax = plt.subplots(figsize=(10, 10))

plt.subplots_adjust(bottom=0.2)
plt.subplots_adjust(left=0.2)

borough_summary = data.groupby('Borough').mean().sort_values(by = 'mode_share', ascending = False)
normalized_borough_summary=(borough_summary-borough_summary.min())/(borough_summary.max()-borough_summary.min())
sns.heatmap(normalized_borough_summary, center=0, cmap = cmap, square=False, linewidths=0.5, cbar_kws={"shrink": .5})
plt.savefig(output_filename)

#Zones

output_filename = '35_Zone_Summary_Heatmap'
f, ax = plt.subplots(figsize=(12, 4))

plt.subplots_adjust(bottom=0.25)
plt.subplots_adjust(left=0.25)
zone_summary = data.groupby('Zone').mean()
normalized_zone_summary=(zone_summary-zone_summary.min())/(zone_summary.max()-zone_summary.min())
sns.heatmap(normalized_zone_summary, center=0, cmap = cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig(output_filename)