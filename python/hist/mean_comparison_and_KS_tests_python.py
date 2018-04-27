# This program performs a mean comparison t-test and a KS test on data stored in a csv file.
# [Or rather, it reports the p-values of these two tests.]

# The data file must be two columns of numbers - no column labels, etc.
# The mean comparison test assumes roughly equal variance of the two samples.

# This data must be saved as a csv file (e.g. use "Save As" in Excel and choose csv format).
# It must be saved in the same folder as this program.

# See the file hyp_test_data.csv for reference.
# [Note that this is the walking speed data from earlier in the course...
# ... the first column is female subjects, the second column male subjects.]

# In the next line, replace demo_multreg_data_example.csv with the filename of your data:
data_filename = 'hyp_test_data.csv'

# The next lines import the necessary packages to perform the regression:
import scipy.stats as sps
import numpy as np
import math as ma

data = np.genfromtxt(data_filename,delimiter = ',')

# If there are errors importing the data, you can also copy it in as follows:
# e.g. data = [[1.5, 1.6],
#              [1.1, 1.6],
#              [1.2, 1.3],
#              ...
#              ['nan', 1.4]]
# (Compare this example with the file hyp_test_data.csv...
# ... Note that 'nan' indicates an empty cell.)

# These lines extract the two data series from the data:
x1_values = data[:,0]
x2_values = data[:,1]

# These lines remove blank entries:
x1_values = np.array([a for a in x1_values if not np.isnan(a)])
x2_values = np.array([a for a in x2_values if not np.isnan(a)])

# These lines perform the comparison of means t-test:
tval, pval_mct = sps.ttest_ind(x1_values,x2_values,equal_var=True)

# These lines perform the KS test:
KSstat, pval_KS = sps.ks_2samp(x1_values,x2_values)

# Print the p-values:
print("The mean comparison t-test p-value =", pval_mct)
print("The KS test p-value =", pval_KS)
print("Remember that you must compare these values to your PRE-DECIDED significance level to draw a conclusion.")
