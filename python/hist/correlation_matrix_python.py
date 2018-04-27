# This program creates a correlation matrix from data stored in a csv file.
# It then saves the correlation matrix as a csv file.

# The data file must be in columns of numbers - no column labels, etc.
# Each column is a data series, X0, X1, X2, etc.

# This data must be saved as a csv file (e.g. use "Save As" in Excel and choose csv format).
# It must be saved in the same folder as this program.
# See the file demo_multreg_data_example.csv for reference.

# In the next line, replace demo_multreg_data_example.csv with the filename of your data:
data_filename = 'demo_multreg_data_example.csv'

# In the next line, replace correlation_matrix with the filename you wish to save as:
output_filename = 'correlation_matrix.csv'

# The next lines import the necessary package to create the correlation analysis:
import numpy as np

data = np.genfromtxt(data_filename,delimiter = ',')

# If there are errors importing the data, you can also copy it in as follows:
# e.g. data = [[737.4776314, 34, 65],
#              [869.2063792, 57, 73],
#              [1033.705248, 59, 100],
#              ...
#              [737.5129466, 66, 49]]
# (Compare this example with the file demo_multreg_data_example.csv)

# These lines create the correlation matrix and print it:
correlation_matrix = np.corrcoef(data.T)
print(correlation_matrix)

# correlation_matrix[i,j] is the correlation between the series Xi and Xj.

# The next line saves the correlation matrix as a csv file:
np.savetxt(output_filename,correlation_matrix,delimiter=',')