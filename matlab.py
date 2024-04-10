import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# Load data from the file S00.mat
mat_data = scipy.io.loadmat('S00.mat')

# Extract the first row of X, Y, and I
X_first_row = mat_data['X'][0]
Y_first_row = mat_data['Y'][0]
I_first_row = mat_data['I'][0]

# Set values in Y outside the range [0, 5] to 0
Y_first_row[(Y_first_row < 0) | (Y_first_row > 5)] = 0

# Combine I_first_row, X_first_row, and Y_first_row into a matrix
data = np.vstack((I_first_row, X_first_row, Y_first_row)).T

# Extract the columns for plotting
x_data = data[:, 0]
y1_data = data[:, 1]
y2_data = data[:, 2]

# Filter the data where the values in the third column are greater than 2
indices = np.where(y2_data > 2)[0]

# Number of points before and after each instance
num_points = 750

# Randomly select one index
random_index = np.random.choice(indices, size=1, replace=False)[0]

# Calculate the indices for the range of points
range_start = max(0, random_index - num_points)
range_end = min(len(x_data), random_index + num_points)

# Calculate time difference from the selected index to each point
time_diff = x_data - x_data[random_index]

# Extract the range of points for the selected instance
filtered_x_data = time_diff[range_start:range_end]
filtered_y1_data = y1_data[range_start:range_end]

# Create a figure for the plot
plt.figure()

# Plot y1_data versus x_data for the selected instance
plt.plot(filtered_x_data, filtered_y1_data, '-', label='Y1 versus X (Filtered)')

plt.xlabel('Seconds')
plt.ylabel('Y1')
plt.title('Plot of Y1 versus X from Data File')
plt.legend(loc='best')
plt.grid(True)
plt.show()
