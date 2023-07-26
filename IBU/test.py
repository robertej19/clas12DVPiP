import numpy as np

# Assuming arrays is your list of n 1xd numpy arrays. For example:
arrays = [
    np.array([0.2, 0.3, np.nan, 0.5]),
    np.array([0.4, np.nan, 0.1, 0.5]),
    np.array([np.nan, 0.5, np.nan, 0.5])
]

# Convert list of arrays into a 2D numpy array
arrays_2d = np.array(arrays)

# Compute the mean of each column (i.e., each element across arrays), ignoring nan values
averages = np.nanmean(arrays_2d, axis=0)

# Normalize the averages so they sum to 1
output = averages / np.nansum(averages)

print(output)