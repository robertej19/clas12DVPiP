import numpy as np
import matplotlib.pyplot as plt

# Create an array of x values from 4 to 400
x = np.linspace(2, 650, 500)

# Define the function
y = ((x**4) / (1024**3)) * 8


# Create specific points
x_points = np.array([81, 256, 625])
y_points = ((x_points**4) / (1024**3)) * 8
markers = ['o', '^', 's']  # circle, triangle, square

#set font to size 20
plt.rcParams.update({'font.size': 20})


plt.figure(figsize=(14,10))
plt.plot(x, y)#, label='Total Number of Bins')
for xp, yp, m in zip(x_points, y_points, markers):
    plt.scatter(xp, yp, marker=m, s=300, label=f'{int(np.sqrt(np.sqrt(xp)))} Bins in Each Dimension')  # plot each point with a different marker

plt.xlabel('Number of Bins Total', fontsize=20)  # increase fontsize here
plt.ylabel('Memory Required (GB)', fontsize=20)  # increase fontsize here
plt.title('Memory Required vs. Number of Bins', fontsize=20)  # increase fontsize here
plt.grid(True, which="both", ls="--")
plt.legend()
#plt.show()

#save the figure
plt.savefig('memory_required_vs_number_of_bins_in_each_dim.png')