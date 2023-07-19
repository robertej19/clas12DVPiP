import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

df = pd.read_pickle("t_dep_of_xsec.pkl")

# Define the data and group it
groups = df.groupby(['xmin', 'qmin'])

# Create a plot for each group
for name, group in groups:
    plt.figure(figsize=(10, 6)) # Adjust the size as needed

    # Define the values and errors to plot
    values_to_plot = ['A', 'B', 'C', 'c6_tel', 'c6_tt', 'c6_lt']
    errors_to_plot = ['A_uncert', 'B_uncert', 'C_uncert', 'c6_tel_err', 'c6_tt_err', 'c6_lt_err']
    x_axes = ['tave', 'tave', 'tave', 'tave_c6', 'tave_c6', 'tave_c6']
    colors = ['blue', 'red', 'green', 'blue', 'red', 'black']
    linestyles = ['-', '-', '-', '--', '--', '--']

    # Create each line on the plot
    for val, err, x_axis, color, linestyle in zip(values_to_plot, errors_to_plot, x_axes, colors, linestyles):
        plt.errorbar(group[x_axis], group[val], yerr=group[err], fmt='o', color=color, linestyle=linestyle, label=val)
    
    plt.title(f'xmin={name[0]}, qmin={name[1]}')
    plt.xlabel('tave / tave_c6')
    plt.legend()
    
    # Show the plot
    #plt.show()
    #plt.show()
    plt.savefig("tdep_test/"+f'figure_{name[0]}_{name[1]}.png')
    plt.close()