import numpy as np
import matplotlib.pyplot as plt

unfolded_data = np.load('final_unfolded_data.npy')
output_matrix = np.load('final_output_matrix.npy')
output_sys_err_matrix = np.load('final_output_sys_err_matrix.npy')
output_stat_err_matrix = np.load('final_output_stat_err_matrix.npy')
truth_data = np.load('final_truth_data.npy')
bins = np.load('bins.npy')
total_unrolled_number_of_bins = np.load('total_unrolled_number_of_bins.npy')
observed_data = np.load('final_observed_data.npy')


# Create a boolean mask where entries are True if truth_data is not equal to 0
mask = truth_data != 0 

# Apply the mask to all the data arrays
truth_data = truth_data[mask]
unfolded_data = unfolded_data[mask]
observed_data = observed_data[mask]
bins = bins[mask]
output_sys_err_matrix = output_sys_err_matrix[mask]
output_stat_err_matrix = output_stat_err_matrix[mask]

# Then, in your plotting code, replace the original data arrays with the masked ones

# fig, ax = plt.subplots()
# ax.step(bins_masked, truth_data_masked, where='mid', lw=3, alpha=0.7, label='True distribution')
# ax.step(bins_masked, observed_data_masked, where='mid', lw=3, alpha=0.7, label='Observed distribution')
# ax.errorbar(bins_masked, unfolded_data_masked,
#         yerr=np.sqrt(np.diag(output_sys_err_matrix_masked)**2+np.diag(output_stat_err_matrix_masked)**2),
#         alpha=0.7,
#         elinewidth=3,
#         capsize=4,
#         ls='None', marker='.', ms=10,
#         label='Unfolded distribution')
# ax.set_xlim(0, np.max(bins_masked))
# ax.set_ylim(0,8000)
# plt.show()


# fig, ax = plt.subplots()
# ax.step(bins, truth_data, where='mid', lw=3,
#         alpha=0.7, label='True distribution')
# ax.step(bins, observed_data, where='mid', lw=3,
#         alpha=0.7, label='Observed distribution')
# ax.errorbar(bins, unfolded_data,
#         #yerr=unfolded_results['sys_err'],
#         alpha=0.7,
#         elinewidth=3,
#         capsize=4,
#         ls='None', marker='.', ms=10,
#         label='Unfolded distribution')

# ax.set(xlabel='X bins', ylabel='Counts')
# plt.legend()
# plt.show()

observed_err = np.abs(observed_data-truth_data)/truth_data*100
unfolded_err = np.abs(unfolded_data-truth_data)/truth_data*100
difference = observed_err-unfolded_err

plt.rcParams.update({'font.size': 20})

#create another plot for residuals
fig, ax = plt.subplots()
#set fontsize to be 20
ax.errorbar(bins,observed_err ,
                alpha=1,
                elinewidth=3,
                capsize=4,
                #make color be red
                color='red',
                ls='None', marker='.', ms=10,
                label='Observed distribution')
ax.errorbar(bins, unfolded_err,
                alpha=0.2,
                elinewidth=3,
                #make color be blue
                color='blue',
                capsize=4,
                ls='None', marker='.', ms=10,
                label='Unfolded distribution')
ax.set(xlabel='Unrolled Bin Number', ylabel='Percent Error from Truth')
#set y axis to log scale
#ax.set_yscale('log')
#set y limit
ax.set_ylim(0,50)
plt.legend()
plt.show()

#create a histogram of unfolded_err
fig, ax = plt.subplots()
#histogram range from 0.1 to 10
ax.hist(observed_err, bins=100, alpha=0.7, label='Observed distribution', range=(0,50),color='red')
ax.hist(unfolded_err, bins=100, alpha=0.7, label='Unfolded distribution', range=(0,50),color='blue')

ax.set(ylabel='Counts', xlabel='Percent Error from Truth')
#set y axis to log scale
#ax.set_yscale('log')
#set y limit
#ax.set_ylim(0.01,10)
plt.legend()
plt.show()

#make a histogram of difference
fig, ax = plt.subplots()
#histogram range from 0.1 to 10
ax.hist(difference, bins=100, alpha=0.7, label='Difference distribution', range=(-100,100))
#set log scale
#ax.set_yscale('log')
plt.show()

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib as mpl

fig, ax = plt.subplots()

# Store the returned image in a variable
img = ax.hist2d(observed_err, unfolded_err, bins=25, alpha=0.7, cmap='viridis', range=((0,25),(0,25)),norm=mpl.colors.LogNorm())

# Create a colorbar
cbar = fig.colorbar(img[3], ax=ax)
cbar.set_label('Counts per bin')

plt.show()
