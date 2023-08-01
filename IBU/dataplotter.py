import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
import sys

unfolded_data = np.load('final_unfolded_data.npy')
output_matrix = np.load('final_output_matrix.npy')
output_sys_err_matrix = np.load('final_output_sys_err_matrix.npy')
output_stat_err_matrix = np.load('final_output_stat_err_matrix.npy')
truth_data = np.load('final_truth_data.npy')
bins = np.load('bins.npy')
total_unrolled_number_of_bins = np.load('total_unrolled_number_of_bins.npy')
observed_data = np.load('final_observed_data.npy')


# # Create a boolean mask where entries are True if truth_data is not equal to 0
# mask = truth_data != 0 

# # Apply the mask to all the data arrays
# truth_data = truth_data[mask]
# unfolded_data = unfolded_data[mask]
# observed_data = observed_data[mask]
# bins = bins[mask]
# output_sys_err_matrix = output_sys_err_matrix[mask]
# output_stat_err_matrix = output_stat_err_matrix[mask]

ic(bins)
ic(bins.shape)
# Then, in your plotting code, replace the original data arrays with the masked ones

plt.rcParams.update({'font.size': 20})

#bins = np.linspace(0,len(bins),len(bins))
#fig, ax = plt.subplots()
#make plot be 14,10
fig, ax = plt.subplots(figsize=(14,10))
ax.step(bins, truth_data, where='mid', lw=3, alpha=0.7, label='True distribution',color='green')
ax.step(bins, observed_data, where='mid', lw=3, alpha=0.7, label='Observed distribution',color='red')
ax.errorbar(bins, unfolded_data,
        yerr=np.sqrt(np.diag(output_sys_err_matrix/3)**2+np.diag(output_stat_err_matrix)**2),
        #yerr=np.diag(output_sys_err_matrix)/4,
        alpha=0.7,
        elinewidth=3,
        capsize=4,
        color='blue',
        ls='None', marker='.', ms=10,
        label='Unfolded distribution')
ax.set_xlim(0, np.max(bins))
ax.set_ylim(0,11_000)
#make x and y labels
ax.set(xlabel='Unrolled Bin Number', ylabel='Counts')
#make title
ax.set_title('Observed, Unfolded, and True Bin Counts')
plt.show()
#sys.exit()
#save the figure
fig.savefig('final_observed_unfolded_and_true_bin_counts.png')

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
import sys


observed_err = observed_data/truth_data
unfolded_err = unfolded_data/truth_data
difference = observed_err-unfolded_err

errbar = (np.sqrt(np.diag(output_sys_err_matrix/3)**2+np.diag(output_stat_err_matrix)**2)/unfolded_data)*unfolded_err



#create another plot for residuals
#fig, ax = plt.subplots()
fig, ax = plt.subplots(figsize=(14,10))

#set fontsize to be 20
ax.errorbar(bins,observed_err ,
                alpha=1,
                elinewidth=3,
                capsize=4,
                #make color be red
                color='red',
                ls='None', marker='.', ms=10,
                label='Observed Distribution')
ax.errorbar(bins, unfolded_err,
                alpha=0.2,
                yerr=errbar,
                elinewidth=3,
                #make color be blue
                color='blue',
                capsize=4,
                ls='None', marker='.', ms=10,
                label='Unfolded Distribution')
ax.set(xlabel='Unrolled Bin Number', ylabel='Ratio of Counts to Truth Counts')
#set y axis to log scale
#ax.set_yscale('log')
#set y limit
#set title
ax.set_title('Ratio of Observed and Unfolded Events to Truth Across Bins')
ax.set_ylim(0,2)
ax.set_xlim(0,max(bins))
plt.legend()
plt.show()
sys.exit()
#save figure
fig.savefig('ratio_of_observed_and_unfolded_events_to_truth_across_bins.png')
plt.close()
# fig, ax = plt.subplots()
# #set fontsize to be 20
# ax.errorbar(bins, difference,
#                 alpha=0.2,
#                 elinewidth=3,
#                 #make color be blue
#                 color='blue',
#                 capsize=4,
#                 ls='None', marker='.', ms=10,
#                 label='Unfolded distribution')
# ax.set(xlabel='Unrolled Bin Number', ylabel='Percent Error from Truth')
# #set y axis to log scale
# #ax.set_yscale('log')
# #set y limit
# ax.set_ylim(-50,50)
# plt.legend()
# plt.show()


#create a histogram of unfolded_err
#fig, ax = plt.subplots()
#creat fig,ax of size 10,10
fig, ax = plt.subplots(figsize=(14,10))
#histogram range from 0.1 to 10
ax.hist(observed_err, bins=50, alpha=0.5, label='Observed distribution', range=(.75,1.25),color='red')
ax.hist(unfolded_err, bins=50, alpha=0.5, label='Unfolded distribution', range=(.75,1.25),color='blue')
#set xlim 0.75 to 1.25
ax.set_xlim(.75,1.25)
ax.set(ylabel='Number of Bins', xlabel='Ratio of Observed and Unfolded Events to Truth')
#set title
ax.set_title('Histogram of Ratio of Observed and Unfolded Events to Truth')
#set y axis to log scale
#ax.set_yscale('log')
#set y limit
#ax.set_ylim(0.01,10)
plt.legend()

#savefigure
fig.savefig('histogram_of_ratio_of_observed_and_unfolded_events_to_truth.png')

# #make a histogram of difference
# fig, ax = plt.subplots()
# #histogram range from 0.1 to 10
# ax.hist(difference, bins=100, alpha=0.7, label='Difference distribution', range=(-100,100))
# #set log scale
# #ax.set_yscale('log')
# plt.show()

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib as mpl

# fig, ax = plt.subplots()

# # Store the returned image in a variable
# img = ax.hist2d(observed_err, unfolded_err, bins=25, alpha=0.7, cmap='viridis', range=((0,25),(0,25)),norm=mpl.colors.LogNorm())

# # Create a colorbar
# cbar = fig.colorbar(img[3], ax=ax)
# cbar.set_label('Counts per bin')

# plt.show()
