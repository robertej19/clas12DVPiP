import numpy as np
from pyunfold import iterative_unfold
from scipy.stats import norm
#from pyunfold.priors import uniform_prior
#from pyunfold.priors import jeffreys_prior
from pyunfold import priors
import pandas as pd
import numpy as np
import sys, os
from icecream import ic
from pyunfold import iterative_unfold
from pyunfold.callbacks import Logger

import numpy as np
np.random.seed(2)
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import filestruct, const, make_histos



"""
With PyUnfold, any response matrix should be 2-dimensional where the first dimension (rows) are effect bins and 
the second dimension (columns) are the causes bins.

In the example outlined above, each sample has a true x and y and an observed x and y. 
Since there are 19 x bins and 9 y bins, altogether there are 171 effects bins and 171 cause bins. 
One possible way to construct the effects bins is to have the first effect bin contain samples 
that are in the first x and first y bin, the second effect bin contains samples that are in the first x bin and second y bin, 
the third effect bin contains samples that are in the first x bin and third y bin, etc. 
A similar scheme could then be used to define the cause bins.
from https://github.com/jrbourbeau/pyunfold/issues/89
'unrolling'
"""
def plot_distributions(bins, truth_data_patch, observed_data_patch, response_hist_patch, response_normalized_patch, unfolded_results):
        fig, ax = plt.subplots()
        ax.step(bins, truth_data_patch, where='mid', lw=3,
                alpha=0.7, label='True distribution')
        ax.step(bins, observed_data_patch, where='mid', lw=3,
                alpha=0.7, label='Observed distribution')
        ax.set(xlabel='X bins', ylabel='Counts')
        ax.legend()
        plt.show()


        fig, ax = plt.subplots()
        im = ax.imshow(response_hist_patch, origin='lower',norm=mpl.colors.LogNorm())
        cbar = plt.colorbar(im, label='Counts')
        ax.set(xlabel='Cause bins', ylabel='Effect bins',title='Response matrix')
        plt.show()

        fig, ax = plt.subplots()
        im = ax.imshow(response_normalized_patch, origin='lower')
        cbar = plt.colorbar(im, label='$P(E_i|C_{\mu})$')
        ax.set(xlabel='Cause bins', ylabel='Effect bins',
        title='Normalized response matrix')
        plt.show()

        fig, ax = plt.subplots()
        ax.step(bins, truth_data_patch, where='mid', lw=3,
                alpha=0.7, label='True distribution')
        ax.step(bins, observed_data_patch, where='mid', lw=3,
                alpha=0.7, label='Observed distribution')
        ax.errorbar(bins, unfolded_results['unfolded'],
                yerr=unfolded_results['sys_err'],
                alpha=0.7,
                elinewidth=3,
                capsize=4,
                ls='None', marker='.', ms=10,
                label='Unfolded distribution')

        ax.set(xlabel='X bins', ylabel='Counts')
        plt.legend()
        plt.show()

        #create another plot for residuals
        fig, ax = plt.subplots()
        ax.errorbar(bins, np.abs(observed_data_patch-truth_data_patch),
                        alpha=0.7,
                        elinewidth=3,
                        capsize=4,
                        ls='None', marker='.', ms=10,
                        label='Observed distribution')
        ax.errorbar(bins, np.abs(unfolded_results['unfolded']-truth_data_patch),
                        alpha=0.7,
                        elinewidth=3,
                        capsize=4,
                        ls='None', marker='.', ms=10,
                        label='Unfolded distribution')
        ax.set(xlabel='X bins', ylabel='Counts')
        plt.legend()
        plt.show()

def combine_matrices(response_matrices,normalization=True):
        # Get the number of matrices (m), rows in each matrix (n), and the number of columns in each matrix (d)
        m, n, d = np.shape(response_matrices)

        # Initialize an empty list to store the averaged rows
        averaged_rows = []

        # Process each row from the response matrices
        for i in range(n):
                #i=1
                # Extract the ith row from each matrix
                rows = [matrix[i] for matrix in response_matrices]

                # Calculate the average values of the rows and add the result to the list
                #averaged_rows.append(calculate_ratios(rows))
                # Convert list of arrays into a 2D numpy array
                arrays_2d = np.array(rows)

                # Compute the mean of each column (i.e., each element across arrays), ignoring nan values
                averages = np.nanmean(arrays_2d, axis=0)

                if normalization:
                        # Normalize the averages so they sum to 1
                        averaged_normalized_row = averages / np.nansum(averages)
                        averaged_rows.append(averaged_normalized_row)
                else:
                        averaged_rows.append(averages)

        # Convert the list of averaged rows to a numpy array
        output_matrix = np.array(averaged_rows)
        #replace nan with zero
        output_matrix[np.isnan(output_matrix)] = 0
        return output_matrix

def unfold_function(truth_data=None,
                observed_data=None,
                response_hist=None,
                vector_ids=None,
                stop_lim = 0.00005):


        #Keep only the rows and columns that are in the vector_ids
        response_hist_patch = response_hist[vector_ids]
        response_hist_patch = response_hist_patch[:,vector_ids]
        observed_data_patch = observed_data[vector_ids]
        truth_data_patch = truth_data[vector_ids]

        ##ic(response_hist_patch)

        observed_data_err = np.sqrt(observed_data_patch)
        efficiencies = np.ones_like(observed_data_patch, dtype=float)
        efficiencies_err = np.full_like(efficiencies, 0.1, dtype=float)
        response_hist_err_patch = np.sqrt(response_hist_patch)

        column_sums = response_hist_patch.sum(axis=0)
        # Add a small constant to avoid division by zero
        column_sums = column_sums + 1e-10
        normalization_factor = efficiencies / column_sums

        response_normalized_patch = response_hist_patch * normalization_factor
        response_err_patch_normalized = response_hist_err_patch * normalization_factor

        uniform_prior = priors.uniform_prior(num_causes=len(truth_data_patch))
        causes = np.arange(len(truth_data_patch))
        num_causes = len(truth_data_patch)
        cause_lim = np.logspace(0, 3, num_causes)
        jeffreys_prior = priors.jeffreys_prior(cause_lim)
        #print(jeffreys_prior)
        ic(response_hist_patch)
        

        unfolded_results = iterative_unfold(data=observed_data_patch,
                                    data_err=observed_data_err,
                                    response=response_normalized_patch,
                                    response_err=response_err_patch_normalized,
                                    efficiencies=efficiencies,
                                    efficiencies_err=efficiencies_err,
                                    prior=jeffreys_prior,
                                    callbacks=[],
                                    ts_stopping=stop_lim,)
        
        return unfolded_results,truth_data_patch,observed_data_patch,response_hist_patch,response_normalized_patch

def unroll(var_1_bin, var_2_bin, var_1_bins):
    unrolled_bin = var_2_bin * len(var_1_bins) + var_1_bin
    return unrolled_bin

def get_data():
        print("Getting data")

        PhysicsConstants = const.PhysicsConstants()


        fs = filestruct.fs()

        test_file = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/inb/rec/dvpip_events_norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon.pkl"

        #df = pd.read_pickle(test_file)

        data_dir = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/inb/rec/"
        read_in = True
        if read_in:
                # for each pickle file in datadir, read it in, and then combine into one dataframe
                df = pd.DataFrame()
                for file in os.listdir(data_dir):
                        if file.endswith(".pkl"):
                                print(file)
                                df = df.append(pd.read_pickle(data_dir+file), ignore_index=True)
        else:
                df = pd.read_pickle(test_file)#.head(300_00)

        bins_Q2,bins_xB, bins_t1, bins_phi1 = fs.Q2bins, fs.xBbins, fs.tbins, fs.phibins


        # Define a dictionary to map column names to their bins
        columns_and_bins = {
                'xB': bins_xB,
                'Q2': bins_Q2,
                't2': bins_t1,
                'phi1': bins_phi1,
                'GenxB': bins_xB,
                'GenQ2': bins_Q2,
                'Gent2': bins_t1,
                'Genphi1': bins_phi1
        }

        # Perform the operation for all specified columns
        for column, bins in columns_and_bins.items():
                # creating the binning
                df[f'{column}_bin_category'] = pd.cut(df[column], bins=bins, right=False, include_lowest=True)

                # getting the minimum edge of the bin
                df[f'{column}_bin'] = df[f'{column}_bin_category'].apply(lambda x: x.left)

                # getting the bin number
                df[f'{column}_bin_number'] = df[f'{column}_bin_category'].cat.codes

                # dropping the temporary bin_category column
                df = df.drop(columns=[f'{column}_bin_category'])

        # Now, filter out rows where any bin_number column contains '-1'
        for column in columns_and_bins.keys():
                df = df[df[f'{column}_bin_number'] != -1]


        q_bins = np.arange(0,len(bins_Q2)-1,1)

        var_2_bins = np.arange(0,len(bins_xB)-1,1) #need to subtract 1 because bins_xB includes edges, and x_bins is just the bin labels number
        df['observed_var_2'] = df['xB_bin_number']
        df['truth_var_2'] = df['GenxB_bin_number']

        var_1_bins = np.arange(0,len(bins_t1)-1,1)
        df['observed_var_1'] = df['t2_bin_number']
        df['truth_var_1'] = df['Gent2_bin_number']

        total_unrolled_number_of_bins = len(var_1_bins)*len(var_2_bins)
        #Unroll the 2D data into 1D columns of observation and truth
        df['unrolled_truth_bins'] = df['truth_var_2']*len(var_1_bins)+df['truth_var_1']
        df['unrolled_observed_bins'] = df['observed_var_2']*len(var_1_bins)+df['observed_var_1']

        return df, total_unrolled_number_of_bins, var_1_bins, var_2_bins


def unstich_unfold_restich(x_width=3,q_width=3,x_stride=1,q_stride=1,stop_lim=0.00005
                           ):

        # if stride = 0, change it to 1. This is a simple fix to avoid a divide by zero error
        if x_stride == 0:
                x_stride = 1
        if q_stride == 0:
                q_stride = 1
        print("Unfolding with x-q kernel of size {}x{} with strides of {}x{}".format(x_width,q_width,x_stride,q_stride))
        unfolding_matrices  = []
        stat_errors  = []
        sys_errors  = []

        v_ids = []
        for x in range(0,len(x_bins)-(x_width-1),x_stride):
                for t in range(0,len(t_bins)-(q_width-1),q_stride):
                        x_range = x_bins[x:x+x_width]
                        t_range = t_bins[t:t+q_width]
                        
                        print("Calculating unfolding matrix for x bins {} and q bins {}".format(x_range,t_range))
                        bin_ids = []
                        for x_bin in x_range:
                                for t_bin in t_range:
                                        unrolled_bin = unroll(x_bin, t_bin, x_bins)
                                        bin_ids.append(unrolled_bin)
                        bin_ids = np.sort(bin_ids)

                        v_ids.append(bin_ids)
                        #sort the bin_ids
                        unfolded_results,truth_data_patch,observed_data_patch,response_hist_patch,response_normalized_patch = unfold_function(truth_data=truth_data,
                                                                                        observed_data=observed_data,
                                                                                        response_hist=response_hist,
                                                                                        vector_ids=bin_ids,
                                                                                        stop_lim=stop_lim)                
                        unfolding_matrices.append(unfolded_results['unfolding_matrix'])

                        ic(unfolding_matrices)
                        # Need to investigate priors, efficencies, and one other thing
                        # Can see everything avaliable with:
                        ###ic(unfolded_results.keys())
                        
                        stat_errors.append(np.diag(unfolded_results['stat_err']))
                        sys_errors.append(np.diag(unfolded_results['sys_err']))

                        ##ic(stat_errors)

                        plotting = True
                        if plotting:
                                plot_distributions(np.arange(0, len(bin_ids), 1), truth_data_patch, 
                                                observed_data_patch, response_hist_patch, 
                                                response_normalized_patch, unfolded_results)
                                
        enlarged_unfolding_matrices = []
        enlarged_sys_errors_matrices = []
        enlarged_stat_errors_matrices = []

        # Iterate over unfolding_matrices
        for count, (element_id, unfolding_matrix,sys_error_mat,stat_error_mat) in enumerate(zip(v_ids,unfolding_matrices,sys_errors,stat_errors)):
                unfolded_response = np.zeros((total_unrolled_number_of_bins,total_unrolled_number_of_bins))
                sys_errors_response = np.zeros((total_unrolled_number_of_bins,total_unrolled_number_of_bins))
                stat_errors_response = np.zeros((total_unrolled_number_of_bins,total_unrolled_number_of_bins))

                ## replace all zeros with nan
                unfolded_response[unfolded_response == 0] = np.nan
                sys_errors_response[sys_errors_response == 0] = np.nan
                stat_errors_response[stat_errors_response == 0] = np.nan

                ##ic(stat_error_mat)
                ##ic(sys_error_mat)
                ##ic(unfolding_matrix)
                for i in range(unfolding_matrix.shape[0]):  # iterating over rows
                        for j in range(unfolding_matrix.shape[1]):  # iterating over columns
                                ###ic(i,j)
                                unfolded_response[element_id[i],element_id[j]] = unfolding_matrix[i][j]
                                sys_errors_response[element_id[i],element_id[j]] = sys_error_mat[i][j]
                                stat_errors_response[element_id[i],element_id[j]] = stat_error_mat[i][j]

                enlarged_unfolding_matrices.append(unfolded_response)
                enlarged_sys_errors_matrices.append(sys_errors_response)
                enlarged_stat_errors_matrices.append(stat_errors_response)



        ###ic(enlarged_unfolding_matrices)
        ###ic(enlarged_sys_errors_matrices)
        output_matrix = combine_matrices(enlarged_unfolding_matrices,normalization=True)
        output_sys_err_matrix = combine_matrices(enlarged_sys_errors_matrices,normalization=False)
        output_stat_err_matrix = combine_matrices(enlarged_stat_errors_matrices,normalization=False)

        
        
        #replace nan values with zero
        output_matrix[np.isnan(output_matrix)] = 0
        output_sys_err_matrix[np.isnan(output_sys_err_matrix)] = 0
        output_stat_err_matrix[np.isnan(output_stat_err_matrix)] = 0

        #take transpose of output matrix
        output_matrix = output_matrix.T
        output_sys_err_matrix = output_sys_err_matrix.T
        output_stat_err_matrix = output_stat_err_matrix.T

        ###ic(output_matrix)
        ###ic(output_sys_err_matrix)
        ##ic(output_stat_err_matrix)
        
        return output_matrix, output_sys_err_matrix, output_stat_err_matrix
        # Old code for manually comparing matrices
        # np.savetxt("full_response_{}.csv".format("reconstructed"), output_matrix, delimiter=",")# header="Column1,Column2,Column3")

        # #read "full_response_{}.csv".format("reconstructed") into a numpy array
        # output_matrix_full = np.genfromtxt("full_response_{}.csv".format("reconstructed"), delimiter=',')

        # diffs = np.abs(output_matrix/output_matrix_full-1)*100
        # #replace nan with zero
        # diffs[np.isnan(diffs)] = 0
        # # round each value to nearest int
        # diffs = np.round(diffs).astype(int)
        # ##ic(diffs)
        # #print the first 6x6 elements of the array
        # for end in range(0,len(diffs),1):
        #         ##ic(diffs[0:end,0:end])
        # #check if any elements are larger than 0.05
        # if np.any(diffs > 2):
        #         #print which elements are
        #         print("The following elements are larger than 5% different from the original matrix:")
        #         print(np.where(np.abs(output_matrix/output_matrix_full-1)*100 > 5))




def calc_resp_matrix(df, total_unrolled_number_of_bins):
        
        binned_truth="unrolled_truth_bins"
        binned_data="unrolled_observed_bins"

        bins = np.arange(0, total_unrolled_number_of_bins, 1)

        value_counts_truth = df[binned_truth].value_counts()
        counts_series_truth = pd.Series(0, index=bins)
        counts_truth = counts_series_truth.add(value_counts_truth, fill_value=0)
        truth_data = counts_truth.values

        value_counts_observed= df[binned_data].value_counts()
        counts_series_observed = pd.Series(0, index=bins)
        counts_observed = counts_series_observed.add(value_counts_observed, fill_value=0)
        observed_data = counts_observed.values
        # Creating a zero-filled DataFrame with desired index and columns
        hist_2d = pd.DataFrame(0, index=bins, columns=bins)

        # Calculating the counts using crosstab
        counts = pd.crosstab(df[binned_data], df[binned_truth])

        # Filling the actual counts into our zero-filled DataFrame
        hist_2d.update(counts)

        # Converting the DataFrame to numpy array
        response_hist = hist_2d.values
        #print the size of the array
        #round the array to the nearest integer
        response_hist = np.round(response_hist).astype(int)

        return truth_data, observed_data, response_hist, bins

######################
#####################

# Get initial data
df, total_unrolled_number_of_bins, x_bins, t_bins = get_data()
# Calculate globablly useful things
truth_data, observed_data, response_hist, bins = calc_resp_matrix(df, total_unrolled_number_of_bins)

####################
####################

print(x_bins)
print(t_bins)
#since using square kernels, need the minimum bin span
min_bin_span = min(len(x_bins),len(t_bins))
print("Minimum bin span is {}".format(min_bin_span))

import time

# Initialize comp_time and average_err arrays
comp_time = np.zeros((min_bin_span, min_bin_span))
average_err = np.zeros((min_bin_span, min_bin_span))
true_output_matrix = None  # Variable to hold the "true" output matrix


for kernel_size in range(min_bin_span,2,-1): #For example, if image is 4x4x4x4, then kernel size can be 4,3,2
                        stride_max = min(kernel_size, min_bin_span - kernel_size +1)
                        stride_range = [0] if kernel_size == min_bin_span else range(1, stride_max + 1)  # add stride 0 if kernel size equals image size
        #for stride in stride_range:
                #if stride == 0 or (min_bin_span - kernel_size) % stride == 0:  # also handle stride=0 case
                        #kernel_size = 2
                        stride = 1
                        print("Unfolding with x-q kernel of size {}x{} with strides of {}x{}".format(kernel_size, kernel_size, stride, stride))

                        start_time = time.time()

                        output_matrix, output_sys_err_matrix, output_stat_err_matrix = unstich_unfold_restich(x_width=kernel_size,
                                                                                                              q_width=kernel_size,x_stride=stride,q_stride=stride,
                                                                                                              stop_lim=0.000005)

                        # Compute elapsed time and store it in comp_time array
                        elapsed_time = time.time() - start_time

                        # If this is the "true" output matrix (i.e., kernel size equals min_bin_span), store it
                        if kernel_size == min_bin_span and stride == 1:
                                true_output_matrix = output_matrix
                                basis_elapse_time = elapsed_time
                                # #calcuate inverse of response matrix
                                # response_matrix_inv = np.linalg.inv(response_hist)
                                # #compare true_output_matrix to response matrix inverse
                                # ##ic(true_output_matrix)
                                # ##ic(response_matrix_inv)
                                # diffs = np.abs(true_output_matrix/response_matrix_inv-1)*100
                                # #replace nan with zero
                                # diffs[np.isnan(diffs)] = 0
                                # # round each value to nearest int
                                # diffs = np.round(diffs).astype(int)
                                # ##ic(diffs)
                                # sys.exit()

                        error_matrix = np.where(np.abs(true_output_matrix) > 1e-7,(output_matrix - true_output_matrix) / true_output_matrix, 0)

                        # Compute average absolute difference of all non-zero and non-nan elements in error_matrix
                        # and store it in average_err array as a percent
                        print("Average error: ", np.nanmean(np.abs(error_matrix))*100)
                        print("Computation time: ", elapsed_time/basis_elapse_time)
                        average_err[kernel_size-1][stride] = np.nanmean(np.abs(error_matrix))*100
                        comp_time[kernel_size-1][stride] = elapsed_time/basis_elapse_time#
                        
                        unfolded_data = np.dot(output_matrix, observed_data)


                        #make a plot showing the unfolded data, the observed data, and the truth data
                        fig, ax = plt.subplots()
                        ax.step(bins, truth_data, where='mid', lw=3,
                                alpha=0.7, label='True distribution')
                        ax.step(bins, observed_data, where='mid', lw=3,
                                alpha=0.7, label='Observed distribution')
                        ax.errorbar(bins, unfolded_data,
                                yerr=np.sqrt(np.diag(output_sys_err_matrix)**2+np.diag(output_stat_err_matrix)**2),
                                alpha=0.7,
                                elinewidth=3,
                                capsize=4,
                                ls='None', marker='.', ms=10,
                                label='Unfolded distribution')
                        plt.show()

# print("Average error matrix:")
# print(average_err)
# print("Computation time matrix:")
# print(comp_time)
# import numpy.ma as ma
# import matplotlib.pyplot as plt

# # Define your colormap
# cmap = plt.cm.viridis
# cmap.set_bad(color='white')

# # Create the masked arrays
# comp_time_masked = ma.masked_where(np.isnan(comp_time) | (comp_time == 0), comp_time)
# average_err_masked = ma.masked_where(np.isnan(average_err) | (average_err == 0), average_err)

# fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# x_ticks = np.arange(min_bin_span+1)  # consider additional column for stride=0
# y_ticks = np.arange(min_bin_span) + 1

# # Modify imshow calls to use masked array and the colormap with white for bad values
# c1 = axs[0].imshow(comp_time_masked.T, cmap=cmap, interpolation='nearest', origin='lower')
# fig.colorbar(c1, ax=axs[0])
# axs[0].set_title('Computation Time')
# axs[0].set_xlabel('Kernel Size')
# axs[0].set_ylabel('Stride')
# axs[0].set_xticks(np.arange(min_bin_span))
# axs[0].set_yticks(np.arange(min_bin_span+1))
# axs[0].set_xticklabels(y_ticks)
# axs[0].set_yticklabels(x_ticks)

# c2 = axs[1].imshow(average_err_masked.T, cmap=cmap, interpolation='nearest', origin='lower')
# fig.colorbar(c2, ax=axs[1])
# axs[1].set_title('Average Error')
# axs[1].set_xlabel('Kernel Size')
# axs[1].set_ylabel('Stride')
# axs[1].set_xticks(np.arange(min_bin_span))
# axs[1].set_yticks(np.arange(min_bin_span+1))
# axs[1].set_xticklabels(y_ticks)
# axs[1].set_yticklabels(x_ticks)

# plt.show()
