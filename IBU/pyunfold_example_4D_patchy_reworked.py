import numpy as np
from pyunfold import iterative_unfold
from scipy.stats import norm
from pyunfold.priors import uniform_prior
from pyunfold.priors import jeffreys_prior
import pandas as pd
from utils import filestruct, const, make_histos
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
PhysicsConstants = const.PhysicsConstants()

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
        im = ax.imshow(response_hist_patch, origin='lower')
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

def calculate_ratios(list_arr):
        # Convert input to numpy array for easier processing
        list_arr = np.array(list_arr)

        # Get the number of lists (d) and the size of each list (n)
        d, n = list_arr.shape

        # Initialize an output array with zeros
        output_arr = np.zeros(n)

        # Process each list in the input array
        ratioed_arrays = []
        for i in range(d):
                # Ignore nan values and normalize remaining values to get ratios
                ratios = np.nan_to_num(list_arr[i])
                #if the array is all zeros, set the min val to zero
                if np.all(ratios == 0):
                        pass
                        # if all zero, we don't have to scale anything
                else:
                        min_val = np.min(ratios[ratios != 0])
                        # if min_val is less than 0.000001 print warning
                        if min_val < 0.000001:
                                print("Warning: min_val is less than 0.000001!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        ratios /= min_val
                        #set zeros in ratio to nan
                ratios[ratios == 0] = np.nan
                ratioed_arrays.append(ratios)
        
        ratioed_arrays_np = np.array(ratioed_arrays)
        output_arr = np.nanmean(ratioed_arrays_np, axis=0)
        not_nan = ~np.isnan(output_arr)
        # Calculate the sum of the non-nan values
        sum_not_nan = np.sum(output_arr[not_nan])
        # Divide the non-nan values by the sum to normalize them
        output_arr[not_nan] /= sum_not_nan
        return output_arr

def combine_matrices(response_matrices):
        # Get the number of matrices (m), rows in each matrix (n), and the number of columns in each matrix (d)
        m, n, d = np.shape(response_matrices)

        print(np.shape(response_matrices))
        # Initialize an empty list to store the averaged rows
        averaged_rows = []

        # Process each row from the response matrices
        for i in range(n):
                # Extract the ith row from each matrix
                rows = [matrix[i] for matrix in response_matrices]

                # Calculate the average values of the rows and add the result to the list
                averaged_rows.append(calculate_ratios(rows))
        # Convert the list of averaged rows to a numpy array
        output_matrix = np.array(averaged_rows)
        return output_matrix

def unfold_function(truth_data=None,
                observed_data=None,
                response_hist=None,
                vector_ids=None):


        #Keep only the rows and columns that are in the vector_ids
        response_hist_patch = response_hist[vector_ids]
        response_hist_patch = response_hist_patch[:,vector_ids]
        observed_data_patch = observed_data[vector_ids]
        truth_data_patch = truth_data[vector_ids]

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
       
        unfolded_results = iterative_unfold(data=observed_data_patch,
                                    data_err=observed_data_err,
                                    response=response_normalized_patch,
                                    response_err=response_err_patch_normalized,
                                    efficiencies=efficiencies,
                                    efficiencies_err=efficiencies_err,
                                    callbacks=[Logger()],
                                    ts_stopping=0.00005,)
        
        return unfolded_results,truth_data_patch,observed_data_patch,response_hist_patch,response_normalized_patch


data = {
        'truth_x':[0,0,1,1,1,2,2,1,2,2,3],
        'truth_q':[0,0,0,0,0,0,0,1,1,1,1],
        'observed_x':[1,1,2,1,1,2,2,1,2,1,2],
        'observed_q':[0,0,1,1,1,0,1,1,1,1,1],
}
df = pd.DataFrame(data)

def unroll(x_bin, q_bin, x_bins):
    unrolled_bin = q_bin * len(x_bins) + x_bin
    return unrolled_bin


x_bins = [0,1,2,3]
q_bins = [0,1]
total_unrolled_number_of_bins = len(x_bins)*len(q_bins)
#Unroll the 2D data into 1D columns of observation and truth
df['unrolled_truth_bins'] = df['truth_q']*len(x_bins)+df['truth_x']
df['unrolled_observed_bins'] = df['observed_q']*len(x_bins)+df['observed_x']

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


results = np.zeros((len(x_bins), len(q_bins)))
x_width = 2 # kernel x width
q_width = 2 # kernel q width
x_stride = 1 # kernel x stride
q_stride = 1 # kernel q stride
unfolding_matrices  = []
v_ids = []
for x in range(0,len(x_bins)-(x_width-1),x_stride):
        for q in range(0,len(q_bins)-(q_width-1),q_stride):
                x_range = x_bins[x:x+x_width]
                q_range = q_bins[q:q+q_width]
                
                bin_ids = []
                for x_bin in x_range:
                        for q_bin in q_range:
                                unrolled_bin = unroll(x_bin, q_bin, x_bins)
                                bin_ids.append(unrolled_bin)
                bin_ids = np.sort(bin_ids)

                v_ids.append(bin_ids)
                #sort the bin_ids
                unfolded_results,truth_data_patch,observed_data_patch,response_hist_patch,response_normalized_patch = unfold_function(truth_data=truth_data,
                                                                                observed_data=observed_data,
                                                                                response_hist=response_hist,
                                                                                vector_ids=bin_ids)                
                unfolding_matrices.append(unfolded_results['unfolding_matrix'])

                # Still need to do something with the err propagation
                # Need to investigate priors, efficencies, and one other thing
                # Can see everything avaliable with:
                # ic(unfolded_results.keys())
                # ic(unfolded_results['sys_err'])

                plotting = False
                if plotting:
                        plot_distributions(np.arange(0, len(bin_ids), 1), truth_data_patch, 
                                           observed_data_patch, response_hist_patch, 
                                           response_normalized_patch, unfolded_results)
                        



enlarged_matrices = []
# Iterate over unfolding_matrices
for count, (element_id, unfolding_matrix) in enumerate(zip(v_ids,unfolding_matrices)):
        unfolded_response = np.zeros((total_unrolled_number_of_bins,total_unrolled_number_of_bins))
        ## replace all zeros with nan
        unfolded_response[unfolded_response == 0] = np.nan

        for i in range(unfolding_matrix.shape[0]):  # iterating over rows
                for j in range(unfolding_matrix.shape[1]):  # iterating over columns
                        ic(i)
                        ic(j)
                        ic(element_id[i])
                        ic(element_id[j])
                        ic(unfolding_matrix[i][j])
                        unfolded_response[element_id[i],element_id[j]] = unfolding_matrix[i][j]
        enlarged_matrices.append(unfolded_response)
                                




output_matrix = combine_matrices(enlarged_matrices)
#replace nan values with zero
output_matrix[np.isnan(output_matrix)] = 0
unfolded_data = np.dot(output_matrix,observed_data)
print(output_matrix)

#make a plot showing the unfolded data, the observed data, and the truth data
fig, ax = plt.subplots()
ax.step(bins, truth_data, where='mid', lw=3,
        alpha=0.7, label='True distribution')
ax.step(bins, observed_data, where='mid', lw=3,
        alpha=0.7, label='Observed distribution')
ax.errorbar(bins, unfolded_data,
        alpha=0.7,
        elinewidth=3,
        capsize=4,
        ls='None', marker='.', ms=10,
        label='Unfolded distribution')
plt.show()






