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


ic.disable()
def unfold_function(truth_data=None,
                observed_data=None,
                response_hist=None,
                bins=None,vector_ids=None):


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

#Unroll the 2D data into 1D columns of observation and truth
df['unrolled_truth_bins'] = df['truth_q']*len(x_bins)+df['truth_x']
df['unrolled_observed_bins'] = df['observed_q']*len(x_bins)+df['observed_x']

binned_truth="unrolled_truth_bins"
binned_data="unrolled_observed_bins"

bins = np.arange(0, 8, 1)

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

                ic(bin_ids)
                v_ids.append(bin_ids)
                #sort the bin_ids
                unfolded_results,truth_data_patch,observed_data_patch,response_hist_patch,response_normalized_patch = unfold_function(truth_data=truth_data,
                                                                                observed_data=observed_data,
                                                                                response_hist=response_hist,
                                                                                bins=bins,vector_ids=bin_ids)                



                bins = np.arange(0, len(bin_ids), 1)


                # x_bins = [0,1]
                # q_bins = [0,1,2]

                # #create an unroll function, which takes in 


                ic(unfolded_results.keys())

                ic(unfolded_results['unfolded'])

                ic(unfolded_results['sys_err'])

                ic("Unfolding matrix is: ")
                ic(unfolded_results['unfolding_matrix'])
                unfolding_matrices.append(unfolded_results['unfolding_matrix'])
                


                ic(truth_data)
                ic(observed_data)

                plotting = False
                if plotting:
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

ic(unfolding_matrices )
ic(v_ids)

my_array = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.3465664, 0.0, 0.0, 0.32757861, 0.325855],
    [0.0, 0.23126918, 0.33383455, 0.0, 0.0, 0.43489627]
])

print("Unfolding matrices:")
ic(unfolding_matrices)
ic(v_ids)
enlarged_matrices = []
# Iterate over unfolding_matrices
for count, (element_id, unfolding_matrix) in enumerate(zip(v_ids,unfolding_matrices)):
        unfolded_response = np.zeros((8,8))
        ## replace all zeros with nan
        unfolded_response[unfolded_response == 0] = np.nan

#     # Pad unfolding_matrix with zeros to make it 6x6
#     if i == 0:  # for the first matrix, pad the bottom and right
#         padded_matrix = np.pad(unfolding_matrix, ((0,2),(0,2)))
#     elif i == 1:  # for the second matrix, pad the top and left
#         padded_matrix = np.pad(unfolding_matrix, ((2,0),(2,0)))
        for i in range(unfolding_matrix.shape[0]):  # iterating over rows
                for j in range(unfolding_matrix.shape[1]):  # iterating over columns
                        ic(i)
                        ic(j)
                        ic(element_id[i])
                        ic(element_id[j])
                        ic(unfolding_matrix[i][j])
                        unfolded_response[element_id[i],element_id[j]] = unfolding_matrix[i][j]
        enlarged_matrices.append(unfolded_response)
                                


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
                print("the list array is:")
                print(list_arr)
                print(d)
                #i = 2
                # get smallest non-zero value in the list
        
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
        
        print("arrays are:")
        print(ratioed_arrays)

        ratioed_arrays_np = np.array(ratioed_arrays)
        
        
        output_arr = np.nanmean(ratioed_arrays_np, axis=0)
        print("output array is:")
        #output_arr = np.array([ 1 ,2, np.nan, np.nan, np.nan, np.nan])

        print(output_arr)
        #normalize the array
        
        not_nan = ~np.isnan(output_arr)
    
        # Calculate the sum of the non-nan values
        sum_not_nan = np.sum(output_arr[not_nan])
        
        # Divide the non-nan values by the sum to normalize them
        output_arr[not_nan] /= sum_not_nan
        #output_arr /= np.sum(output_arr)
        print("output array is:")

        print(output_arr)
        return output_arr

def combine_matrices(response_matrices):
        # Get the number of matrices (m), rows in each matrix (n), and the number of columns in each matrix (d)
        m, n, d = np.shape(response_matrices)

        print(np.shape(response_matrices))
        # Initialize an empty list to store the averaged rows
        averaged_rows = []

        # Process each row from the response matrices
        for i in range(n):
                #i=1
                # Extract the ith row from each matrix
                rows = [matrix[i] for matrix in response_matrices]

                print(rows)

                # Calculate the average values of the rows and add the result to the list
                averaged_rows.append(calculate_ratios(rows))
                print(averaged_rows)
        # Convert the list of averaged rows to a numpy array
        output_matrix = np.array(averaged_rows)

        return output_matrix


print("Enlarged matrices:")
print(enlarged_matrices[0])
print(enlarged_matrices[1])
# Test the function
response_matrices = [np.array([[0.5, 0.5, np.nan], [0.5, 0.5, np.nan]]), np.array([[np.nan, 0.5, 0.5], [np.nan, 0.25, 0.75]])]


#output_matrix = combine_matrices(response_matrices)
output_matrix = combine_matrices(enlarged_matrices)

print(output_matrix)
print(my_array)

sys.exit()

def combine_response_matrices(response_matrices):
        # Initialize a zero matrix with the same shape as the response matrices
        total_matrix = np.zeros_like(response_matrices[0])
        
        # Initialize a counter matrix to keep track of how many times each bin is used
        counter_matrix = np.zeros_like(response_matrices[0])

        # Iterate over response matrices and add them to the total matrix
        for matrix in response_matrices:
                # Wherever matrix has a value, increase the counter
                counter_matrix += (matrix != 0)
                #total_matrix += matrix

        # Replace zeros with ones in counter_matrix to avoid division by zero
        counter_matrix[counter_matrix == 0] = 1

        print(counter_matrix)
        # Divide the total matrix by counter_matrix to get the average
        avg_matrix = total_matrix / counter_matrix

        return avg_matrix
a = combine_response_matrices(enlarged_matrices)
print(a)


# Test the function
#output_arr = calculate_ratios(list_arr)
#print(unfolding_matrix)


    # Add padded_matrix to unfolded_response
    #unfolded_response = np.add(unfolded_response, padded_matrix)

# Print the unfolded response matrix
ic(unfolded_response)
ic(my_array)
sys.exit()

# With PyUnfold, any response matrix should be 2-dimensional where the first dimension (rows) are effect bins and 
# the second dimension (columns) are the causes bins.

"""
In the example outlined above, each sample has a true x and y and an observed x and y. 
Since there are 19 x bins and 9 y bins, altogether there are 171 effects bins and 171 cause bins. 
One possible way to construct the effects bins is to have the first effect bin contain samples 
that are in the first x and first y bin, the second effect bin contains samples that are in the first x bin and second y bin, 
the third effect bin contains samples that are in the first x bin and third y bin, etc. 
A similar scheme could then be used to define the cause bins.
from https://github.com/jrbourbeau/pyunfold/issues/89
'unrolling'
"""

test_file = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/inb/rec/dvpip_events_norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon.pkl"
binned_test_file = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/rec/singles_t2/binned_dvpip_events_norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon.pkl"
df = pd.read_pickle(test_file)#.head(1000)
ic(df.columns.values)


# for 'xB' and 'GenxB'
# for 'xB' and 'GenxB'
bins_xB_extended = np.concatenate(([-np.inf], bins_xB, [np.inf]))

#ic(bins_xB_extended)
#ic(bins_xB_extended[:-1])
df['bin_xB'] = pd.cut(df['xB'], bins=bins_xB_extended, right=False, labels=bins_xB_extended[:-1])
df['bin_GenxB'] = pd.cut(df['GenxB'], bins=bins_xB_extended, right=False, labels=bins_xB_extended[:-1])

# for 'Q2' and 'GenQ2'
bins_Q2_extended = np.concatenate(([-np.inf], bins_Q2, [np.inf]))
df['bin_Q2'] = pd.cut(df['Q2'], bins=bins_Q2_extended, right=False, labels=bins_Q2_extended[:-1])
df['bin_GenQ2'] = pd.cut(df['GenQ2'], bins=bins_Q2_extended, right=False, labels=bins_Q2_extended[:-1])

# for 't1' and 'Gent1'
bins_t1_extended = np.concatenate(([-np.inf], bins_t1, [np.inf]))
df['bin_t1'] = pd.cut(df['t1'], bins=bins_t1_extended, right=False, labels=bins_t1_extended[:-1])
df['bin_Gent1'] = pd.cut(df['Gent1'], bins=bins_t1_extended, right=False, labels=bins_t1_extended[:-1])

# for 'phi1' and 'Genphi1'
bins_phi1_extended = np.concatenate(([-np.inf], bins_phi1, [np.inf]))
df['bin_phi1'] = pd.cut(df['phi1'], bins=bins_phi1_extended, right=False, labels=bins_phi1_extended[:-1])
df['bin_Genphi1'] = pd.cut(df['Genphi1'], bins=bins_phi1_extended, right=False, labels=bins_phi1_extended[:-1])


#print xB, bin_xB, GenxB, bin_GenxB
ic(df[['xB', 'bin_xB', 'GenxB', 'bin_GenxB']].head(10))
#print Q2, bin_Q2, GenQ2, bin_GenQ2
ic(df[['Q2', 'bin_Q2', 'GenQ2', 'bin_GenQ2']].head(10))
#print t1, bin_t1, Gent1, bin_Gent1
ic(df[['t1', 'bin_t1', 'Gent1', 'bin_Gent1']].head(10))
#print phi1, bin_phi1, Genphi1, bin_Genphi1
ic(df[['phi1', 'bin_phi1', 'Genphi1', 'bin_Genphi1']].head(10))

# create a column with unique bin combination for 'xB', 'Q2', 't1', 'phi1'
df['bin_comb'] = df['bin_xB'].astype(str) + '_' + df['bin_Q2'].astype(str) + '_' + df['bin_t1'].astype(str) + '_' + df['bin_phi1'].astype(str)

# create a column with unique bin combination for 'GenxB', 'GenQ2', 'Gent1', 'Genphi1'
df['bin_comb_Gen'] = df['bin_GenxB'].astype(str) + '_' + df['bin_GenQ2'].astype(str) + '_' + df['bin_Gent1'].astype(str) + '_' + df['bin_Genphi1'].astype(str)

# create a column for unique bin numbers, using pandas' factorize method
df['bin_number'] = pd.factorize(df['bin_comb'])[0]

# similarly for the 'Gen' bin combinations
df['bin_number_Gen'] = pd.factorize(df['bin_comb_Gen'])[0]

ic(df['bin_number_Gen'])


# Define all bins arrays
bins_arrays = {'xB': bins_xB_extended, 'Q2': bins_Q2_extended, 't1': bins_t1_extended, 'phi1': bins_phi1_extended}

# Create the bin_number columns for all variables
plot_individual_bins = False
if plot_individual_bins:
    for var in bins_arrays.keys():
        # create a dictionary to map bin edges to bin indices
        bin_dict = {edge: i for i, edge in enumerate(bins_arrays[var][:-1])}
        bin_Gen_dict = {edge: i for i, edge in enumerate(bins_arrays[var][:-1])}

        # create the bin_number columns
        df[f'bin_{var}_number'] = df[f'bin_{var}'].map(bin_dict)
        df[f'bin_Gen{var}_number'] = df[f'bin_Gen{var}'].map(bin_Gen_dict)
        plt.figure(figsize=(10, 8))

        plt.hist2d(df[f'bin_{var}_number'], df[f'bin_Gen{var}_number'], bins=[df[f'bin_{var}_number'].nunique(), df[f'bin_Gen{var}_number'].nunique()])#, norm=mpl.colors.LogNorm())
        plt.colorbar(label='Counts')
        plt.xlabel(f'bin_{var}_number')
        plt.ylabel(f'bin_Gen{var}_number')
        plt.title(f'bin_{var}_number')
        plt.show()
    


# Print the number of unique bin numbers
ic("Unique 'bin_number' values:", df['bin_number'].nunique())

# Print the number of unique 'Gen' bin numbers
ic("Unique 'bin_number_Gen' values:", df['bin_number_Gen'].nunique())


#create a plot of bin_number vs bin_number_Gen using matplotlib

#plt.scatter(df['bin_number'], df['bin_number_Gen'])
#plt.show()

# Create a 2D histogram of bin_number vs bin_number_Gen
# plot with lognorm virdis
ic(df['bin_t1'].unique())
ic(df['bin_Gent1'].unique())

plt.figure(figsize=(10, 8))

#print the rows where bin_Gent1 is NaN
ic(df[df['bin_Gent1'].isnull()])
#save this row as a csv
df[df['bin_Gent1'].isnull()].to_csv('bin_Gent1_isnull.csv')
#plt.hist2d(df['bin_number'], df['bin_number_Gen'], bins=[df['bin_number'].nunique(), df['bin_number_Gen'].nunique()], norm=mpl.colors.LogNorm())
plt.hist2d(df['bin_number'], df['bin_number_Gen'], bins=[df['bin_number'].nunique(), df['bin_number_Gen'].nunique()], norm=mpl.colors.LogNorm())
plt.colorbar(label='Counts')
plt.xlabel('bin_number')
plt.ylabel('bin_number_Gen')
plt.title('2D Histogram of bin_number vs bin_number_Gen')
plt.show()


