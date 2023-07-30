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
        im = ax.imshow(response_normalized_patch, origin='lower',norm=mpl.colors.LogNorm())
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


def restitc_original(v_ids, unfolding_matrices, sys_errors, stat_errors, total_unrolled_number_of_bins):

        # WE NEED TO RESTITCH THE UNFOLDING MATRICES 
        enlarged_unfolding_matrices = []
        enlarged_sys_errors_matrices = []
        enlarged_stat_errors_matrices = []

        # Iterate over unfolding_matrices
        for count, (element_id, unfolding_matrix,sys_error_mat,stat_error_mat) in enumerate(zip(v_ids,unfolding_matrices,sys_errors,stat_errors)):
                unfolded_response = np.zeros((total_unrolled_number_of_bins,total_unrolled_number_of_bins))
                sys_errors_response = np.zeros((total_unrolled_number_of_bins,total_unrolled_number_of_bins))
                stat_errors_response = np.zeros((total_unrolled_number_of_bins,total_unrolled_number_of_bins))

                print("On restiching iteration {}".format(count))
                ## replace all zeros with nan
                unfolded_response[unfolded_response == 0] = np.nan
                sys_errors_response[sys_errors_response == 0] = np.nan
                stat_errors_response[stat_errors_response == 0] = np.nan

                for i in range(unfolding_matrix.shape[0]):  # iterating over rows
                        for j in range(unfolding_matrix.shape[1]):  # iterating over columns
                                unfolded_response[element_id[i],element_id[j]] = unfolding_matrix[i][j]
                                sys_errors_response[element_id[i],element_id[j]] = sys_error_mat[i][j]
                                stat_errors_response[element_id[i],element_id[j]] = stat_error_mat[i][j]

                enlarged_unfolding_matrices.append(unfolded_response)
                enlarged_sys_errors_matrices.append(sys_errors_response)
                enlarged_stat_errors_matrices.append(stat_errors_response)

        output_matrix = combine_matrices(enlarged_unfolding_matrices,normalization=True)
        output_sys_err_matrix = combine_matrices(enlarged_sys_errors_matrices,normalization=False)
        output_stat_err_matrix = combine_matrices(enlarged_stat_errors_matrices,normalization=False)

        #replace nan values with zero
        output_matrix[np.isnan(output_matrix)] = 0
        output_sys_err_matrix[np.isnan(output_sys_err_matrix)] = 0
        output_stat_err_matrix[np.isnan(output_stat_err_matrix)] = 0

        return output_matrix.T, output_sys_err_matrix.T, output_stat_err_matrix.T

def restitch_old(v_ids, unfolding_matrices,total_unrolled_number_of_bins):

        # WE NEED TO RESTITCH THE UNFOLDING MATRICES 
        enlarged_unfolding_matrices = []

        # Iterate over unfolding_matrices
        for count, (element_id, unfolding_matrix) in enumerate(zip(v_ids,unfolding_matrices)):
                unfolded_response = np.zeros((total_unrolled_number_of_bins,total_unrolled_number_of_bins))

                print("On restiching iteration {}".format(count))
                ## replace all zeros with nan
                unfolded_response[unfolded_response == 0] = np.nan

                for i in range(unfolding_matrix.shape[0]):  # iterating over rows
                        for j in range(unfolding_matrix.shape[1]):  # iterating over columns
                                unfolded_response[element_id[i],element_id[j]] = unfolding_matrix[i][j]

                enlarged_unfolding_matrices.append(unfolded_response)

        m, n, d = np.shape(enlarged_unfolding_matrices)

        # Initialize an empty list to store the averaged rows
        averaged_rows = []

        # Process each row from the response matrices
        for i in range(n):
                # Extract the ith row from each matrix
                rows = [matrix[i] for matrix in enlarged_unfolding_matrices]

                arrays_2d = np.array(rows)

                # Compute the mean of each column (i.e., each element across arrays), ignoring nan values
                averages = np.nanmean(arrays_2d, axis=0)

                # Normalize the averages so they sum to 1
                averaged_normalized_row = averages / np.nansum(averages)
                averaged_rows.append(averaged_normalized_row)

        # Convert the list of averaged rows to a numpy array
        output_matrix = np.array(averaged_rows)
        #replace nan with zero
        output_matrix[np.isnan(output_matrix)] = 0
        return output_matrix.T

def restitch(v_ids, unfolding_matrices, total_unrolled_number_of_bins,normalization=True):

    # Initialize sums and counts matrix to compute mean
    unfolding_sums = np.zeros((total_unrolled_number_of_bins, total_unrolled_number_of_bins))
    unfolding_counts = np.zeros((total_unrolled_number_of_bins, total_unrolled_number_of_bins))

    # Iterate over unfolding_matrices
    for count, (element_id, unfolding_matrix) in enumerate(zip(v_ids, unfolding_matrices)):
        print("On restitching iteration {}".format(count))

        # Add unfolding matrix to the corresponding position in the sums matrix
        unfolding_sums[np.ix_(element_id, element_id)] += unfolding_matrix

        # Increment counts where a number was added in sums matrix
        unfolding_counts[np.ix_(element_id, element_id)] += 1

    # Calculate means while handling divide by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        unfolded_means = np.true_divide(unfolding_sums, unfolding_counts)
        unfolded_means[unfolding_counts == 0] = np.nan

    # # Normalize the averages so they sum to 1 along each row
    # row_sums = np.nansum(unfolded_means, axis=1, keepdims=True)
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     output_matrix = np.true_divide(unfolded_means, row_sums)
    #     output_matrix[row_sums == 0] = 0

    # # replace nan with zero
    # output_matrix[np.isnan(output_matrix)] = 0

    row_sums = np.nansum(unfolded_means, axis=1, keepdims=True)

    if normalization:
        # Normalize the averages so they sum to 1 along each row
        with np.errstate(divide='ignore', invalid='ignore'):
            output_matrix = np.true_divide(unfolded_means, row_sums)
    else:
        output_matrix = unfolded_means

    # If a row in row_sums is 0, then set the corresponding row in output_matrix to 0
    output_matrix[(row_sums == 0).flatten(), :] = 0

    # replace nan with zero
    output_matrix[np.isnan(output_matrix)] = 0

    return output_matrix.T



def calc_resp_matrix(df, total_unrolled_number_of_bins):
        print("calculating response matrix")
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

        print("done calculating response matrix")
        return truth_data, observed_data, response_hist, bins

def get_data():
        ##ic("Getting data")

        PhysicsConstants = const.PhysicsConstants()


        fs = filestruct.fs()
        df_name = "df_quickproc.pkl"

        data_dir = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/inb/rec/"
        read_in = True
        validation = False
        if read_in:
                # for each pickle file in datadir, read it in, and then combine into one dataframe
                df = pd.DataFrame()
                pickle_filelist = []
                filelist = os.listdir(data_dir)
                for file in filelist:
                    if file.endswith(".pkl"):
                           pickle_filelist.append(file)
                print(pickle_filelist)
                files_training = pickle_filelist[:-1]
                files_verification = [pickle_filelist[-1]]
                if validation:
                    print(files_verification)
                    for file in files_verification:
                        ic(file)
                        df = df.append(pd.read_pickle(data_dir+file), ignore_index=True)
                else: 
                    print("training")
                    for file in files_training:
                        df = df.append(pd.read_pickle(data_dir+file), ignore_index=True)
        else:
                #df = pd.read_pickle(df_name)#.head(300_00)
                print("reading in")
                test_file = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/inb/rec/dvpip_events_norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon.pkl"

                df = pd.read_pickle(test_file)
        bins_Q2,bins_xB, bins_t1, bins_phi1 = fs.Q2bins, fs.xBbins, fs.tbins, fs.phibins


        # Define a dictionary to map column names to their bins
        columns_and_bins = {
                'xB': bins_xB,
                'Q2': bins_Q2,
                't1': bins_t1,
                'phi1': bins_phi1,
                'GenxB': bins_xB,
                'GenQ2': bins_Q2,
                'Gent1': bins_t1,
                'Genphi1': bins_phi1
        }

        # Perform the operation for all specified columns
        for column, bins in columns_and_bins.items():
                # creating the binning
                df[f'{column}_bin_category'] = pd.cut(df[column], bins=bins, right=False, include_lowest=True)

                # getting the minimum edge of the bin
                df[f'{column}_bin'] = df[f'{column}_bin_category'].apply(lambda x: x.left)

                # getting the bin number
                df[f'{column}_bin_number'] = df[f'{column}_bin_category'].cat.codes.astype(np.int64)

                # dropping the temporary bin_category column
                df = df.drop(columns=[f'{column}_bin_category'])

        # Now, filter out rows where any bin_number column contains '-1'
        for column in columns_and_bins.keys():
                df = df[df[f'{column}_bin_number'] != -1]

        x_bins = np.arange(0,len(bins_xB)-1,1) #need to subtract 1 because bins_xB includes edges, and x_bins is just the bin labels number
        q_bins = np.arange(0,len(bins_Q2)-1,1)
        t_bins = np.arange(0,len(bins_t1)-1,1)
        phi_bins = np.arange(0,len(bins_phi1)-1,1)


        df['observed_x'] = df['xB_bin_number']
        df['observed_q'] = df['Q2_bin_number']
        df['observed_t'] = df['t1_bin_number']
        df['observed_phi'] = df['phi1_bin_number']
        df['truth_x'] = df['GenxB_bin_number']
        df['truth_q'] = df['GenQ2_bin_number']
        df['truth_t'] = df['Gent1_bin_number']
        df['truth_phi'] = df['Genphi1_bin_number']

        #print unique values of all columns
        # total_unrolled_number_of_bins = len(x_bins)*len(q_bins)
        # df['unrolled_truth_bins'] = df['truth_q']*len(x_bins)+df['truth_x']
        # df['unrolled_observed_bins'] = df['observed_q']*len(x_bins)+df['observed_x']

        #Unroll the 4D data into 1D columns of observation and truth
        total_unrolled_number_of_bins = len(x_bins)*len(q_bins)*len(t_bins)*len(phi_bins)
        df['unrolled_truth_bins'] = df['truth_x']*len(q_bins)*len(t_bins)*len(phi_bins)+df['truth_q']*len(t_bins)*len(phi_bins)+df['truth_phi']*len(t_bins)+df['truth_t']
        df['unrolled_observed_bins'] = df['observed_x']*len(q_bins)*len(t_bins)*len(phi_bins)+df['observed_q']*len(t_bins)*len(phi_bins)+df['observed_phi']*len(t_bins)+df['observed_t']

        #get the rows wehre unrolled_truth_bins = -116
        if read_in:
                #save in local dir for quick rerunning
                df.to_pickle(df_name)
        print('done getting data')
        return df, total_unrolled_number_of_bins, x_bins, q_bins, t_bins, phi_bins

def load_data(save_dir='./saved_data'):
    iteration = 1
    bin_ids = []
    unfolding_matrices = []
    stat_errs = []
    sys_errs = []


    while True:
        try:
            # Load the data
            print("loading iteration {}".format(iteration))
            bin_ids.append(np.load(os.path.join(save_dir, f'bin_ids_{iteration}.npy')))
            unfolding_matrices.append(np.load(os.path.join(save_dir, f'unfolding_matrix_{iteration}.npy')))
            stat_errs.append(np.load(os.path.join(save_dir, f'stat_err_{iteration}.npy')))
            sys_errs.append(np.load(os.path.join(save_dir, f'sys_err_{iteration}.npy')))
            iteration += 1
        except FileNotFoundError:
            # If the file doesn't exist, we've reached the end of the files, so break out of the loop
            break
    
    return bin_ids, unfolding_matrices, stat_errs, sys_errs


data_dir = 'finalized_saved_data/'
#data_dir = 'saved_data/'



df, total_unrolled_number_of_bins, x_bins, q_bins, t_bins, phi_bins = get_data()
truth_data, observed_data, response_hist, bins = calc_resp_matrix(df, total_unrolled_number_of_bins)

np.save('final_truth_data.npy', truth_data)
np.save('final_observed_data.npy', observed_data)
np.save('bins.npy', bins)
np.save('total_unrolled_number_of_bins.npy', total_unrolled_number_of_bins)
sys.exit()

v_ids, unfolding_matrices_to_process, stat_errors, sys_errors = load_data(save_dir=data_dir)
print(len(v_ids))


#output_matrix, output_sys_err_matrix, output_stat_err_matrix = restitch(v_ids, unfolding_matrices, sys_errors, stat_errors, total_unrolled_number_of_bins)

output_matrix = restitch(v_ids, unfolding_matrices_to_process, total_unrolled_number_of_bins,normalization=True)
output_sys_err_matrix = restitch(v_ids, sys_errors, total_unrolled_number_of_bins,normalization=False)
output_stat_err_matrix = restitch(v_ids, stat_errors, total_unrolled_number_of_bins,normalization=False)

unfolded_data = np.dot(output_matrix, observed_data)


np.save('final_unfolded_data.npy', unfolded_data)
np.save('final_output_matrix.npy', output_matrix)
np.save('final_output_sys_err_matrix.npy', output_sys_err_matrix)
np.save('final_output_stat_err_matrix.npy', output_stat_err_matrix)


np.save('final_truth_data.npy', truth_data)
np.save('bins.npy', bins)
np.save('total_unrolled_number_of_bins.npy', total_unrolled_number_of_bins)

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
ax.set_xlim(0, total_unrolled_number_of_bins)
ax.set_ylim(0,8000)
plt.show()

sys.exit()

#bins for test_3_saved_data
"""
		self.xBbins = [0.1,0.15,0.20,0.25,0.30]#,0.38,0.48,0.58,0.80]
		self.Q2bins =  [1,1.5,2,2.5,3,3.5,4,4.5]#,5.5,7,11]
		self.tbins =  [0.09,0.15,0.2,0.3,0.4]#,0.6,1,1.5,2]
		#self.phibins =  [0,36,72,108,144,180,216,252,288,324,360]
		self.phibins =  [0,90,180,270,360]
"""

#bins for test_4_saved_data
"""
		self.xBbins = [0.1,0.15,0.20,0.25]#,0.30]#,0.38,0.48,0.58,0.80]
		self.Q2bins =  [1,1.5,2,2.5,3]#,3.5,4,4.5]#,5.5,7,11]
		self.tbins =  [0.09,0.15,0.2,0.3]#,0.4]#,0.6,1,1.5,2]
		#self.phibins =  [0,36,72,108,144,180,216,252,288,324,360]
		self.phibins =  [0,90,180,270,360]
"""

# need to get data too