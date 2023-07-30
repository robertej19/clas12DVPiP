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
def plot_distributions(bins, truth_data_patch, observed_data_patch, response_hist_patch, response_normalized_patch, unfolded_results,output_sys_err_matrix,output_stat_err_matrix):

        plt.rcParams.update({'font.size': 20})

        #fig, ax = plt.subplots()
        #make fig ax with size 14,10
        fig, ax = plt.subplots(figsize=(14,10))
        ax.step(bins, truth_data_patch, where='mid', lw=3,color='green',
                alpha=0.7, label='True distribution')
        ax.step(bins, observed_data_patch, where='mid', lw=3,color='red',
                alpha=0.7, label='Observed distribution')
        ax.set(xlabel='X bins', ylabel='Counts')
        #input title
        ax.set_title('4D 4-Bin Sample Observed and True Distributions')
        ax.legend()
        #plt.show()
        #save the figure
        plt.savefig('4D_4bin_sample_observed_and_true_distributions.png')
        plt.close()


        fig, ax = plt.subplots(figsize=(14,10))

        im = ax.imshow(response_hist_patch, origin='lower',norm=mpl.colors.LogNorm())
        #cbar = plt.colorbar(im, label='Counts')
        ax.set(xlabel='Bins, Truth', ylabel='Bins, Observed',title='$x_B$ Tiling on Response Matrix')

        # Add vertical bars
        alpha_value = 0

        # # Add horizontal bars
        ax.axhspan(-0.5, 7.5, alpha=alpha_value, color='red')
        ax.axhspan(len(response_hist_patch)-8-.5, len(response_hist_patch)-.5, alpha=alpha_value, color='green')

        ##Add vertical bars
        ax.axvspan(-0.5, 7.5, alpha=alpha_value, color='red')
        ax.axvspan(len(response_hist_patch[0])-8-0.5, len(response_hist_patch[0]-.5), alpha=alpha_value, color='green')


        #limit x range to -0.5, 15.5
        ax.set_xlim(-0.5, 15.5)
        #limit y range to -0.5, 15.5
        ax.set_ylim(-0.5, 15.5)
        #make layout tight
        plt.tight_layout()
        #save the figure
        plt.savefig('4D_4bin_sample_response_matrix_xb.png')
        #plt.show()
        plt.close()


        fig, ax = plt.subplots(figsize=(14,10))

        im = ax.imshow(response_hist_patch, origin='lower',norm=mpl.colors.LogNorm())
        ax.set(xlabel='Bins, Truth', ylabel='Bins, Observed',title='$x_B$-Q$^2$ Tiling on Response Matrix')

        # Set alpha value and color list
        alpha_value = 0.4

        # # Add horizontal bars
        ax.axhspan(-0.5, 7.5, alpha=alpha_value, color='red')
        ax.axhspan(len(response_hist_patch)-8-.5, len(response_hist_patch)-.5, alpha=alpha_value, color='green')

        ##Add vertical bars
        ax.axvspan(-0.5, 7.5, alpha=alpha_value, color='red')
        ax.axvspan(len(response_hist_patch[0])-8-0.5, len(response_hist_patch[0]-.5), alpha=alpha_value, color='green')

        colors = ['orange', 'blue']*2#yellow', 'green', 'blue']

        # Calculate the width of each band
        band_width = len(response_hist_patch) / 4
        band_width_x = len(response_hist_patch[0]) / 4

        # Add horizontal and vertical bars
        for i in range(4):
                ax.axhspan(i*band_width-0.5, (i+1)*band_width-0.5, alpha=alpha_value, color=colors[i])
                ax.axvspan(i*band_width_x-0.5, (i+1)*band_width_x-0.5, alpha=alpha_value, color=colors[i])

        # Set x and y range
        ax.set_xlim(-0.5, len(response_hist_patch[0])-0.5)
        ax.set_ylim(-0.5, len(response_hist_patch)-0.5)

        #save the figure
        #make layout tight
        plt.tight_layout()
        plt.savefig('4D_2bin_sample_response_matrix_xb_q2.png')
        plt.close()
        #plt.show()

        fig, ax = plt.subplots(figsize=(14,10))

        im = ax.imshow(response_hist_patch, origin='lower',norm=mpl.colors.LogNorm())
        ax.set(xlabel='Bins, Truth', ylabel='Bins, Observed',title='$x_B$-Q$^2$-$\phi$ Tiling on Response Matrix')

        alpha_value = 0.2

        # # Add horizontal bars
        ax.axhspan(-0.5, 7.5, alpha=alpha_value, color='red')
        ax.axhspan(len(response_hist_patch)-8-.5, len(response_hist_patch)-.5, alpha=alpha_value, color='green')

        ##Add vertical bars
        ax.axvspan(-0.5, 7.5, alpha=alpha_value, color='red')
        ax.axvspan(len(response_hist_patch[0])-8-0.5, len(response_hist_patch[0]-.5), alpha=alpha_value, color='green')

        #alpha_value = 0.4
        colors = ['orange', 'blue']*2#, 'red', 'blue']

        # Calculate the width of each band
        band_width = len(response_hist_patch) / 4
        band_width_x = len(response_hist_patch[0]) / 4

        # Add horizontal and vertical bars
        for i in range(4):
                ax.axhspan(i*band_width-0.5, (i+1)*band_width-0.5, alpha=alpha_value, color=colors[i])
                ax.axvspan(i*band_width_x-0.5, (i+1)*band_width_x-0.5, alpha=alpha_value, color=colors[i])


        # Set alpha value and color list
        colors = ['yellow', 'purple']*4#'orange', 'yellow', 'green', 'blue', 'indigo', 'violet','black']

        # Calculate the width of each band
        band_width = len(response_hist_patch) / 8
        band_width_x = len(response_hist_patch[0]) / 8

        # Add horizontal and vertical bars
        for i in range(8):
                ax.axhspan(i*band_width-0.5, (i+1)*band_width-0.5, alpha=alpha_value, color=colors[i])
                ax.axvspan(i*band_width_x-0.5, (i+1)*band_width_x-0.5, alpha=alpha_value, color=colors[i])

        # Set x and y range
        ax.set_xlim(-0.5, len(response_hist_patch[0])-0.5)
        ax.set_ylim(-0.5, len(response_hist_patch)-0.5)

        plt.tight_layout()
        plt.savefig('4D_2bin_sample_response_matrix_xb_q2_phi.png')
        plt.close()

        fig, ax = plt.subplots(figsize=(14,10))

        im = ax.imshow(response_hist_patch, origin='lower',norm=mpl.colors.LogNorm())
        ax.set(xlabel='Bins, Truth', ylabel='Bins, Observed',title='$x_B$-Q$^2$-$\phi$-t Tiling on Response Matrix')

        # Set alpha value and color list
        alpha_value = 0.1

        # # Add horizontal bars
        ax.axhspan(-0.5, 7.5, alpha=alpha_value, color='red')
        ax.axhspan(len(response_hist_patch)-8-.5, len(response_hist_patch)-.5, alpha=alpha_value, color='green')

        ##Add vertical bars
        ax.axvspan(-0.5, 7.5, alpha=alpha_value, color='red')
        ax.axvspan(len(response_hist_patch[0])-8-0.5, len(response_hist_patch[0]-.5), alpha=alpha_value, color='green')

        #alpha_value = 0.4
        colors = ['orange', 'blue']*2#, 'red', 'blue']

        # Calculate the width of each band
        band_width = len(response_hist_patch) / 4
        band_width_x = len(response_hist_patch[0]) / 4

        # Add horizontal and vertical bars
        for i in range(4):
                ax.axhspan(i*band_width-0.5, (i+1)*band_width-0.5, alpha=alpha_value, color=colors[i])
                ax.axvspan(i*band_width_x-0.5, (i+1)*band_width_x-0.5, alpha=alpha_value, color=colors[i])


        # Set alpha value and color list
        colors = ['yellow', 'purple']*4#'orange', 'yellow', 'green', 'blue', 'indigo', 'violet','black']

        # Calculate the width of each band
        band_width = len(response_hist_patch) / 8
        band_width_x = len(response_hist_patch[0]) / 8

        # Add horizontal and vertical bars
        for i in range(8):
                ax.axhspan(i*band_width-0.5, (i+1)*band_width-0.5, alpha=alpha_value, color=colors[i])
                ax.axvspan(i*band_width_x-0.5, (i+1)*band_width_x-0.5, alpha=alpha_value, color=colors[i])

        colors = ['none', 'black']*8#, 'red', 'blue']

        # Calculate the width of each band
        band_width = len(response_hist_patch) / 16
        band_width_x = len(response_hist_patch[0]) / 16

        # Add horizontal and vertical bars
        for i in range(16):
                ax.axhspan(i*band_width-0.5, (i+1)*band_width-0.5, alpha=alpha_value, color=colors[i])
                ax.axvspan(i*band_width_x-0.5, (i+1)*band_width_x-0.5, alpha=alpha_value, color=colors[i])

        # Set x and y range
        ax.set_xlim(-0.5, len(response_hist_patch[0])-0.5)
        ax.set_ylim(-0.5, len(response_hist_patch)-0.5)

        plt.tight_layout()
        plt.savefig('4D_2bin_sample_response_matrix_xb_q2_phi_t.png')
        plt.close()
        # fig, ax = plt.subplots(figsize=(14,10))

        # im = ax.imshow(response_hist_patch, origin='lower',norm=mpl.colors.LogNorm())
        # cbar = plt.colorbar(im, label='Counts')
        # ax.set(xlabel='Bins, Truth', ylabel='Bins, Observed',title='4D 2-Bin Sample Response Matrix')
        # #plt.show()
        # #save the figure
        # #plt.savefig('4D_2bin_sample_response_matrix.png')
        # plt.show()
        #plt.close()

        fig, ax = plt.subplots(figsize=(14,10))

        im = ax.imshow(response_normalized_patch, origin='lower',norm=mpl.colors.LogNorm())
        cbar = plt.colorbar(im, label='$P(E_i|C_{\mu})$')
        ax.set(xlabel='Bins, Truth', ylabel='Bins, Observed',
        title='4D 4-Bin Sample Normalized Response Matrix')
        #plt.show()
        #save the figure
        plt.savefig('4D_4bin_sample_normalized_response_matrix.png')
        plt.close()


        # Create figure with two subplots with different heights
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 10),
                                gridspec_kw={'height_ratios': [5, 1]}, sharex=True)

        # Plot for truth, observed, and unfolded distributions
        ax1.step(bins, truth_data_patch, where='mid', lw=3, alpha=0.7, color='green', label='True (Gen.) Distribution')
        ax1.step(bins, observed_data_patch, where='mid', lw=3, alpha=0.7, color='red', label='Observed (Rec.) distribution')
        ax1.errorbar(bins, unfolded_results,
                yerr=np.sqrt(np.diag(output_sys_err_matrix)**2+np.diag(output_stat_err_matrix)**2),
                alpha=0.7, elinewidth=3, capsize=4, ls='None', marker='.', ms=10, color='blue',
                label='Unfolded Distribution')

        ax1.legend()
        #ax1.grid(True)

        # Remove x-axis label for the first subplot
        ax1.set(xlabel='', ylabel='Counts')

        # Plot for residuals
        ax2.errorbar(bins, observed_data_patch/truth_data_patch,
                alpha=0.7, elinewidth=3, capsize=4, ls='None', marker='.', ms=10, color='red',
                label='Observed Distribution Residuals')
        ax2.errorbar(bins, unfolded_results/truth_data_patch,
                #yerr=np.sqrt(np.diag(output_sys_err_matrix)**2+np.diag(output_stat_err_matrix)**2)/truth_data_patch,
                alpha=0.7, elinewidth=3, capsize=4, ls='None', marker='.', ms=10, color='blue',
                label='Unfolded Distribution Residuals')

        ax2.axhline(y=1, color='green', linestyle='--')
        ax2.set(xlabel='Unrolled Bin Number', ylabel='Ratio to\n Truth')
        #set axis limit 
        ax2.set_ylim(0.5,1.5)
        #ax2.grid(True)

        # Set a title to the figure
        plt.suptitle('4D 4-Bin Sample Unfolding', fontsize=22, y=0.92)

        # Adjust the vertical spacing between subplots
        plt.subplots_adjust(hspace=0.0)
        #save the figure
        plt.savefig('4D_4bin_sample_unfolding.png')
        #plt.show()

        # fig, ax = plt.subplots()
        # ax.step(bins, truth_data_patch, where='mid', lw=3,
        #         alpha=0.7, label='True distribution')
        # ax.step(bins, observed_data_patch, where='mid', lw=3,
        #         alpha=0.7, label='Observed distribution')
        # ax.errorbar(bins, unfolded_results,
        #         yerr=np.sqrt(np.diag(output_sys_err_matrix)**2+np.diag(output_stat_err_matrix)**2),
        #         alpha=0.7,
        #         elinewidth=3,
        #         capsize=4,
        #         ls='None', marker='.', ms=10,
        #         label='Unfolded distribution')

        # ax.set(xlabel='X bins', ylabel='Counts')
        # plt.legend()
        # plt.show()

        # #create another plot for residuals
        # fig, ax = plt.subplots()
        # ax.errorbar(bins, np.abs(observed_data_patch-truth_data_patch),
        #                 alpha=0.7,
        #                 elinewidth=3,
        #                 capsize=4,
        #                 ls='None', marker='.', ms=10,
        #                 label='Observed Distribution Residuals')
        # ax.errorbar(bins, np.abs(unfolded_results-truth_data_patch),
        #                 alpha=0.7,
        #                 elinewidth=3,
        #                 capsize=4,
        #                 ls='None', marker='.', ms=10,
        #                 label='Unfolded Distribtuion Residuals')
        # ax.set(xlabel='X bins', ylabel='Counts')
        # plt.legend()
        #plt.show()

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
                vector_ids=None):


        #Keep only the rows and columns that are in the vector_ids
        response_hist_patch = response_hist[vector_ids]
        response_hist_patch = response_hist_patch[:,vector_ids]
        observed_data_patch = observed_data[vector_ids]
        truth_data_patch = truth_data[vector_ids]

        ##ic(truth_data_patch)
        ##ic(observed_data_patch)

        #instead get truth_data_path by summing over the rows of response_hist_patch
        truth_data_patch = np.sum(response_hist_patch,axis=0)
        ##ic(truth_data_patch)
        #similar for observed_data_patch
        observed_data_patch = np.sum(response_hist_patch,axis=1)
        ##ic(observed_data_patch)

        ##ic(response_hist_patch)

        #print size of response_hist_patch
        ###ic(response_hist_patch.shape)
        observed_data_err = np.sqrt(observed_data_patch)
        efficiencies = np.ones_like(observed_data_patch, dtype=float)
        efficiencies_err = np.full_like(efficiencies, 0.1, dtype=float)
        response_hist_err_patch = np.sqrt(response_hist_patch)

        column_sums = response_hist_patch.sum(axis=0)
        # Add a small constant to avoid division by zero
        column_sums = column_sums + 1e-12
        normalization_factor = efficiencies / column_sums

        response_normalized_patch = response_hist_patch * normalization_factor
        response_err_patch_normalized = response_hist_err_patch * normalization_factor

        uniform_prior = priors.uniform_prior(num_causes=len(truth_data_patch))
        causes = np.arange(len(truth_data_patch))
        num_causes = len(truth_data_patch)
        cause_lim = np.logspace(0, 3, num_causes)
        #jeffreys_prior = priors.jeffreys_prior(cause_lim)
        ###ic(jeffreys_prior)

        ##ic('unfolding')
        unfolded_results = iterative_unfold(data=observed_data_patch,
                                    data_err=observed_data_err,
                                    response=response_normalized_patch,
                                    response_err=response_err_patch_normalized,
                                    efficiencies=efficiencies,
                                    efficiencies_err=efficiencies_err,
                                    prior=uniform_prior,
                                    #callbacks=[Logger()],
                                    callbacks=[],
                                    ts_stopping=0.00005,)

        return unfolded_results,truth_data_patch,observed_data_patch,response_hist_patch,response_normalized_patch

def unroll(x_bin, q_bin, t_bin, phi_bin, x_bins, q_bins, t_bins, phi_bins):

    unrolled_bin = x_bin*len(q_bins)*len(t_bins)*len(phi_bins)+q_bin*len(t_bins)*len(phi_bins)+phi_bin*len(t_bins)+t_bin

    return unrolled_bin

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
                files_training = pickle_filelist#[:-1]
                #files_verification = [pickle_filelist[-1]]
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


def unstich_unfold_restich(x_bins, q_bins, t_bins, phi_bins,
                           x_width=3,q_width=3,t_width=3,phi_width=3,
                           x_stride=1,q_stride=1,t_stride=1,phi_stride=1):

        # if stride = 0, change it to 1. This is a simple fix to avoid a divide by zero error
        if x_stride == 0:
                x_stride = 1
        if q_stride == 0:
                q_stride = 1
        if t_stride == 0:
                t_stride = 1
        if phi_stride == 0:
                phi_stride = 1
        ###ic("Unfolding with x-q-t-p kernel of size {}x{}x{}x{} with strides of {}x{}x{}x{}".format(x_width,q_width,t_width,phi_width,
                                                                                                     #x_stride,q_stride,t_stride,phi_stride))


        total_its = 0
        for x in range(0,len(x_bins)-(x_width-1),x_stride):
                for q in range(0,len(q_bins)-(q_width-1),q_stride):
                        for t in range(0,len(t_bins)-(t_width-1),t_stride):
                                #for phi in range(0,len(phi_bins)-(phi_width-1),phi_stride):
                                for phi in range(0, len(phi_bins), phi_stride):
                                        total_its +=1

        unfolding_matrices  = []
        stat_errors  = []
        sys_errors  = []
        v_ids = []
        iteration = 0

        # Define a directory to save your data
        save_dir = './saved_data'
        if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        for x in range(0,len(x_bins)-(x_width-1),x_stride):
                for q in range(0,len(q_bins)-(q_width-1),q_stride):
                        for t in range(0,len(t_bins)-(t_width-1),t_stride):
                                for phi in range(0,len(phi_bins)-(phi_width-1),phi_stride):
                                #for phi in range(0, len(phi_bins), phi_stride):
                                        iteration +=1
                                        x_range = x_bins[x:x+x_width]
                                        q_range = q_bins[q:q+q_width]
                                        t_range = t_bins[t:t+t_width]
                                        phi_range = phi_bins[phi:phi+phi_width]
                                #        phi_indices = [(phi + i) % len(phi_bins) for i in range(phi_width)]
                                 #       phi_range = phi_bins[phi_indices]

                                        print(x_range,q_range,t_range,phi_range,iteration,'of',total_its)
                                        bin_ids = []
                                        for x_bin in x_range:
                                                for q_bin in q_range:
                                                        for t_bin in t_range:
                                                                for phi_bin in phi_range:
                                                                        unrolled_bin = unroll(x_bin, q_bin, t_bin, phi_bin, x_bins, q_bins, t_bins, phi_bins)
                                                                        bin_ids.append(unrolled_bin)
                                        bin_ids = np.sort(bin_ids)

                                        v_ids.append(bin_ids)
                                        unfolded_results,truth_data_patch,observed_data_patch,response_hist_patch,response_normalized_patch = unfold_function(truth_data=truth_data,
                                                                                                                observed_data=observed_data,
                                                                                                                response_hist=response_hist,
                                                                                                                vector_ids=bin_ids)

                                        unfolding_matrix = unfolded_results['unfolding_matrix']
                                        unfolding_matrices.append(unfolding_matrix)

                                        ic(unfolding_matrix.shape)
                                        ic(unfolding_matrix)
                                        stat_err = np.diag(unfolded_results['stat_err'])
                                        sys_err = np.diag(unfolded_results['sys_err'])

                                        stat_errors.append(stat_err)
                                        sys_errors.append(sys_err)

                                        # # Save the data
                                        # np.save(os.path.join(save_dir, f'bin_ids_{iteration}.npy'), bin_ids)
                                        # np.save(os.path.join(save_dir, f'unfolding_matrix_{iteration}.npy'), unfolding_matrix)
                                        # np.save(os.path.join(save_dir, f'stat_err_{iteration}.npy'), stat_err)
                                        # np.save(os.path.join(save_dir, f'sys_err_{iteration}.npy'), sys_err)

                                        plotting = False
                                        if plotting:
                                                plot_distributions(np.arange(0, len(bin_ids), 1), truth_data_patch,
                                                                observed_data_patch, response_hist_patch,
                                                                response_normalized_patch, unfolded_results)


        print("done unstiching, unfolding, onto restiching")
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

                ###ic(stat_error_mat)
                ###ic(sys_error_mat)
                ###ic(unfolding_matrix)
                for i in range(unfolding_matrix.shape[0]):  # iterating over rows
                        for j in range(unfolding_matrix.shape[1]):  # iterating over columns
                                unfolded_response[element_id[i],element_id[j]] = unfolding_matrix[i][j]
                                sys_errors_response[element_id[i],element_id[j]] = sys_error_mat[i][j]
                                stat_errors_response[element_id[i],element_id[j]] = stat_error_mat[i][j]

                #ic(enlarged_unfolding_matrices)
                #save as csv file
                #np.savetxt("recon_enlarged_unfolding_matrices_{}.csv".format(count), unfolded_response, delimiter=",")
                # save unenlarged unfolding matrix
                enlarged_unfolding_matrices.append(unfolded_response)
                enlarged_sys_errors_matrices.append(sys_errors_response)
                enlarged_stat_errors_matrices.append(stat_errors_response)

        print(enlarged_unfolding_matrices)


        ###ic(enlarged_unfolding_matrices)
        ###ic(enlarged_sys_errors_matrices)
        output_matrix = combine_matrices(enlarged_unfolding_matrices,normalization=True)
        #np.savetxt("recon_unfolding_matrix{}_{}_{}_{}.csv".format(x,q,t,phi), output_matrix, delimiter=",")


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
        ###ic(output_stat_err_matrix)

        return output_matrix, output_sys_err_matrix, output_stat_err_matrix




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

        column_sums = response_hist.sum(axis=0)
        # Add a small constant to avoid division by zero
        column_sums = column_sums + 1e-12
        efficiencies = np.ones_like(observed_data, dtype=float)
        normalization_factor = efficiencies / column_sums

        response_norm = response_hist * normalization_factor

        print("done calculating response matrix")
        return truth_data, observed_data, response_hist,response_norm, bins

######################
#####################

# Get initial data
df, total_unrolled_number_of_bins, x_bins, q_bins, t_bins, phi_bins = get_data()
# Calculate globablly useful things
truth_data, observed_data, response_hist, response_norm,bins = calc_resp_matrix(df, total_unrolled_number_of_bins)


#make font size 20
# plt.rcParams.update({'font.size': 20})
# fig, ax = plt.subplots(figsize=(14,10))

# im = ax.imshow(response_hist, origin='lower')#,norm=mpl.colors.LogNorm())
# cbar = plt.colorbar(im, label='Counts')
# ax.set(xlabel='Bins, Truth', ylabel='Bins, Observed',
# title='Full Response Matrix')
# #plt.show()
# #save the figure
# plt.savefig('4D_all_bins_full_response_matrix.png')
# plt.close()

# fig, ax = plt.subplots(figsize=(14,10))

# im = ax.imshow(response_norm, origin='lower')#,norm=mpl.colors.LogNorm())
# cbar = plt.colorbar(im, label='$P(E_i|C_{\mu})$')
# ax.set(xlabel='Bins, Truth', ylabel='Bins, Observed',
# title='Normalized Full Response Matrix')
# #plt.show()

# #save the figure
# plt.savefig('4D_all_bins_normalized_full_response_matrix.png')
# plt.close()


####################
####################

##ic(x_bins)
##ic(q_bins)
#since using square kernels, need the minimum bin span
min_bin_span = min(len(x_bins),len(q_bins),len(t_bins),len(phi_bins))
##ic("Minimum bin span is {}".format(min_bin_span))

import time

# Initialize comp_time and average_err arrays
comp_time = np.zeros((min_bin_span, min_bin_span))
average_err = np.zeros((min_bin_span, min_bin_span))
true_output_matrix = None  # Variable to hold the "true" output matrix


#for kernel_size in range(min_bin_span+1,1,-1): #For example, if image is 4x4x4x4, then kernel size can be 4,3,2
kernel_size = 2
stride_max = min(kernel_size, min_bin_span - kernel_size +1)
stride_range = [0] if kernel_size == min_bin_span else range(1, stride_max + 1)  # add stride 0 if kernel size equals image size
#for stride in stride_range:
#if stride == 0 or (min_bin_span - kernel_size) % stride == 0:  # also handle stride=0 case
kernel_size = 4
stride = 1
##ic("Unfolding with x-q kernel of size {}x{} with strides of {}x{}".format(kernel_size, kernel_size, stride, stride))

start_time = time.time()

output_matrix, output_sys_err_matrix, output_stat_err_matrix = unstich_unfold_restich(x_bins, q_bins, t_bins, phi_bins,
                                                                                        x_width=kernel_size,
                                                                                        q_width=kernel_size,
                                                                                        t_width=kernel_size,
                                                                                        phi_width=kernel_size,
                                                                                        x_stride=stride,
                                                                                        q_stride=stride,
                                                                                        t_stride=stride,
                                                                                        phi_stride=stride)

unfolded_data = np.dot(output_matrix, observed_data)
plot_distributions(np.arange(0, len(bins), 1), truth_data,
                                                        observed_data, response_hist,
                                                        response_norm, unfolded_data,
                                                        output_sys_err_matrix,output_stat_err_matrix)

# Compute elapsed time and store it in comp_time array
elapsed_time = time.time() - start_time

#Save all 3 matrixes to csv files
#np.savetxt("recon_unfolding_matrix{}_{}_{}_{}.csv".format(len(x_bins),len(q_bins),len(t_bins),len(phi_bins)), output_matrix, delimiter=",")
#np.savetxt("recon_unfolding_sys_err_matrix{}_{}_{}_{}.csv".format(len(x_bins),len(q_bins),len(t_bins),len(phi_bins)), output_sys_err_matrix, delimiter=",")
#np.savetxt("recon_unfolding_stat_err_matrix{}_{}_{}_{}.csv".format(len(x_bins),len(q_bins),len(t_bins),len(phi_bins)), output_stat_err_matrix, delimiter=",")

# If this is the "true" output matrix (i.e., kernel size equals min_bin_span), store it
if kernel_size == min_bin_span and stride == 0:
        true_output_matrix = output_matrix
        basis_elapse_time = elapsed_time
        # #calcuate inverse of response matrix
        # response_matrix_inv = np.linalg.inv(response_hist)
        # #compare true_output_matrix to response matrix inverse
        # ###ic(true_output_matrix)
        # ###ic(response_matrix_inv)
        # diffs = np.abs(true_output_matrix/response_matrix_inv-1)*100
        # #replace nan with zero
        # diffs[np.isnan(diffs)] = 0
        # # round each value to nearest int
        # diffs = np.round(diffs).astype(int)
        # ##ic(diffs)
        # sys.exit()

# error_matrix = np.where(np.abs(true_output_matrix) > 1e-7,
#         (output_matrix - true_output_matrix) / true_output_matrix, 0)

# # Compute average absolute difference of all non-zero and non-nan elements in error_matrix
# # and store it in average_err array as a percent
# ##ic("Average error: ", np.nanmean(np.abs(error_matrix))*100)
# ##ic("Computation time: ", elapsed_time/basis_elapse_time)
# average_err[kernel_size-1][stride] = np.nanmean(np.abs(error_matrix))*100
# comp_time[kernel_size-1][stride] = elapsed_time/basis_elapse_time#



#make a plot showing the unfolded data, the observed data, and the truth data
# fig, ax = plt.subplots()
# ax.step(bins, truth_data, where='mid', lw=3,
#         alpha=0.7, label='True distribution')
# ax.step(bins, observed_data, where='mid', lw=3,
#         alpha=0.7, label='Observed distribution')
# ax.errorbar(bins, unfolded_data,
#         yerr=np.sqrt(np.diag(output_sys_err_matrix)**2+np.diag(output_stat_err_matrix)**2),
#         alpha=0.7,
#         elinewidth=3,
#         capsize=4,
#         ls='None', marker='.', ms=10,
#         label='Unfolded distribution')
# #set the x and y axis limits
# ax.set_xlim(0, total_unrolled_number_of_bins)
# #ax.set_ylim(0,8000)
# plt.show()

sys.exit()

plot_dists = False
if plot_dists:
        ##ic("Average error matrix:")
        ##ic(average_err)
        ##ic("Computation time matrix:")
        ##ic(comp_time)
        import numpy.ma as ma
        import matplotlib.pyplot as plt

        # Define your colormap
        cmap = plt.cm.viridis
        cmap.set_bad(color='white')

        # Create the masked arrays
        comp_time_masked = ma.masked_where(np.isnan(comp_time) | (comp_time == 0), comp_time)
        average_err_masked = ma.masked_where(np.isnan(average_err) | (average_err == 0), average_err)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        x_ticks = np.arange(min_bin_span+1)  # consider additional column for stride=0
        y_ticks = np.arange(min_bin_span) + 1

        # Modify imshow calls to use masked array and the colormap with white for bad values
        c1 = axs[0].imshow(comp_time_masked.T, cmap=cmap, interpolation='nearest', origin='lower')
        fig.colorbar(c1, ax=axs[0])
        axs[0].set_title('Computation Time')
        axs[0].set_xlabel('Kernel Size')
        axs[0].set_ylabel('Stride')
        axs[0].set_xticks(np.arange(min_bin_span))
        axs[0].set_yticks(np.arange(min_bin_span+1))
        axs[0].set_xticklabels(y_ticks)
        axs[0].set_yticklabels(x_ticks)

        c2 = axs[1].imshow(average_err_masked.T, cmap=cmap, interpolation='nearest', origin='lower')
        fig.colorbar(c2, ax=axs[1])
        axs[1].set_title('Average Error')
        axs[1].set_xlabel('Kernel Size')
        axs[1].set_ylabel('Stride')
        axs[1].set_xticks(np.arange(min_bin_span))
        axs[1].set_yticks(np.arange(min_bin_span+1))
        axs[1].set_xticklabels(y_ticks)
        axs[1].set_yticklabels(x_ticks)

        plt.show()






