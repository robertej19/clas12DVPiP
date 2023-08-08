import pandas as pd
import numpy as np
import os, sys
from icecream import ic
import matplotlib.pyplot as plt

from utils import filestruct, const, make_histos

# import time module
import time

import os
from PIL import Image
import numpy as np
import re

def get_data(df,xbin,qbin,tbin,phibin):
      
        bins_Q2,bins_xB, bins_t1, bins_phi1 = xbin,qbin,tbin,phibin


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

        bins_Q2,bins_xB, bins_t1, bins_phi1
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

######################
#####################




def fit_function(phi,A,B,C):
    #A + B*np.cos(2*phi) +C*np.cos(phi)
    rads = phi*np.pi/180
    #return (A * np.exp(-x/beta) + B * np.exp(-1.0 * (x - mu)**2 / (2 * sigma**2)))
    #A = T+L, B=TT, C=LT
    #A = black, B=blue, C=red
    return A + B*np.cos(2*rads) + C*np.cos(rads)

def get_image_params(image_name):
    # extract params from image name using regular expression
    match = re.search(r"\((.*?), (.*?), (.*?)\)", image_name)

    #Reduced Cross Section in bin ($x_{B,min} Q^2_{min} t_{min}$)=(0.58, 4.5, 1.5).png

    if match:
        return float(match.group(1).split("(")[-1]), float(match.group(2)), float(match.group(3))
    else:
        raise ValueError(f"Image name {image_name} is not in the expected format.")

def get_images_dict(t, dir_path="."):
    # get list of all image files in directory
    image_files = [f for f in os.listdir(dir_path) if f.endswith(".png")]

    # filter for images with the correct t value
    t_images = [f for f in image_files if get_image_params(f)[2] == t]

    # create a dict of images with keys (x_B, Q2)
    images_dict = {(get_image_params(img)[0], get_image_params(img)[1]): img for img in t_images}

    return images_dict

def create_image_grid(image_dict, xBbins, Q2bins, dir_path="."):
    # load images
    images = {k: Image.open(os.path.join(dir_path, v)) for k, v in image_dict.items()}

    # assume all images are the same size
    img_width, img_height = next(iter(images.values())).size
    
    # cut the bottom 5% of every image
    cut_height = int(0.1* img_height)
    cropped_height = img_height - cut_height
    cut_width = int(0.15* img_width)

    cropped_width = img_width - cut_width

    # create new image
    combined = Image.new("RGB", (cropped_width * len(xBbins), cropped_height * len(Q2bins)), "white")

    # place images
    for i, xB in enumerate(xBbins):
        for j, Q2 in enumerate(reversed(Q2bins)):
            if (xB, Q2) in images:
                cropped_img = images[(xB, Q2)].crop((0, cut_width, img_width, cropped_height))
                combined.paste(cropped_img, (i * img_width, j * cropped_height))

    return combined

def main(t, xBbins, Q2bins, in_dir_path=".",out_dir_path="."):
    image_dict = get_images_dict(t, in_dir_path)

    combined = create_image_grid(image_dict, xBbins, Q2bins, in_dir_path)

    combined.save(os.path.join(out_dir_path, f"combined_t{t}.png"))

#pd.set_option('mode.chained_assignment', None)

# sys.exit()
show_plots = 0
show_xsec = 0
show_xsec_2 = 0
show_xsec_22 = 0
xerr_value = 0
plot_ylabel = 0 

combine_plots = 1
output_image_dir = "plot_t1_with_unfolding/"
plot_corrs = 0

PhysicsConstants = const.PhysicsConstants()


fs = filestruct.fs()

bins_x,bins_q,bins_t,bins_p = fs.xBbins, fs.Q2bins, fs.tbins, fs.phibins
# binned_outb_exp = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/outb/exp/final_f18_outb_exp_binned_with_area.pkl"
# binned_inb_exp = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/exp/final_f18_inb_exp_binned_with_area.pkl"

binned_exp = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/exp/singles_t1/final_f18_inb_exp_binned_with_area_t1.pkl"
#binned_exp = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/exp/with_area_binned_final_inbending_exclusive_t2.pkl"
binned_rec = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/rec/final_f18_inb_rec_binned_t1.pkl"
#binned_rec = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/rec/final_f18_inb_rec_binned_t2.pkl"
binned_gen = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/gen/final_f18_inb_gen_binned.pkl"
binned_gen_alt = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/gen_wq2_cut/final_f18_inb_gen_binned.pkl"


df_errs = pd.read_pickle("rad_recon_uncertainty_df.pkl")


df_exp = pd.read_pickle(binned_exp)
df_rec = pd.read_pickle(binned_rec)
df_gen = pd.read_pickle(binned_gen)

print(df_exp.columns.values)
print(df_rec.columns.values)

print(df_gen)


# Set the index in both dataframes to be the keys
df_exp.set_index(['tmin', 'pmin', 'xmin', 'qmin', 'tmax', 'pmax', 'xmax', 'qmax'], inplace=True)
#df_rec.set_index(['tmin', 'pmin', 'xmin', 'qmin', 'tmax', 'pmax', 'xmax', 'qmax'], inplace=True)
#df_gen.set_index(['tmin', 'pmin', 'xmin', 'qmin', 'tmax', 'pmax', 'xmax', 'qmax'], inplace=True)
df_errs.set_index(['tmin', 'pmin', 'xmin', 'qmin', 'tmax', 'pmax', 'xmax', 'qmax'], inplace=True)

# Merge the dataframes

# Merge df1 and df2
combined_df = pd.merge(df_exp, df_rec, left_index=True, right_index=True)

# Then merge the result with df3
combined_df = pd.merge(combined_df, df_gen, left_index=True, right_index=True)
combined_df = pd.merge(combined_df, df_errs, left_index=True, right_index=True)

print(combined_df)

print(combined_df.columns.values)
#print indexes
print(combined_df.index.names)
#print unique values in index
print(combined_df.index.unique().values)




combined_df['x_bin_num'] = np.digitize(combined_df.index.get_level_values('xmin'), bins=bins_x) - 1
combined_df['q_bin_num'] = np.digitize(combined_df.index.get_level_values('qmin'), bins=bins_q) - 1
combined_df['p_bin_num'] = np.digitize(combined_df.index.get_level_values('pmin'), bins=bins_p) - 1
combined_df['t_bin_num'] = np.digitize(combined_df.index.get_level_values('tmin'), bins=bins_t) - 1


x_bins = np.arange(0,len(bins_x)-1,1) #need to subtract 1 because bins_xB includes edges, and x_bins is just the bin labels number
q_bins = np.arange(0,len(bins_q)-1,1)
t_bins = np.arange(0,len(bins_t)-1,1)
phi_bins = np.arange(0,len(bins_p)-1,1)


combined_df['observed_x'] = combined_df['x_bin_num']
combined_df['observed_q'] = combined_df['q_bin_num']
combined_df['observed_t'] = combined_df['t_bin_num']
combined_df['observed_phi'] = combined_df['p_bin_num']


unfolded_data_IBU = np.load('final_unfolded_data.npy')
output_matrix_IBU = np.load('final_output_matrix.npy')
output_sys_err_matrix_IBU = np.load('final_output_sys_err_matrix.npy')
output_stat_err_matrix_IBU = np.load('final_output_stat_err_matrix.npy')
truth_data_IBU = np.load('final_truth_data.npy')
bins_IBU = np.load('bins.npy')
total_unrolled_number_of_bins_IBU = np.load('total_unrolled_number_of_bins.npy')
observed_data_IBU = np.load('final_observed_data.npy')


rel_sys_err_IBU = np.diag(output_sys_err_matrix_IBU)/unfolded_data_IBU
rel_stat_err_IBU = np.diag(output_stat_err_matrix_IBU)/unfolded_data_IBU

#Unroll the 4D data into 1D columns of observation and truth
total_unrolled_number_of_bins = len(x_bins)*len(q_bins)*len(t_bins)*len(phi_bins)
#combined_df['unrolled_truth_bins'] = combined_df['truth_x']*len(q_bins)*len(t_bins)*len(phi_bins)+combined_df['truth_q']*len(t_bins)*len(phi_bins)+combined_df['truth_phi']*len(t_bins)+combined_df['truth_t']
combined_df['unrolled_observed_bins'] = combined_df['observed_x']*len(q_bins)*len(t_bins)*len(phi_bins)+combined_df['observed_q']*len(t_bins)*len(phi_bins)+combined_df['observed_phi']*len(t_bins)+combined_df['observed_t']


combined_df['exp_observed_data'] = combined_df['counts']
combined_df['rec_observed_data'] = combined_df['rec_counts']

combined_df = combined_df.sort_values('unrolled_observed_bins')
print(combined_df['unrolled_observed_bins'])
observed_data_rec = combined_df['rec_observed_data'].values#.reshape(-1, 1)
observed_data_exp = combined_df['exp_observed_data'].values#.reshape(-1, 1)
print(observed_data_rec.shape)
print(observed_data_rec)

unfolded_data_rec = np.dot(output_matrix_IBU, observed_data_rec)
unfolded_data_exp = np.dot(output_matrix_IBU, observed_data_exp)

# fig, ax = plt.subplots()
# ax.step(combined_df['unrolled_observed_bins'], observed_data_rec, where='mid', lw=3,
#         alpha=0.7, label='Observed distribution')
# ax.errorbar(combined_df['unrolled_observed_bins'], unfolded_data_rec,
#         yerr=rel_sys_err_IBU*unfolded_data_rec,
#         alpha=0.7,
#         elinewidth=3,
#         capsize=4,
#         ls='None', marker='.', ms=10,
#         label='Unfolded distribution')
# plt.show()

# fig, ax = plt.subplots()
# ax.step(combined_df['unrolled_observed_bins'], observed_data_exp, where='mid', lw=3,
#         alpha=0.7, label='Observed distribution')
# ax.errorbar(combined_df['unrolled_observed_bins'], unfolded_data_exp,
#         yerr=np.sqrt(rel_sys_err_IBU**2+rel_stat_err_IBU**2)*unfolded_data_exp,
#         alpha=0.7,
#         elinewidth=3,
#         capsize=4,
#         ls='None', marker='.', ms=10,
#         label='Unfolded distribution')
# plt.show()


print(unfolded_data_exp)
unfolded_data_rec_series = pd.Series(unfolded_data_rec.flatten())
unfolded_data_exp_series = pd.Series(unfolded_data_exp.flatten())

print(unfolded_data_exp_series)
combined_df = combined_df.reset_index()
# Add these series as new columns to your dataframe. The index of the series will align with the DataFrame's index.
combined_df['rec_counts_unfolded'] = unfolded_data_rec_series
combined_df['counts_unfolded'] = unfolded_data_exp_series

combined_df['unfolded_data_rec_stat_err'] = rel_sys_err_IBU*combined_df['rec_counts_unfolded']
combined_df['unfolded_data_rec_sys_err'] = rel_stat_err_IBU*combined_df['rec_counts_unfolded']
combined_df['unfolded_data_exp_stat_err'] = rel_stat_err_IBU*combined_df['counts_unfolded']
combined_df['unfolded_data_exp_sys_err'] = rel_sys_err_IBU*combined_df['counts_unfolded']

# print(combined_df['unfolded_data_exp'])

# fig, ax = plt.subplots()
# ax.step(combined_df['unrolled_observed_bins'], combined_df['rec_observed_data'], where='mid', lw=3,
#         alpha=0.7, label='Observed distribution')
# ax.errorbar(combined_df['unrolled_observed_bins'], combined_df['unfolded_data_rec'],
#         yerr=np.sqrt(combined_df['unfolded_data_rec_stat_err']**2+combined_df['unfolded_data_rec_sys_err']**2),
#         alpha=0.7,
#         elinewidth=3,
#         capsize=4,
#         ls='None', marker='.', ms=10,
#         label='Unfolded distribution')
# plt.show()

#sys.exit()

# unfolded_data = np.dot(output_matrix, observed_data)

# yerr=np.sqrt(np.diag(output_sys_err_matrix/3)**2+np.diag(output_stat_err_matrix)**2)



#print unique values of unrolled_observed_bins


# value_counts_observed= df[binned_data].value_counts()
# counts_series_observed = pd.Series(0, index=bins)
# counts_observed = counts_series_observed.add(value_counts_observed, fill_value=0)
# observed_data = counts_observed.values
# print(combined_df['rec_counts'].head(2))


# # Get initial data
# df, total_unrolled_number_of_bins, x_bins, q_bins, t_bins, phi_bins = get_data()
# # Calculate globablly useful things
# truth_data, observed_data, response_hist, bins = calc_resp_matrix(df, total_unrolled_number_of_bins)



#make a 2d histogram

# names = ['x', 'q', 'p', 't']
# for name in names:
# # Extract the 'xmin' and 'x_bin_num' data
#     xmin = combined_df.index.get_level_values('{}min'.format(name))
#     x_bin_num = combined_df['{}_bin_num'.format(name)]

#     # Create the 2D histogram
#     plt.hist2d(xmin, x_bin_num, bins=[100, 20])  # adjust bins as needed

#     # Add labels and a colorbar for clarity
#     plt.xlabel('{}min'.format(name))
#     plt.ylabel('{}_bin_num'.format(name))
#     plt.colorbar(label='Counts')

#     plt.show()






combined_df['acc_corr'] = combined_df['rec_counts'] / combined_df['gen_counts']
combined_df['acc_corr_unfolded'] = combined_df['rec_counts_unfolded'] / combined_df['gen_counts']

combined_df = combined_df[combined_df['acc_corr'] >= .005]
#make a 1d histogram of acc_corr
#set font size to be large
# plt.rcParams.update({'font.size': 18})
# fig, ax = plt.subplots(1,1, figsize=(12,6))
# ax.hist(combined_df['acc_corr'], bins=100, range=[0,0.15])
# #set x axis to range to 0 to 0.3
# ax.set_xlim(0,0.15)
# #set y axis log scale
# ax.set_yscale('log')

# ax.set_xlabel("Acceptance Correction Factor")
# ax.set_ylabel("Counts")
# #set title
# ax.set_title("Acceptance Correction Factor for All Bins")

# plt.savefig("acccorr.png")
# sys.exit()
combined_df['rad_corr_alt'] = combined_df['acc_rad'] / combined_df['acc_corr']
#error on acc_corr
combined_df['acc_corr_err'] = combined_df['acc_corr']*np.sqrt(1/combined_df['rec_counts']+1/combined_df['gen_counts'])
#error on rad_corr
combined_df['acc_rad_err'] = combined_df['acc_rad']*np.sqrt(1/combined_df['counts_rec_rad']+1/combined_df['counts_gen_rad'])
#error on rad_corr_alt
combined_df['rad_corr_alt_err'] = combined_df['rad_corr_alt']*np.sqrt(1/combined_df['counts_rec_rad']+1/combined_df['rec_counts'])

combined_df['counts_err'] = np.sqrt(combined_df['counts'])
combined_df['rec_counts_err'] = np.sqrt(combined_df['rec_counts'])
combined_df['gen_counts_err'] = np.sqrt(combined_df['gen_counts'])

combined_df['acc_corr_counts'] = combined_df['counts'] / combined_df['acc_corr']
combined_df['acc_corr_counts_unfolded'] = combined_df['counts_unfolded'] / combined_df['acc_corr_unfolded']


combined_df['acc_corr_counts_err'] = combined_df['acc_corr_counts']*np.sqrt(1/combined_df['counts']+1/combined_df['rec_counts']+1/combined_df['gen_counts'])
combined_df['acc_corr_counts_err_unfolded'] = combined_df['acc_corr_counts_unfolded']*np.sqrt(1/combined_df['counts']+1/combined_df['rec_counts']+1/combined_df['gen_counts']+(combined_df['unfolded_data_rec_stat_err']/combined_df['rec_counts_unfolded'])**2)



combined_df['acc_corr_counts_err'] = combined_df['acc_corr_counts_err'].fillna(0)
combined_df['acc_corr_counts_err'] = combined_df['acc_corr_counts_err'].replace(np.inf, 0)
combined_df['acc_corr_counts_err'] = combined_df['acc_corr_counts_err'].replace(np.nan, 0)
#do the same for acc_corr_counts_err_unfolded
combined_df['acc_corr_counts_err_unfolded'] = combined_df['acc_corr_counts_err_unfolded'].fillna(0)
combined_df['acc_corr_counts_err_unfolded'] = combined_df['acc_corr_counts_err_unfolded'].replace(np.inf, 0)
combined_df['acc_corr_counts_err_unfolded'] = combined_df['acc_corr_counts_err_unfolded'].replace(np.nan, 0)

print(combined_df.columns.values)
# combined_df['xsec'] = combined_df['acc_corr_counts']  /fs.f18_inbending_total_lumi_inv_fb *180/(3.14159) / combined_df['true_total_vol']
# combined_df['xsec_err'] = combined_df['acc_corr_counts_err'] /fs.f18_inbending_total_lumi_inv_fb*180/(3.14159)/ combined_df['true_total_vol']

ic(combined_df['acc_corr_counts'] )
ic(combined_df['nominal_xbq2_bin_volume'] )
ic(combined_df['tp_bin_volume'] )


combined_df['xsec'] = combined_df['acc_corr_counts']  /fs.f18_inbending_total_lumi_inv_fb *180/(3.14159) / combined_df['nominal_xbq2_bin_volume'] / combined_df['tp_bin_volume']
combined_df['xsec_unfolded'] = combined_df['acc_corr_counts_unfolded']  /fs.f18_inbending_total_lumi_inv_fb *180/(3.14159) / combined_df['nominal_xbq2_bin_volume'] / combined_df['tp_bin_volume']

combined_df['xsec_err'] = combined_df['acc_corr_counts_err'] /fs.f18_inbending_total_lumi_inv_fb*180/(3.14159)/ combined_df['nominal_xbq2_bin_volume'] / combined_df['tp_bin_volume']
combined_df['xsec_err_unfolded'] = combined_df['acc_corr_counts_err_unfolded'] /fs.f18_inbending_total_lumi_inv_fb*180/(3.14159)/ combined_df['nominal_xbq2_bin_volume'] / combined_df['tp_bin_volume']


#!!CONVERT PHI FROM DEG TO RAD

combined_df['y_bin'] = combined_df['qave']/(2*PhysicsConstants.electron_beam_energy*combined_df['xave']*PhysicsConstants.proton_mass)

combined_df['epsi_num'] = 1 - combined_df['y_bin'] - (combined_df['qave'])/(4*(PhysicsConstants.electron_beam_energy**2))

combined_df['epsi_denom'] = 1 - combined_df['y_bin'] + (combined_df['y_bin']**2)/2+(combined_df['qave'])/(4*(PhysicsConstants.electron_beam_energy**2))

combined_df['epsilon'] = combined_df['epsi_num'] / combined_df['epsi_denom']


combined_df['Gamma'] = (1/137/(8*3.14159)*combined_df['qave'] /
                        (((PhysicsConstants.proton_mass)**2)*((PhysicsConstants.electron_beam_energy)**2))*
                        ((1-combined_df['xave'])/(combined_df['xave']**3)) *(1/(1-combined_df['epsilon']))
)


#pi2_fact = 2*3.14159
pi2_fact = 1


combined_df['xsec_red'] = combined_df['xsec'] *pi2_fact/combined_df['Gamma']/combined_df['rad_corr_alt']
combined_df['xsec_red_unfolded'] = combined_df['xsec_unfolded'] *pi2_fact/combined_df['Gamma']/combined_df['rad_corr_alt']
combined_df['folding_ratio'] = (combined_df['xsec_red_unfolded']/combined_df['xsec_red'])**(1)
combined_df['folding_ratio_delta'] = np.sqrt((combined_df['xsec_red_unfolded']/combined_df['xsec_red'] - 1)**2)
#print mean of folding ratio
print(combined_df['folding_ratio_delta'].mean())
print(combined_df['folding_ratio'].mean())

# # Group by 'pmin' and calculate the mean of 'folding_ratio_delta'
# names = ['$\phi$', '$Q^2$', 't', '$x_B$']
# for name,vari in zip(names,['p','q','t','x']):
#     combined_df['{}ave2'.format(vari)] = (combined_df['{}min'.format(vari)] + combined_df['{}max'.format(vari)]) / 2

#     first_pave = combined_df['{}ave2'.format(vari)].iloc[0]

#     # Print all values of "folding_ratio" for the first group
#     # first_group_folding_ratio = combined_df[combined_df['{}ave2'.format(vari)] == first_pave]['folding_ratio']
#     # ic(first_group_folding_ratio)
#     # #make histogram of folding ratio for each bin
#     # #set font size to be large
#     # plt.rcParams.update({'font.size': 20})
#     # fig, ax = plt.subplots(1,1, figsize=(12,6))
#     # ax.hist(first_group_folding_ratio, bins=100)#, range=[0,2])
#     # plt.show()
#     # print(first_group_folding_ratio.mean())
#     # print(first_group_folding_ratio.std())
#     # sys.exit()
#     grouped_df = combined_df.groupby('{}ave2'.format(vari)).agg({'folding_ratio': ['mean', 'std']}).reset_index()

#     # Flatten the multi-index columns
#     grouped_df.columns = ['_'.join(col).strip() for col in grouped_df.columns.values]

#     print(grouped_df)

#     # Create the scatter plot
#     #increase font size
#     plt.rcParams.update({'font.size': 20})
#     plt.figure(figsize=(14, 6))
#     plt.errorbar(grouped_df['{}ave2_'.format(vari)], grouped_df['folding_ratio_mean'], yerr=grouped_df['folding_ratio_std'], fmt='o',capsize=4)
#     #plt.scatter(grouped_df['{}ave2'.format(vari)], grouped_df['folding_ratio_delta'])
#     plt.xlabel('{} Bin Center'.format(name))
#     plt.ylabel('Normalized Difference Between \n Cross Section Results')
#     plt.title('Normalized Mean Difference in Cross Section Results for {} Bins'.format(name))
#     #plt.grid(True)
#     #plt.show()
#     #save the figure
#     plt.ylim(.4,2)
#     plt.savefig('folding_ratio_{}.png'.format(vari))
#     #close the figure
#     plt.close()
# sys.exit()

# #plot histogram of folding ratio
# #set font size to be large
# #select only bins with pmin<90
# #combined_df = combined_df[combined_df['tave'] < .4]
# plt.rcParams.update({'font.size': 20})
# fig, ax = plt.subplots(1,1, figsize=(12,6))
# ax.hist(combined_df['folding_ratio_delta'], bins=100, range=[0,1])
# #draw a vertical line at 1
# ax.axvline(x=1, color='r', linestyle='--')
# #make a scatter plot of folding ratio vs tave
# #ax.scatter(combined_df['tave'], combined_df['folding_ratio'], s=10)
# #label x axis of ratio of unfolded to bin-by-bin cross sections
# ax.set_xlabel("Ratio of Unfolded to Bin-by-Bin Cross Sections")
# #label y axis of counts
# ax.set_ylabel("Counts")
# #set title
# ax.set_title("Ratio of Cross Section Results for All Bins")
# # #set x axis to range to 0 to 0.3
# ax.set_xlim(0,1)
# #set y axis range to 0.1 to 10
# #ax.set_ylim(0.5,5)
# # #set y axis log scale
# #ax.set_yscale('log')
# # show plot
# plt.show()
# sys.exit()


combined_df['xsec_red_err'] = combined_df['xsec_err'] *pi2_fact/combined_df['Gamma']
combined_df['xsec_red_err_unfolded'] = combined_df['xsec_err_unfolded'] *pi2_fact/combined_df['Gamma']

combined_df['acc_corr_counts_err_alt'] = combined_df['acc_corr_counts']*np.sqrt(1/combined_df['counts'])#+1/combined_df['rec_counts']+1/combined_df['gen_counts'])

#combined_df['xsec_err_alt'] = combined_df['acc_corr_counts_err_alt']  /fs.f18_inbending_total_lumi_inv_fb*180/(3.14159) / combined_df['true_total_vol']
combined_df['xsec_err_alt'] = combined_df['acc_corr_counts_err_alt']  /fs.f18_inbending_total_lumi_inv_fb*180/(3.14159) / combined_df['nominal_xbq2_bin_volume'] / combined_df['tp_bin_volume'] / combined_df['rad_corr_alt']*pi2_fact/combined_df['Gamma']


combined_df['xsec_red_err_alt'] = combined_df['xsec_err_alt'] 

combined_df = combined_df[combined_df['counts'] > 0]
combined_df = combined_df[combined_df['counts_unfolded'] > 0]

# combined_df = combined_df[combined_df['rec_counts'] >= 2]
# combined_df = combined_df[combined_df['gen_counts'] >= 2]
combined_df = combined_df[combined_df['acc_corr'] >= .005]
combined_df = combined_df[combined_df['acc_corr_unfolded'] >= .005]

combined_df = combined_df[combined_df['rad_corr_alt'] >= .5]
combined_df = combined_df[combined_df['rad_corr_alt'] <= 2]
#combined_df = combined_df[combined_df['volume_ratio'] >= .3]#dont want to mess around with bin volume effects too much
#make histogram of volume_ratio
#set font size to be large
# plt.rcParams.update({'font.size': 20})
# fig, ax = plt.subplots(1,1, figsize=(12,6))
# ax.hist(combined_df['rad_corr_alt'], bins=100, range=[0,2])
# plt.show()
# sys.exit()

# 45na-nom
# 55na-nom
# nom-nom

combined_df['sys_uncert_45na'] = np.sqrt((1-combined_df['45na-nom'])**2)
combined_df['sys_uncert_55na'] = np.sqrt((1-combined_df['55na-nom'])**2)
#calculate average between 45na and 55na
combined_df['sys_uncert_recon'] = (combined_df['sys_uncert_45na']+combined_df['sys_uncert_55na'])/2
combined_df['sys_uncert_acc-corr'] = np.sqrt((1-combined_df['nom-nom'])**2)
combined_df['sys_uncert_rad'] = combined_df['rad_corr_alt_err']

#combined sys error

combined_df['rel_sys_unfolding_uncert'] = combined_df['unfolded_data_exp_sys_err']/combined_df['counts_unfolded']

combined_df['total_sys_uncert_unfolded'] = combined_df['xsec_red_unfolded']*np.sqrt(combined_df['sys_uncert_recon']**2+combined_df['sys_uncert_acc-corr']**2+combined_df['sys_uncert_rad']**2+(combined_df['unfolded_data_exp_sys_err']/combined_df['counts_unfolded'])**2)

combined_df['total_uncert_unfolded'] = np.sqrt(combined_df['total_sys_uncert_unfolded']**2+combined_df['xsec_red_err_unfolded']**2)


print(combined_df.columns.values)

uncertainties = ['sys_uncert_recon', 'sys_uncert_acc-corr', 'sys_uncert_rad', 'rel_sys_unfolding_uncert']

combined_df = combined_df[~(combined_df[uncertainties] > 0.5).any(axis=1)]

# # calculate and print mean and median for each column
# colors = ['steelblue', 'firebrick', 'purple', 'green']
# #increase fontsize
# plt.rcParams.update({'font.size': 20})
# plt.figure(figsize=(14,10))

# for color, uncertainty in zip(colors, uncertainties):
#     mean_value = combined_df[uncertainty].mean()
#     median_value = combined_df[uncertainty].median()

#     # format as percentage
#     mean_percentage = "{:.1f}%".format(mean_value * 100)
#     median_percentage = "{:.1f}%".format(median_value * 100)

#     print(f'For {uncertainty}:')
#     print(f'Mean = {mean_percentage}')
#     print(f'Median = {median_percentage}\n')

#     # plot histogram
#     plt.hist(combined_df[uncertainty]*100, bins=50, color=color, alpha=0.5, label=uncertainty)

# plt.title('Histogram of Uncertainties')
# plt.xlabel('Uncertainty %')
# #set xrange to 30
# plt.xlim(0,30)
# plt.ylabel('Number of Bins')
# plt.legend(loc='upper right')
# #plt.grid(True)
# plt.show()







# get mean and median value for sys_uncert_recon




combined_df = combined_df.reset_index()

#save as pickle file
combined_df.to_pickle("full_cross_section_clas12_unfolded.pkl")


# combined_df.to_csv("combined_df_3.csv")
# sys.exit()

# for index, row in combined_df.iterrows():
#     print('tmin:', row['tmin'], 'pmin:', row['pmin'], 'xmin:', row['xmin'], 'qmin:', row['qmin'],
#           'counts:', row['counts'], 'rec_counts:', row['rec_counts'], 'gen_counts:', row['gen_counts'])



if show_plots:
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Group the dataframe by the specific columns
    groups = combined_df.groupby(['xmin', 'qmin'])

    # Iterate through each group to create a separate heatmap
    for name, group in groups:
        plt.figure()  # create new figure for each group

        # Create a pivot table with 'tmin'-'tmax' as the index, 'pmin'-'pmax' as the columns, and 'rec_counts' as the values
        pivot_table = group.pivot(index='tmin', columns='pmin', values='rec_counts')

        # Generate a mask for the upper triangle (optional)
        #mask = np.triu(np.ones_like(pivot_table, dtype=bool))



        # Create a heatmap from the pivot table
        sns.heatmap(pivot_table, cmap='viridis', square=True, linewidths=.5, cbar_kws={"shrink": .5})

        plt.title(f'Heatmap for xmin = {name[0]}, qmin = {name[1]}')
        plt.xlabel('Pmin-Pmax range')
        plt.ylabel('Tmin-Tmax range')
        plt.gca().invert_yaxis()
        plt.show()

import matplotlib.lines as mlines
from scipy.optimize import curve_fit

clas_df = pd.read_pickle("3_calculate_xsec/CLAS6_struct_funcs_raw.pkl")
clas_df.replace(to_replace=r'−', value='-', regex=True, inplace=True)

for column in clas_df.columns:
    #convert type into float
    clas_df[column] = clas_df[column].astype(float)
print(clas_df.columns.values)

print(clas_df.head(3))
#['Q2_C6' 'xB_C6' 't_C6' 'tel_C6' 'telstat_C6' 'telsys_C6' 'lt_C6'
#'ltstat_C6' 'ltsys_C6' 'tt_C6' 'ttstat_C6' 'ttsys_C6']
#calc rel sys uncert for all 3 types
clas_df['rel_telsys_uncert'] = np.abs(clas_df['telsys_C6']/clas_df['tel_C6'])*100
clas_df['rel_ltsys_uncert'] = np.abs(clas_df['ltsys_C6']/clas_df['lt_C6'])*100
clas_df['rel_ttsys_uncert'] = np.abs(clas_df['ttsys_C6']/clas_df['tt_C6'])*100

# # #now make histograms for these 3, and color code them
# # #set font size to be large
# # plt.rcParams.update({'font.size': 20})
# # fig, ax = plt.subplots(1,1, figsize=(12,6))
# # ax.hist(clas_df['rel_telsys_uncert'], bins=100, range=[0,100],color='red', alpha=0.5, label='Tel')
# # #ax.hist(clas_df['rel_ltsys_uncert'], bins=100, range=[0,1000],color='blue', alpha=0.5, label='LT')
# # #ax.hist(clas_df['rel_ttsys_uncert'], bins=100, range=[0,1000],color='green', alpha=0.5, label='TT')
# # plt.show()
# # sys.exit()


clas_dtp = pd.read_csv("3_calculate_xsec/xs_clas6.csv")
#set the columns
#the columns are the first row
#clas_dtp.columns = clas_dtp.iloc[0]

clas_dtp['rel'] =clas_dtp['sys']/clas_dtp['dsdtdp']*100
#make histogram of clas_dtp['sys]/clas_dtp['dsdtdp']
#set font size to be large
# # plt.rcParams.update({'font.size': 20})
# # fig, ax = plt.subplots(1,1, figsize=(12,6))
# # ax.hist(clas_dtp['rel'], bins=100)#, range=[0,0.15])
# # print(clas_dtp['rel'].mean())
# # print(clas_dtp['rel'].median())
# # plt.show()
# # sys.exit()


print(clas_dtp)

def match_row(row):
    # Identify the rows in clas_df that fall within the ranges specified in row
    matching_rows = clas_dtp[
        (clas_dtp['q'] > row['qmin']) & (clas_dtp['q'] < row['qmax']) & 
        (clas_dtp['x'] > row['xmin']) & (clas_dtp['x'] < row['xmax']) &
        (clas_dtp['t'] > row['tmin']) & (clas_dtp['t'] < row['tmax']) &
        (clas_dtp['p'] > row['pmin']) & (clas_dtp['p'] < row['pmax'])
    ]

    # If no matching rows, return NaN
    if matching_rows.empty:
        return np.nan, np.nan, np.nan
    #print(matching_rows)

    # If there are matching rows, calculate average of dsdtdp, stat, and sys
    avg_dsdtdp = matching_rows['dsdtdp'].mean()
    avg_stat = matching_rows['stat'].mean()
    avg_sys = matching_rows['sys'].mean()

    return avg_dsdtdp, avg_stat, avg_sys

# # # # # # # # # # # Apply match_row function to each row in combined_df
# # # # # # # # # # combined_df[['avg_dsdtdp', 'avg_stat', 'avg_sys']] = combined_df.apply(match_row, axis=1, result_type='expand')

# # # # # # # # # # #make histogram of avg_dsdtdp
# # # # # # # # # # # #set font size to be large
# # # # # # # # # # # plt.rcParams.update({'font.size': 20})
# # # # # # # # # # # fig, ax = plt.subplots(1,1, figsize=(12,6))
# # # # # # # # # # # ax.hist(combined_df['avg_dsdtdp'], bins=100)#, range=[0,0.15])
# # # # # # # # # # # plt.show()

# # # # # # # # # # # remove bins where tav is greater than 0.4
# # # # # # # # # # #combined_df = combined_df[combined_df['tave'] < 1]
# # # # # # # # # # simple_ratio = True
# # # # # # # # # # if simple_ratio:
# # # # # # # # # #     combined_df['ratio_c612'] = combined_df['xsec_red_unfolded'] / combined_df['avg_dsdtdp']
# # # # # # # # # #     combined_df['total_uncertainty_c612'] = np.sqrt((combined_df['avg_sys']/combined_df['avg_dsdtdp'])**2 + (combined_df['total_uncert_unfolded']/combined_df['xsec_red_unfolded'])**2)

# # # # # # # # # #     # Filter rows that have non-null ratio and total_uncertainty values
# # # # # # # # # #     filtered_df = combined_df[combined_df[['ratio_c612', 'total_uncertainty_c612']].notnull().all(axis=1)]

# # # # # # # # # #     # Create a new x-axis
# # # # # # # # # #     x_axis = np.arange(len(filtered_df))

# # # # # # # # # #     # Plot
# # # # # # # # # #     plt.errorbar(x_axis, filtered_df['ratio_c612'], yerr=filtered_df['total_uncertainty_c612'], fmt='o')
# # # # # # # # # #     plt.xlabel('Continuous Index')
# # # # # # # # # #     plt.ylabel('Ratio')
# # # # # # # # # #     plt.title('Ratio of xsec_red_unfolded to avg_dsdtdp')
# # # # # # # # # #     #log y axis
# # # # # # # # # #     plt.yscale('log')
# # # # # # # # # #     #y axis 0.1 to 10
# # # # # # # # # #     plt.ylim(.5,2)
# # # # # # # # # #     plt.show()
# # # # # # # # # #     #create histogram of log_ratio_c612
# # # # # # # # # #     #set font size to be large
# # # # # # # # # #     plt.rcParams.update({'font.size': 20})
# # # # # # # # # #     fig, ax = plt.subplots(1,1, figsize=(12,6))
# # # # # # # # # #     ax.hist(combined_df['ratio_c612'], bins=100)#, range=[0,0.15])
# # # # # # # # # #     #set xlim to 0.5 to 2
# # # # # # # # # #     ax.set_xlim(.5,2)
# # # # # # # # # #     plt.show()
# # # # # # # # # #     #print mean of ratio
# # # # # # # # # #     print(combined_df['ratio_c612'].mean())
# # # # # # # # # #     print(combined_df['ratio_c612'].std())
# # # # # # # # # #     print(combined_df['ratio_c612'].median())
# # # # # # # # # # else:
# # # # # # # # # #     combined_df['log_ratio_c612'] = np.log2(combined_df['xsec_red_unfolded'] / combined_df['avg_dsdtdp'])
# # # # # # # # # #     combined_df['total_uncertainty_c612'] = np.sqrt((combined_df['avg_sys']/combined_df['avg_dsdtdp'])**2 + (combined_df['total_uncert_unfolded']/combined_df['xsec_red_unfolded'])**2)

# # # # # # # # # #     # Filter rows that have non-null ratio and total_uncertainty values
# # # # # # # # # #     filtered_df = combined_df[combined_df[['log_ratio_c612', 'total_uncertainty_c612']].notnull().all(axis=1)]

# # # # # # # # # #     # Create a new x-axis
# # # # # # # # # #     x_axis = np.arange(len(filtered_df))

# # # # # # # # # #     # Plot
# # # # # # # # # #     plt.errorbar(x_axis, filtered_df['log_ratio_c612'], yerr=filtered_df['total_uncertainty_c612'], fmt='o')
# # # # # # # # # #     plt.xlabel('Continuous Index')
# # # # # # # # # #     plt.ylabel('Log Ratio')
# # # # # # # # # #     #y range -1 to 1
# # # # # # # # # #     plt.ylim(-1,1)
# # # # # # # # # #     plt.title('Log Ratio of xsec_red_unfolded to avg_dsdtdp')
# # # # # # # # # #     plt.show()

# # # # # # # # # #     #create histogram of log_ratio_c612
# # # # # # # # # #     #set font size to be large
# # # # # # # # # #     plt.rcParams.update({'font.size': 20})
# # # # # # # # # #     fig, ax = plt.subplots(1,1, figsize=(12,6))
# # # # # # # # # #     ax.hist(combined_df['log_ratio_c612'], bins=25, range=[-1,1])

# # # # # # # # # #     plt.show()

# # # # # # # # # # # combined_df['q_avg'] = (combined_df['tmax'] + combined_df['tmin']) / 2

# # # # # # # # # # # # Filter rows that have non-null 'log_ratio_c612' and 'q_avg' values
# # # # # # # # # # # filtered_df = combined_df[combined_df[['ratio_c612', 'q_avg']].notnull().all(axis=1)]

# # # # # # # # # # # # Create a 2D histogram
# # # # # # # # # # # plt.hist2d(filtered_df['q_avg'], filtered_df['ratio_c612'], bins=[50, 50], cmap='plasma')

# # # # # # # # # # # # Add a colorbar
# # # # # # # # # # # plt.colorbar(label='Counts')

# # # # # # # # # # # # Add labels and a title
# # # # # # # # # # # plt.xlabel('Mean of qmax and qmin')
# # # # # # # # # # # plt.ylabel('Log Ratio')
# # # # # # # # # # # plt.title('2D Histogram of Log Ratio vs Mean of qmax and qmin')

# # # # # # # # # # # plt.show()

# # # # # # # # # # import scipy.stats as stats

# # # # # # # # # # # your measurements and uncertainties
# # # # # # # # # # measurements = filtered_df['ratio_c612'].values
# # # # # # # # # # uncertainties = filtered_df['total_uncertainty_c612'].values

# # # # # # # # # # # expected value
# # # # # # # # # # expected = 1.0

# # # # # # # # # # # calculate chi-square
# # # # # # # # # # chi_square = np.sum(((measurements - expected) / uncertainties)**2)

# # # # # # # # # # # degrees of freedom is number of measurements - 1
# # # # # # # # # # dof = len(measurements) - 1

# # # # # # # # # # # find the p-value
# # # # # # # # # # p_value = 1 - stats.chi2.cdf(chi_square, dof)

# # # # # # # # # # print(f"Chi-square: {chi_square}, p-value: {p_value}")

# # # # # # # # # # # calculate mean and standard deviation
# # # # # # # # # # mean = np.mean(measurements)
# # # # # # # # # # std_dev = np.std(measurements)

# # # # # # # # # # # number of measurements
# # # # # # # # # # n = len(measurements)

# # # # # # # # # # # z-score for 90% confidence
# # # # # # # # # # z_score = stats.norm.ppf(0.95)  # for two-sided 90% confidence interval

# # # # # # # # # # # calculate confidence interval
# # # # # # # # # # lower_bound = mean - z_score * (std_dev / np.sqrt(n))
# # # # # # # # # # upper_bound = mean + z_score * (std_dev / np.sqrt(n))

# # # # # # # # # # print(f"90% confidence interval: ({lower_bound}, {upper_bound})")
# # # # # # # # # # sys.exit()

if plot_corrs:

    # grouping by 'xmin', 'qmin', 'tmin'
    groups = combined_df.groupby(['xmin', 'qmin', 'tmin'])



    ind = 0
    for name, group in groups:
        #only plot if there are more than 10 bins

        print(clas_df[['Q2_C6', 'xB_C6', 't_C6']].dtypes)
        print(group[['qmin', 'xmin', 'tmin', 'qmax', 'xmax', 'tmax']].dtypes)


        print(clas_df['Q2_C6'])
        print(group['qmin'].values[0])


        print(group['qmin'])

        mask = (clas_df['Q2_C6'] >= group['qmin'].values[0]) & (clas_df['Q2_C6'] <= group['qmax'].values[0]) & \
       (clas_df['xB_C6'] >= group['xmin'].values[0]) & (clas_df['xB_C6'] <= group['xmax'].values[0]) & \
       (clas_df['t_C6'] >= group['tmin'].values[0]) & (clas_df['t_C6'] <= group['tmax'].values[0])

        filtered_df = clas_df[mask]


        if len(group) < 5:
            print("skipping")
            continue
        # Also need to skip if there is no data between 'pave' = 100 and 'pave' = 260
        # select the elements with 100<pave<260
        filtered_group = group[(group['pave'] > 100) & (group['pave'] < 260)]

        if len(filtered_group) == 0:
            print("skipping")
            continue
        if True:
            print("PLOTTING")
            plt.rcParams["font.size"] = "30"
            plt.figure(figsize=(20,14))

            slabel = "Stat. Err. from Sim."
            elabel = "Stat. Err. from Exp."
            #plot with larger marker size
            #for showing different uncertainties
            #plt.errorbar(group['pave'], group['xsec_red'], yerr=group['xsec_red_err'],fmt='r+', markersize=50,label=#slabel)
            #plot again but with red error bars
            xerr_value = 10
            plt.errorbar((group['pmin']+group['pmax'])/2, group['rad_corr_alt'], xerr=xerr_value,yerr=group['rad_corr_alt_err'],fmt='k.',  markersize=5,label="CLAS12 Data")#elabel)

            



            #blue_line = mlines.Line2D([], [], color='r', marker='None', markersize=10, linestyle='-', label=slabel)
            #red_line = mlines.Line2D([], [], color='k', marker='None', markersize=10, linestyle='-', label=elabel)


            plt.xlabel('Lepton-Hadron Angle $\phi$')
            plt.ylabel('Cross Section Ratio')
            #pltt = 'Reduced Cross Section in bin ({})={}'.format(r'$x_{B,min} Q^2_{min} t_{min}$',str(name))
            pltt = '({})={}'.format(r'$x_{B,min} Q^2_{min} t_{min}$',str(name))
            plt.title(pltt)
            #set y limits
            plt.ylim([.7,1])
            #plt.show()
            
            plt.savefig("alt_cor/"+pltt+".png")
            plt.close()

# merged_df['45na-nom'] = merged_df['acc_45na']/merged_df['acc_nominal_1']
# merged_df['55na-nom'] = merged_df['acc_55na']/merged_df['acc_nominal_1']
# merged_df['rad-nom'] = merged_df['acc_rad']/merged_df['acc_nominal_1']
# merged_df['45na-55na'] = merged_df['acc_45na']/merged_df['acc_55na']
# merged_df['nom-nom'] = merged_df['acc_nominal_1']/merged_df['acc_nominal_2']

# #now take ratios of ratios
# merged_df['45na-nom-nom-nom'] = merged_df['45na-nom']/merged_df['nom-nom']
# merged_df['55na-nom-nom-nom'] = merged_df['55na-nom']/merged_df['nom-nom']
# merged_df['rad-nom-nom-nom'] = merged_df['rad-nom']/merged_df['nom-nom']


    # loop through the groups
    #groups = combined_df.groupby(['xmin', 'qmin', 'tmin'])


        # access the corresponding subplot
        

"""
    ax.set_title(f'Subplot for (x,q) = ({xmin_idx}, {qmin_idx})')
    ax.set_xlabel('(x)')
    ax.set_ylabel('(q)')
"""

if show_xsec_2:
    # first, grouping by 'tmin'
    tmin_groups = combined_df.groupby(['tmin'])
    
    # for each 'tmin' group, group again by 'xmin' and 'qmin'
    for tmin_name, tmin_group in tmin_groups:
        # grouping by 'xmin', 'qmin'
        groups = tmin_group.groupby(['xmin', 'qmin'])

        # determine the number of unique values for 'xmin' and 'qmin'
        unique_xmin = tmin_group['xmin'].nunique()
        unique_qmin = tmin_group['qmin'].nunique()
        
        fig, axes = plt.subplots(unique_xmin, unique_qmin, figsize=(15, 15), sharex='col', sharey='row')

        # Enumerate over the groups to get group index and group data
        for (name, group), ax in zip(groups, axes.flatten(order='F')):
            # Check if group is not empty
            if not group.empty:
                # Your plot command
                ax.errorbar(group['pave'], group['xsec_red'], xerr=xerr_value, yerr=group['total_uncert'], fmt='r.',  markersize=5, elinewidth=5)
                ax.set_title(f'(x={name[0]}, q={name[1]})')  # Set subplot title
                ax.set_xlabel('q')  # Set x-axis label
                ax.set_ylabel('x')  # Set y-axis label
                ax.set_ylim(bottom=0)  # Set lower y limit

        # Remove empty subplots
        #for i in range(len(groups), unique_qmin*unique_xmin):
        #    fig.delaxes(axes.flatten(order='F')[i])
        
        fig.suptitle(f'tmin = {tmin_name}', fontsize=16)
        plt.tight_layout()
        plt.show()




if show_xsec_22:



    # grouping by 'xmin', 'qmin', 'tmin'
    groups = combined_df.groupby(['xmin', 'qmin', 'tmin'])

    
    # determine the number of unique values for 'xmin' and 'qmin'
    unique_xmin = combined_df['xmin'].nunique()
    unique_qmin = combined_df['qmin'].nunique()
    
    ind = 0
    for name, group in groups:
        #only plot if there are more than 10 bins
            # create a figure with a grid of subplots
        fig, axes = plt.subplots(unique_qmin, unique_xmin, figsize=(15, 15))

                # determine the position in the grid for this group
        qmin_idx = group['qmin'].unique()[0]
        xmin_idx = group['xmin'].unique()[0]
        # get the index of unique_xmin where xmin_idx is located
        xmin_idx = np.where(fs.xBbins == xmin_idx)[0][0]
        # get the index of unique_qmin where qmin_idx is located
        qmin_idx = np.where(fs.Q2bins == qmin_idx)[0][0]

        print(qmin_idx, xmin_idx)

        ax = axes[qmin_idx, xmin_idx]
        

        print(clas_df['Q2_C6'])
        print(group['qmin'].values[0])


        print(group['qmin'])

        mask = (clas_df['Q2_C6'] >= group['qmin'].values[0]) & (clas_df['Q2_C6'] <= group['qmax'].values[0]) & \
       (clas_df['xB_C6'] >= group['xmin'].values[0]) & (clas_df['xB_C6'] <= group['xmax'].values[0]) & \
       (clas_df['t_C6'] >= group['tmin'].values[0]) & (clas_df['t_C6'] <= group['tmax'].values[0])

        filtered_df = clas_df[mask]


        if len(group) < 5:
            print("skipping")
            continue
        # Also need to skip if there is no data between 'pave' = 100 and 'pave' = 260
        # select the elements with 100<pave<260
        filtered_group = group[(group['pave'] > 100) & (group['pave'] < 260)]

        if len(filtered_group) == 0:
            print("skipping")
            continue
        else:
            print("PLOTTING")
            plt.rcParams["font.size"] = "30"
            #plt.figure(figsize=(20,14))

            slabel = "Stat. Err. from Sim."
            elabel = "Stat. Err. from Exp."
            #plot with larger marker size
            #for showing different uncertainties
            #plt.errorbar(group['pave'], group['xsec_red'], yerr=group['xsec_red_err'],fmt='r+', markersize=50,label=#slabel)
            #plot again but with red error bars
            xerr_value = 10
            ax.errorbar(group['pave'], group['xsec_red_unfolded'], xerr=xerr_value,yerr=group['total_uncert_unfolded'],fmt='r.',  markersize=5,elinewidth=5)
            #,label="CLAS12 Data")#elabel)

            ax.errorbar(group['pave'], group['xsec_red'], xerr=xerr_value,yerr=group['xsec_red_err_alt'],fmt='k.',  markersize=5,label="CLAS12 Data",elinewidth=5)#elabel)

            # plot the data on the subplot
            # ax.errorbar(group['pave'], group['xsec_red'], xerr=xerr_value, yerr=group['total_uncert'], fmt='r.', markersize=5, elinewidth=5)





            # fit the function to the data
            popt, pcov = curve_fit(fit_function, group['pave'], group['xsec_red'], sigma=group['xsec_red_err'], absolute_sigma=True)

            # print out the fit parameters
            print(f"A = {popt[0]}, B = {popt[1]}, C = {popt[2]}")

            # plot the fit
            #phis = np.linspace(group['pave'].min(), group['pave'].max(), 1000)
            phis = np.linspace(0,360,1000)
            ax.plot(phis, fit_function(phis, *popt), 'k-', label="CLAS12 Fit",linewidth=5)
            #

            #blue_line = mlines.Line2D([], [], color='r', marker='None', markersize=10, linestyle='-', label=slabel)
            #red_line = mlines.Line2D([], [], color='k', marker='None', markersize=10, linestyle='-', label=elabel)


            plt.xlabel('Lepton-Hadron Angle $\phi$')
            plt.ylabel('Reduced Cross Section (nb/$GeV^2$)')
            #pltt = 'Reduced Cross Section in bin ({})={}'.format(r'$x_{B,min} Q^2_{min} t_{min}$',str(name))
            pltt = '({})={}'.format(r'$x_{B,min} Q^2_{min} t_{min}$',str(name))
            
            plt.title(pltt)
            #plt.legend(handles=[blue_line, red_line])



            if len(filtered_df) > 0:
                mask = (clas_dtp['q'] >= group['qmin'].values[0]) & (clas_dtp['q'] <= group['qmax'].values[0]) & \
                (clas_dtp['x'] >= group['xmin'].values[0]) & (clas_dtp['x'] <= group['xmax'].values[0]) & \
                (clas_dtp['t'] >= group['tmin'].values[0]) & (clas_dtp['t'] <= group['tmax'].values[0])

                filtered_df_dtp = clas_dtp[mask]
                print(filtered_df_dtp)
                sys.exit()

                #plt.errorbar(filtered_df_dtp['p'], filtered_df_dtp['dsdtdp'], yerr=np.sqrt(filtered_df_dtp['stat']**2+filtered_df_dtp['sys']**2),fmt='r+', markersize=50,label='CLAS6')

                errband_width = np.sqrt(filtered_df_dtp['stat']**2+filtered_df_dtp['sys']**2).mean()

                



            # plot the CLAS6 fit if it exists
                print("PLOTTING CLAS6 FIT")

                phi = np.linspace(0, 360, 1000)  # Replace 100 with the desired number of points

                # Assuming taking the first row of the filtered DataFrame
                print("FILTERED DF IS:")
                print(group.columns.values)
                print(filtered_df)
                print("xsec values are")
                print(group['pave'])
                row = filtered_df.iloc[0]

                A = row['tel_C6']
                B = row['tt_C6']
                C = row['lt_C6']
                pi = 3.14159
                #fact = group['Gamma'].mean()/(2*pi)
                fact = 1/(2*pi)
                fact2 = fact*group['epsilon'].mean()
                fact3 = fact*np.sqrt(2*group['epsilon'].mean()*(1+group['epsilon'].mean()))
                y = fit_function(phi, A*fact, B*fact2, C*fact3)

                print(A,B,C)

                ax.plot(phi, y,'b-',label='CLAS6 Result',linewidth=5)
                #make line be 50% transparent
                #plt.plot(phi, y+errband_width,'k-',label='CLAS6 Fit',alpha=0.5)#
                #plt.plot(phi, y-errband_width,'k-',label='CLAS6 Fit',alpha=0.5)#
                
                y_bottom = y-errband_width
                y_top = y+errband_width

                ax.fill_between(phi, y_bottom, y_top, color='b', alpha=.1)#, label='CLAS6 Fit')

                #,linewidth = errband_width)

                phi = np.linspace(0, 360, 1000)  # Replace 100 with the desired number of points

                # Assuming taking the first row of the filtered DataFrame
                #print("FILTERED DF IS:")
                #print(group.columns.values)
                #print(filtered_df)
                #print("xsec values are")
                #print(group['pave'])
                # # row = filtered_df.iloc[0]

                # # A = row['tel_C6']+np.sqrt(row['telsys_C6']**2+row['telsys_C6']**2)
                # # B = row['tt_C6']+np.sqrt(row['ttsys_C6']**2+row['ttsys_C6']**2)
                # # C = row['lt_C6']+np.sqrt(row['ltsys_C6']**2+row['ltsys_C6']**2)
                # # pi = 3.14159
                # # #fact = group['Gamma'].mean()/(2*pi)
                # # fact = 1/(2*pi)
                # # fact2 = fact*group['epsilon'].mean()
                # # fact3 = fact*np.sqrt(2*group['epsilon'].mean()*(1+group['epsilon'].mean()))
                # # y = fit_function(phi, A*fact, B*fact2, C*fact3)

                # # print(A,B,C)

                # # plt.plot(phi, y,'k-',label='CLAS6 Fit high',linewidth=5)



                # # A = row['tel_C6']-np.sqrt(row['telsys_C6']**2+row['telsys_C6']**2)
                # # B = row['tt_C6']-np.sqrt(row['ttsys_C6']**2+row['ttsys_C6']**2)
                # # C = row['lt_C6']-np.sqrt(row['ltsys_C6']**2+row['ltsys_C6']**2)
                # # pi = 3.14159
                # # #fact = group['Gamma'].mean()/(2*pi)
                # # fact = 1/(2*pi)
                # # fact2 = fact*group['epsilon'].mean()
                # # fact3 = fact*np.sqrt(2*group['epsilon'].mean()*(1+group['epsilon'].mean()))
                # # y = fit_function(phi, A*fact, B*fact2, C*fact3)

                # # print(A,B,C)

                # # plt.plot(phi, y,'k-',label='CLAS6 Fit high',linewidth=5)
            #plt.xlabel('Phi')
            #plt.ylabel('Fit Function')
            #plt.title('Phi vs Fit Function')
            #plt.show()

                #plt.legend()

                #plt.show()
            #restrict vertical axis to start at 0
            
            #ax.ylim(bottom=0)
            #plt.show()
            #sys.exit()

        # show the plot
        #plt.tight_layout()  # adjust the layout to prevent overlap

        #plt.show()

if show_xsec:


    # drop rows from combined_df where uncertainty/value > 1
    combined_df = combined_df[combined_df['total_uncert_unfolded']/combined_df['xsec_red_unfolded'] < .7] #cutoff chosen emperically

    # make histogram of total_uncert_unfolded
    # plt.hist(combined_df['total_uncert_unfolded']/combined_df['xsec_red_unfolded'], bins=100)
    # #set y axis to log
    # plt.yscale('log')
    # plt.show()
    # sys.exit()

    # grouping by 'xmin', 'qmin', 'tmin'
    groups = combined_df.groupby(['xmin', 'qmin', 'tmin'])



    ind = 0
    counter = 0
    for name, group in groups:
        #only plot if there are more than 10 bins
        print("in name")
        counter += 1
        print(counter)
        print(clas_df[['Q2_C6', 'xB_C6', 't_C6']].dtypes)
        print(group[['qmin', 'xmin', 'tmin', 'qmax', 'xmax', 'tmax']].dtypes)


        print(clas_df['Q2_C6'])
        print(group['qmin'].values[0])


        print(group['qmin'])

        mask = (clas_df['Q2_C6'] >= group['qmin'].values[0]) & (clas_df['Q2_C6'] <= group['qmax'].values[0]) & \
       (clas_df['xB_C6'] >= group['xmin'].values[0]) & (clas_df['xB_C6'] <= group['xmax'].values[0]) & \
       (clas_df['t_C6'] >= group['tmin'].values[0]) & (clas_df['t_C6'] <= group['tmax'].values[0])

        filtered_df = clas_df[mask]
        filtered_group = group[(group['pave'] > 100) & (group['pave'] < 260)]

        if len(group) < 1:
            print("skipping")
            continue

        plt.rcParams["font.size"] = "30"
        fig, ax = plt.subplots(figsize=(20,14))
        slabel = "Stat. Err. from Sim."
        elabel = "Stat. Err. from Exp."
        #plot with larger marker size
        #for showing different uncertainties
        #plt.errorbar(group['pave'], group['xsec_red'], yerr=group['xsec_red_err'],fmt='r+', markersize=50,label=#slabel)
        #plot again but with red error bars
        xerr_value = 5

        if len(group) < 5 or len(filtered_group) == 0:
            
        
            xerr_value = 0
            ax.errorbar(group['pave'], group['xsec_red_unfolded'], xerr=xerr_value,yerr=1.5*group['total_uncert_unfolded'],fmt='k.',  markersize=5,elinewidth=5,capsize=10, capthick=5)#elabel)
            xerr_value=9
            ax.errorbar(group['pave'], group['xsec_red_unfolded'], xerr=xerr_value,yerr=group['xsec_red_err_unfolded'],fmt='r.',  markersize=5,label="Unfolded",elinewidth=5)#,capsize=10, capthick=5)#elabel)
            xerr_value = 0
            ax.errorbar(group['pave'], group['xsec_red_unfolded'], xerr=xerr_value,yerr=0,fmt='r.',  markersize=5,elinewidth=5,capsize=10, capthick=5)#elabel)



            ax.plot(group['pave']-6, group['xsec_red'], marker='^', color='orange', markersize=21, label="Bin-by-Bin",linestyle="None")

        else:
            # fit the function to the data
            popt, pcov = curve_fit(fit_function, group['pave'], group['xsec_red_unfolded'], sigma=group['total_uncert_unfolded'], absolute_sigma=True)
            popt_top, pcov_top = curve_fit(fit_function, group['pave'], group['xsec_red_unfolded']+group['total_uncert_unfolded'], sigma=group['total_uncert_unfolded'], absolute_sigma=True)
            popt_bot, pcov_bot = curve_fit(fit_function, group['pave'], group['xsec_red_unfolded']-group['total_uncert_unfolded'], sigma=group['total_uncert_unfolded'], absolute_sigma=True)

            # calculate the standard deviation of the fitted parameters
            perr = np.sqrt(np.diag(pcov))

            # calculate the top and bottom bounds of the fit
            phis = np.linspace(0,360,1000)
            y = fit_function(phis, *popt)
            #y_top = fit_function(phis, *(popt + perr))
            y_top = fit_function(phis, *(popt_top))#+np.sqrt(np.diag(pcov_top))))
            #y_bottom = fit_function(phis, *(popt - perr))
            y_bottom = fit_function(phis, *(popt_bot))#-np.sqrt(np.diag(pcov_bot))))

            plt.plot(phis, y, 'b-', label="Trig. Fit",linewidth=5)
            # # # plt.fill_between(phis, y_bottom, y_top, color='b', alpha=0.2)  # this adds the band of uncertainty

            num_bands = 500  # number of bands in the gradient
            cmap = plt.get_cmap('winter')  # get the colormap

            for i in range(num_bands):
                # Create a range of color values
                color_val = i / num_bands
                # Calculate the y value for this band
                y_band_top = y_bottom + (y_top - y_bottom) * (i+1) / num_bands
                y_band_bottom = y_bottom + (y_top - y_bottom) * i / num_bands
                # Plot the band with the color from the colormap
                plt.fill_between(phis, y_band_bottom, y_band_top, color=cmap(color_val),alpha=0.1)
            xerr_value = 0
            plt.errorbar(group['pave'], group['xsec_red_unfolded'], xerr=xerr_value,yerr=1.5*group['total_uncert_unfolded'],fmt='k.',  markersize=5,elinewidth=5,capsize=10, capthick=5)#elabel)
            xerr_value=9
            plt.errorbar(group['pave'], group['xsec_red_unfolded'], xerr=xerr_value,yerr=group['xsec_red_err_unfolded'],fmt='r.',  markersize=5,label="Unfolded",elinewidth=5)#,capsize=10, capthick=5)#elabel)
            xerr_value = 0
            plt.errorbar(group['pave'], group['xsec_red_unfolded'], xerr=xerr_value,yerr=0,fmt='r.',  markersize=5,elinewidth=5,capsize=10, capthick=5)#elabel)
            #plt.errorbar(group['pave'], group['xsec_red_unfolded'], xerr=xerr_value,yerr=group['xsec_red_err_alt'],fmt='g.',  markersize=5,label="Unfolded",elinewidth=5)#elabel)



            plt.plot(group['pave']-6, group['xsec_red'], marker='^', color='orange', markersize=21, label="Bin-by-Bin",linestyle="None")

            #             #get errors from covariance matrix
            # perr = np.sqrt(np.diag(pcov))
            # print(perr)


            # errband_width

            # #y_bottom = fit_function(phis, *popt)-errband_width
            # #instead, calc y_bottom with errors from covariance matrix
            
            # y_top = fit_function(phis, *popt)+errband_width

            # plt.fill_between(phi, y_bottom, y_top, color='b', alpha=.1)#, label='CLAS6 Fit')



            #blue_line = mlines.Line2D([], [], color='r', marker='None', markersize=10, linestyle='-', label=slabel)
            #red_line = mlines.Line2D([], [], color='k', marker='None', markersize=10, linestyle='-', label=elabel)


        ax.set_xlabel('Lepton-Hadron Angle $\phi$')
        if plot_ylabel:
            ax.set_ylabel('Reduced Cross Section (nb/$GeV^2$)')
        #set xaxis range from 0 to 360
        ax.set_xlim([0,360])
        #set y bottom to 0
        ax.set_ylim(bottom=0)
        #pltt = 'Reduced Cross Section in bin ({})={}'.format(r'$x_{B,min} Q^2_{min} t_{min}$',str(name))
        pltt = '({})={}'.format(r'$x_{B,min} Q^2_{min} t_{min}$',str(name))
        #instead make plot title as averages xave qave tave
        #plot_title = '({})=({:.2f}, {:.2f}, {:.2f})'.format(r'$\langle x_{B}\rangle, \langle Q^2 \rangle, \lange t \rangle$',group['xave'].mean(),group['qave'].mean(),group['tave'].mean())
        plot_title = '{}={:.2f},{}={:.2f} GeV$^2$,{}={:.2f} GeV$^2$'.format(r'$\langle x_{B}\rangle$',group['xave'].mean(), r'$\langle Q^2 \rangle$',group['qave'].mean(), r'$\langle t \rangle$',group['tave'].mean())

        ax.set_title(plot_title)


        # Update the y-tick labels color based on their values
        yticks = ax.get_yticks()
        colors = []

        # for value in yticks:
        #     print(yticks)
        #     if value < 10:
        #         colors.append('black')
        #     elif 10 <= value < 100:
        #         colors.append('green')
        #     else:
        #         colors.append('red')
        # ax.set_yticklabels(yticks, colors=colors)
        yticks = ax.get_yticks()
        for i, value in enumerate(yticks):
            if value < 10:
                ax.get_yticklabels()[i].set_color('black')
            elif 10 <= value < 100:
                ax.get_yticklabels()[i].set_color('green')
            else:
                ax.get_yticklabels()[i].set_color('red')
        # Format y-axis tick labels to show rounded integers
        # Format y-axis tick labels based on the criteria
        def custom_formatter(x, _):
            if x < 10:
                return f'{x:.1f}'
            else:
                return f'{int(np.round(x))}'

        ax.yaxis.set_major_formatter(custom_formatter)

        #plt.show()
        
        plt.savefig(output_image_dir+pltt+".png")#,bbox_inches='tight')
        plt.close()
            # if counter > 3:
            #     sys.exit()
            # #plt.legend(handles=[blue_line, red_line])



            # if len(filtered_df) > 0:
            #     mask = (clas_dtp['q'] >= group['qmin'].values[0]) & (clas_dtp['q'] <= group['qmax'].values[0]) & \
            #     (clas_dtp['x'] >= group['xmin'].values[0]) & (clas_dtp['x'] <= group['xmax'].values[0]) & \
            #     (clas_dtp['t'] >= group['tmin'].values[0]) & (clas_dtp['t'] <= group['tmax'].values[0])

            #     filtered_df_dtp = clas_dtp[mask]
            #     print(filtered_df_dtp)

            #     #plt.errorbar(filtered_df_dtp['p'], filtered_df_dtp['dsdtdp'], yerr=np.sqrt(filtered_df_dtp['stat']**2+filtered_df_dtp['sys']**2),fmt='r+', markersize=50,label='CLAS6')

            #     errband_width = np.sqrt(filtered_df_dtp['stat']**2+filtered_df_dtp['sys']**2).mean()

                



            #     # plot the CLAS6 fit if it exists
            #     print("PLOTTING CLAS6 FIT")

            #     phi = np.linspace(0, 360, 1000)  # Replace 100 with the desired number of points

            #     # Assuming taking the first row of the filtered DataFrame
            #     print("FILTERED DF IS:")
            #     print(group.columns.values)
            #     print(filtered_df)
            #     print("xsec values are")
            #     print(group['pave'])
            #     row = filtered_df.iloc[0]

            #     A = row['tel_C6']
            #     B = row['tt_C6']
            #     C = row['lt_C6']
            #     pi = 3.14159
            #     #fact = group['Gamma'].mean()/(2*pi)
            #     fact = 1/(2*pi)
            #     fact2 = fact*group['epsilon'].mean()
            #     fact3 = fact*np.sqrt(2*group['epsilon'].mean()*(1+group['epsilon'].mean()))
            #     y = fit_function(phi, A*fact, B*fact2, C*fact3)

            #     print(A,B,C)

            #     #plt.plot(phi, y,'b-',label='CLAS6 Result',linewidth=5)
            #     #make line be 50% transparent
            #     #plt.plot(phi, y+errband_width,'k-',label='CLAS6 Fit',alpha=0.5)#
            #     #plt.plot(phi, y-errband_width,'k-',label='CLAS6 Fit',alpha=0.5)#
                
            #     y_bottom = y-errband_width
            #     y_top = y+errband_width

            #     plt.fill_between(phi, y_bottom, y_top, color='b', alpha=.1)#, label='CLAS6 Fit')

            #     #,linewidth = errband_width)

            #     phi = np.linspace(0, 360, 1000)  # Replace 100 with the desired number of points


                #pause to get input

                # Assuming taking the first row of the filtered DataFrame
                #print("FILTERED DF IS:")
                #print(group.columns.values)
                #print(filtered_df)
                #print("xsec values are")
                #print(group['pave'])
                # # row = filtered_df.iloc[0]

                # # A = row['tel_C6']+np.sqrt(row['telsys_C6']**2+row['telsys_C6']**2)
                # # B = row['tt_C6']+np.sqrt(row['ttsys_C6']**2+row['ttsys_C6']**2)
                # # C = row['lt_C6']+np.sqrt(row['ltsys_C6']**2+row['ltsys_C6']**2)
                # # pi = 3.14159
                # # #fact = group['Gamma'].mean()/(2*pi)
                # # fact = 1/(2*pi)
                # # fact2 = fact*group['epsilon'].mean()
                # # fact3 = fact*np.sqrt(2*group['epsilon'].mean()*(1+group['epsilon'].mean()))
                # # y = fit_function(phi, A*fact, B*fact2, C*fact3)

                # # print(A,B,C)

                # # plt.plot(phi, y,'k-',label='CLAS6 Fit high',linewidth=5)



                # # A = row['tel_C6']-np.sqrt(row['telsys_C6']**2+row['telsys_C6']**2)
                # # B = row['tt_C6']-np.sqrt(row['ttsys_C6']**2+row['ttsys_C6']**2)
                # # C = row['lt_C6']-np.sqrt(row['ltsys_C6']**2+row['ltsys_C6']**2)
                # # pi = 3.14159
                # # #fact = group['Gamma'].mean()/(2*pi)
                # # fact = 1/(2*pi)
                # # fact2 = fact*group['epsilon'].mean()
                # # fact3 = fact*np.sqrt(2*group['epsilon'].mean()*(1+group['epsilon'].mean()))
                # # y = fit_function(phi, A*fact, B*fact2, C*fact3)

                # # print(A,B,C)

                # # plt.plot(phi, y,'k-',label='CLAS6 Fit high',linewidth=5)
            #plt.xlabel('Phi')
            #plt.ylabel('Fit Function')
            #plt.title('Phi vs Fit Function')
            #plt.show()

            #plt.legend()

                #plt.show()
            #restrict vertical axis to start at 0
            # else:
            #     print("NO CLAS6 DATA, SKIPPING")
            #     plt.close()
                    #pass
            # # plt.ylim(bottom=0)
            # # #set xrange 0 to 360
            # # plt.xlim([0,360])
            # # #plt.show()
            # # counter +=1
            # # print("counter",counter)
            # # #if counter > 3:
            # # #    sys.exit()

            # # # plt.show()
            # # # ind += 1
            # # # if ind > 20:
            # # #     break
            # #     #plt.close()

if combine_plots:
    for t_val in combined_df['tmin'].unique():
        print("t_val",t_val)
        #only plot if there are more than 10 bins
        # use it for t_min = 1.0 and directory "images"
        t_output_image_dir = "t_xsec_unfolded_0807/"

        main(t_val, fs.xBbins, fs.Q2bins, output_image_dir,t_output_image_dir)
