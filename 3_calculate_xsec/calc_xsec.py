import pandas as pd
import numpy as np
import os, sys
from icecream import ic

from utils import filestruct, const, make_histos


import os
from PIL import Image
import numpy as np
import re

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

    # create new image
    combined = Image.new("RGB", (img_width * len(xBbins), img_height * len(Q2bins)),"white")

    # place images
    for i, xB in enumerate(xBbins):
        for j, Q2 in enumerate(reversed(Q2bins)):
            if (xB, Q2) in images:
                combined.paste(images[(xB, Q2)], (i * img_width, j * img_height))

    return combined

def main(t, xBbins, Q2bins, in_dir_path=".",out_dir_path="."):
    image_dict = get_images_dict(t, in_dir_path)

    combined = create_image_grid(image_dict, xBbins, Q2bins, in_dir_path)

    combined.save(os.path.join(out_dir_path, f"combined_t{t}.png"))

#pd.set_option('mode.chained_assignment', None)


PhysicsConstants = const.PhysicsConstants()


fs = filestruct.fs()


# binned_outb_exp = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/outb/exp/final_f18_outb_exp_binned_with_area.pkl"
# binned_inb_exp = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/exp/final_f18_inb_exp_binned_with_area.pkl"

binned_exp = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/exp/with_area_binned_final_inbending_exclusive_t2.pkl"
binned_rec = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/rec/final_f18_inb_rec_binned_t2.pkl"
binned_gen = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/gen/final_f18_inb_gen_binned.pkl"
binned_gen_alt = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/gen_wq2_cut/final_f18_inb_gen_binned.pkl"


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

# Merge the dataframes

# Merge df1 and df2
combined_df = pd.merge(df_exp, df_rec, left_index=True, right_index=True)

# Then merge the result with df3
combined_df = pd.merge(combined_df, df_gen, left_index=True, right_index=True)

print(combined_df.columns.values)
print(combined_df)

combined_df['acc_corr'] = combined_df['rec_counts'] / combined_df['gen_counts']

combined_df['counts_err'] = np.sqrt(combined_df['counts'])
combined_df['rec_counts_err'] = np.sqrt(combined_df['rec_counts'])
combined_df['gen_counts_err'] = np.sqrt(combined_df['gen_counts'])

combined_df['acc_corr_counts'] = combined_df['counts'] / combined_df['acc_corr']

combined_df['acc_corr_counts_err'] = combined_df['acc_corr_counts']*np.sqrt(1/combined_df['counts']+1/combined_df['rec_counts']+1/combined_df['gen_counts'])



combined_df['acc_corr_counts_err'] = combined_df['acc_corr_counts_err'].fillna(0)
combined_df['acc_corr_counts_err'] = combined_df['acc_corr_counts_err'].replace(np.inf, 0)
combined_df['acc_corr_counts_err'] = combined_df['acc_corr_counts_err'].replace(np.nan, 0)

print(combined_df.columns.values)
# combined_df['xsec'] = combined_df['acc_corr_counts']  /fs.f18_inbending_total_lumi_inv_fb *180/(3.14159) / combined_df['true_total_vol']
# combined_df['xsec_err'] = combined_df['acc_corr_counts_err'] /fs.f18_inbending_total_lumi_inv_fb*180/(3.14159)/ combined_df['true_total_vol']

combined_df['xsec'] = combined_df['acc_corr_counts']  /fs.f18_inbending_total_lumi_inv_fb *180/(3.14159) / combined_df['nominal_xbq2_bin_volume'] / combined_df['tp_bin_volume']
combined_df['xsec_err'] = combined_df['acc_corr_counts_err'] /fs.f18_inbending_total_lumi_inv_fb*180/(3.14159)/ combined_df['nominal_xbq2_bin_volume'] / combined_df['tp_bin_volume']


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


combined_df['xsec_red'] = combined_df['xsec'] *pi2_fact/combined_df['Gamma']
combined_df['xsec_red_err'] = combined_df['xsec_err'] *pi2_fact/combined_df['Gamma']

combined_df['acc_corr_counts_err_alt'] = combined_df['acc_corr_counts']*np.sqrt(1/combined_df['counts'])#+1/combined_df['rec_counts']+1/combined_df['gen_counts'])

#combined_df['xsec_err_alt'] = combined_df['acc_corr_counts_err_alt']  /fs.f18_inbending_total_lumi_inv_fb*180/(3.14159) / combined_df['true_total_vol']
combined_df['xsec_err_alt'] = combined_df['acc_corr_counts_err_alt']  /fs.f18_inbending_total_lumi_inv_fb*180/(3.14159) / combined_df['nominal_xbq2_bin_volume'] / combined_df['tp_bin_volume']


combined_df['xsec_red_err_alt'] = combined_df['xsec_err_alt'] *pi2_fact/combined_df['Gamma']

combined_df = combined_df[combined_df['counts'] >= 1]


# combined_df = combined_df[combined_df['rec_counts'] >= 2]
# combined_df = combined_df[combined_df['gen_counts'] >= 2]
combined_df = combined_df[combined_df['acc_corr'] >= .005]

print(combined_df)

combined_df = combined_df.reset_index()

#save as pickle file
combined_df.to_pickle("full_cross_section_clas12.pkl")

# combined_df.to_csv("combined_df_3.csv")
# sys.exit()

# for index, row in combined_df.iterrows():
#     print('tmin:', row['tmin'], 'pmin:', row['pmin'], 'xmin:', row['xmin'], 'qmin:', row['qmin'],
#           'counts:', row['counts'], 'rec_counts:', row['rec_counts'], 'gen_counts:', row['gen_counts'])

# sys.exit()
show_plots = 0
show_xsec = 1
combine_plots = 1
output_image_dir = "plot_test_t2_nom_vol/"


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

clas_dtp = pd.read_csv("3_calculate_xsec/xs_clas6.csv")
#set the columns
#the columns are the first row
#clas_dtp.columns = clas_dtp.iloc[0]

#print(clas_dtp)
#sys.exit()




if show_xsec:
    import matplotlib.pyplot as plt

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
        else:
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
            plt.errorbar(group['pave'], group['xsec_red'], xerr=xerr_value,yerr=group['xsec_red_err_alt'],fmt='k.',  markersize=5,label="CLAS12 Data")#elabel)

            # fit the function to the data
            popt, pcov = curve_fit(fit_function, group['pave'], group['xsec_red'], sigma=group['xsec_red_err'], absolute_sigma=True)

            # print out the fit parameters
            print(f"A = {popt[0]}, B = {popt[1]}, C = {popt[2]}")

            # plot the fit
            #phis = np.linspace(group['pave'].min(), group['pave'].max(), 1000)
            phis = np.linspace(0,360,1000)
            plt.plot(phis, fit_function(phis, *popt), 'k-', label="CLAS12 Fit",linewidth=5)
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

                plt.plot(phi, y,'r-',label='CLAS6 Result',linewidth=5)
                #make line be 50% transparent
                #plt.plot(phi, y+errband_width,'k-',label='CLAS6 Fit',alpha=0.5)#
                #plt.plot(phi, y-errband_width,'k-',label='CLAS6 Fit',alpha=0.5)#
                
                y_bottom = y-errband_width
                y_top = y+errband_width

                plt.fill_between(phi, y_bottom, y_top, color='r', alpha=.1)#, label='CLAS6 Fit')

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
            plt.ylim(bottom=0)
            plt.savefig(output_image_dir+pltt+".png")
            # plt.show()
            # ind += 1
            # if ind > 20:
            #     break
                #plt.close()

if combine_plots:
    for t_val in combined_df['tmin'].unique():
        print("t_val",t_val)
        #only plot if there are more than 10 bins
        # use it for t_min = 1.0 and directory "images"
        t_output_image_dir = "t_xsec/"

        main(t_val, fs.xBbins, fs.Q2bins, output_image_dir,t_output_image_dir)
