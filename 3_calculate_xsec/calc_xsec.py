import pandas as pd
import numpy as np
import os, sys
from icecream import ic

from utils import filestruct, const, make_histos

#pd.set_option('mode.chained_assignment', None)


PhysicsConstants = const.PhysicsConstants()


fs = filestruct.fs()


# binned_outb_exp = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/outb/exp/final_f18_outb_exp_binned_with_area.pkl"
# binned_inb_exp = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/exp/final_f18_inb_exp_binned_with_area.pkl"

binned_exp = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/exp/singles/final_f18_inb_exp_binned_with_area.pkl"
binned_rec = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/rec/final_f18_inb_rec_binned.pkl"
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
combined_df['xsec'] = combined_df['acc_corr_counts'] / combined_df['true_total_vol'] /fs.f18_inbending_total_lumi_inv_fb *180/(3.14159)
combined_df['xsec_err'] = combined_df['acc_corr_counts_err'] / combined_df['true_total_vol'] /fs.f18_inbending_total_lumi_inv_fb*180/(3.14159)

#!!CONVERT PHI FROM DEG TO RAD

combined_df['y_bin'] = combined_df['qave']/(2*PhysicsConstants.electron_beam_energy*combined_df['xave']*PhysicsConstants.proton_mass)

combined_df['epsi_num'] = 1 - combined_df['y_bin'] - (combined_df['qave'])/(4*(PhysicsConstants.electron_beam_energy**2))

combined_df['epsi_denom'] = 1 - combined_df['y_bin'] + (combined_df['y_bin']**2)/2+(combined_df['qave'])/(4*(PhysicsConstants.electron_beam_energy**2))

combined_df['epsilon'] = combined_df['epsi_num'] / combined_df['epsi_denom']


combined_df['Gamma'] = (1/137/(8*3.14159)*combined_df['qave'] / 
                        (((PhysicsConstants.proton_mass)**2)*((PhysicsConstants.electron_beam_energy)**2))*
                        ((1-combined_df['xave'])/(combined_df['xave']**3)) *(1/(1-combined_df['epsilon']))
)



combined_df['xsec_red'] = combined_df['xsec'] *2*3.14159/combined_df['Gamma']
combined_df['xsec_red_err'] = combined_df['xsec_err'] *2*3.14159/combined_df['Gamma']

combined_df['acc_corr_counts_err_alt'] = combined_df['acc_corr_counts']*np.sqrt(1/combined_df['counts'])#+1/combined_df['rec_counts']+1/combined_df['gen_counts'])
combined_df['xsec_err_alt'] = combined_df['acc_corr_counts_err_alt'] / combined_df['true_total_vol'] /fs.f18_inbending_total_lumi_inv_fb*180/(3.14159)
combined_df['xsec_red_err_alt'] = combined_df['xsec_err_alt'] *2*3.14159/combined_df['Gamma']

combined_df = combined_df[combined_df['counts'] >= 1]


# combined_df = combined_df[combined_df['rec_counts'] >= 2]
# combined_df = combined_df[combined_df['gen_counts'] >= 2]
combined_df = combined_df[combined_df['acc_corr'] >= .005]

print(combined_df)

combined_df = combined_df.reset_index()

# combined_df.to_csv("combined_df_3.csv")
# sys.exit()

# for index, row in combined_df.iterrows():
#     print('tmin:', row['tmin'], 'pmin:', row['pmin'], 'xmin:', row['xmin'], 'qmin:', row['qmin'], 
#           'counts:', row['counts'], 'rec_counts:', row['rec_counts'], 'gen_counts:', row['gen_counts'])

# sys.exit()
show_plots = 0
show_xsec = 1


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

if show_xsec:
    import matplotlib.pyplot as plt

    # grouping by 'xmin', 'qmin', 'tmin'
    groups = combined_df.groupby(['xmin', 'qmin', 'tmin'])

    ind = 0
    for name, group in groups:
        #only plot if there are more than 10 bins
        if len(group) < 5:
            print("skipping")
            continue
        else:
            print("PLOTTING")
            plt.rcParams["font.size"] = "30"
            plt.figure(figsize=(20,14))
            
            slabel = "Stat. Err. from Sim."
            elabel = "Stat. Err. from Exp."
            #plot with larger marker size
            plt.errorbar(group['pave'], group['xsec_red'], yerr=group['xsec_red_err'],fmt='r+', markersize=30,label=slabel)
            #plot again but with red error bars
            plt.errorbar(group['pave'], group['xsec_red'], yerr=group['xsec_red_err_alt'],fmt='b+',  markersize=30,label=elabel)


            blue_line = mlines.Line2D([], [], color='r', marker='None', markersize=10, linestyle='-', label=slabel)
            red_line = mlines.Line2D([], [], color='b', marker='None', markersize=10, linestyle='-', label=elabel)


            plt.xlabel('Lepton-Hadron Angle $\phi$')
            plt.ylabel('Reduced Cross Section (nb/$GeV^2$)')
            pltt = 'Reduced Cross Section in bin ({})={}'.format(r'$x_{B,min} Q^2_{min} t_{min}$',str(name))
            plt.title(pltt)
            plt.legend(handles=[blue_line, red_line])

            #plt.show()
            #sys.exit()
            plt.savefig("plot_test/"+pltt+".png")
            # plt.show()
            # ind += 1
            # if ind > 20:
            #     break
            plt.close()