


# 1.) Necessary imports.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import argparse
import sys, os
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib as mpl
from icecream import ic
import scipy.integrate as integrate
from scipy.optimize import root_scalar



def bottom_line(x):
    return (4-0.938**2)/(1/x-1) 
    
    #y = Q^2/(2ME xB) < (10.604-2) / (10.604)
def top_line(x):
    return (10.604-2) / (10.604)*2*0.938*x*10.604

def right_line(y):
    return 1/((4-0.938**2)/y+1)

def left_line(y):
    return y/((10.604-2) / (10.604)*2*0.938*10.604)

    

def plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
            saveplot=True,pics_dir="none",plot_title="none",logger=False,first_label="rad",figsize=(18,10),
            filename="ExamplePlot",units=["",""],extra_data=None,xaxis_lines=None,yaxis_lines=None,xbq2plot=False):
    
    #plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "30"
    # Initalize parameters
    x_name = var_names[0]
    y_name = var_names[1]
    xmin = ranges[0][0]
    xmax =  ranges[0][1]
    num_xbins = ranges[0][2]
    ymin = ranges[1][0]
    ymax =  ranges[1][1]
    num_ybins = ranges[1][2]
    x_bins = np.linspace(xmin, xmax, num_xbins) 
    y_bins = np.linspace(ymin, ymax, num_ybins) 

    # Creating plot
    fig, ax = plt.subplots(figsize =figsize) 
    if units[0] == "None":
        ax.set_xlabel("{}".format(x_name))
    else:
        ax.set_xlabel("{} ({})".format(x_name,units[0]))
    if units[1] == "None":
        ax.set_ylabel("{}".format(y_name))
    else:
        ax.set_ylabel("{} ({})".format(y_name,units[1]))

    plt.hist2d(x_data, y_data, bins =[x_bins, y_bins],
        range=[[xmin,xmax],[ymin,ymax]],norm=mpl.colors.LogNorm())# cmap = plt.cm.nipy_spectral) 

    # Adding color bar 
    if colorbar:
        plt.colorbar()

    #plt.tight_layout()  

    
    #Generate plot title
    if plot_title == "none":
        plot_title = '{} vs {}'.format(x_name,y_name)
    
    plt.title(plot_title) 

    plotlines = 1
    if plotlines: 
        for xbin in xaxis_lines:   
            if xbq2plot:
                plt.vlines(xbin,ymin=ymin,ymax=ymax, color='lightgray')
                plt.vlines(xbin,ymin=np.max([bottom_line(xbin),1]),ymax=top_line(xbin), color='r')
            else:
                plt.vlines(xbin,ymin=ymin,ymax=ymax, color='r')

        for ybin in yaxis_lines:
            print(ybin)
            if xbq2plot:
                plt.hlines(ybin,xmin=xmin, xmax=xmax, color='lightgray')
                plt.hlines(ybin,xmin=left_line(ybin), xmax=right_line(ybin), color='r')
            else:
                plt.hlines(ybin,xmin=xmin, xmax=xmax, color='r')


        if xbq2plot:
            plt.hlines(1,xmin=left_line(1), xmax=right_line(1), color='orange', linewidth=5) #Q2>1 GeV^2 cut
            
            #make x range for plotting
            x_range = np.linspace(left_line(1),xmax,1000)
            #populate yvals with the y values for the fit
            yvals = top_line(x_range)
            #plot the fit with thick line
            plt.plot(x_range,yvals,'b',linewidth=5)
            #make x range for plotting
            #populate yvals with the y values for the fit
            x_range2 = np.linspace(right_line(1),xmax,1000)

            yvals_bottom = bottom_line(x_range2)
            #plot the fit
            plt.plot(x_range2,yvals_bottom,'k',linewidth=5)

    #plt.yscale('log')

    #plt.savefig("1.png")

    if saveplot:
        #plot_title.replace("/","")
        new_plot_title = plot_title.replace("/","").replace(" ","_").replace("$","").replace("^","").replace("\\","").replace(".","").replace("<","").replace(">","")
        print(new_plot_title)
        if not os.path.exists(pics_dir):
            os.makedirs(pics_dir)
        plt.savefig(pics_dir + new_plot_title+".png")
        plt.close()
        print("Figure {} saved to {}".format(new_plot_title,pics_dir))

    else:
       plt.show()


    if extra_data is not None:
        plt.hist2d(extra_data[0], extra_data[1], bins =[x_bins, y_bins],
            range=[[xmin,xmax],[ymin,ymax]], cmap = plt.cm.nipy_spectral) #norm=mpl.colors.LogNorm())#
        plt.show()


from utils import filestruct

fs = filestruct.fs()



inbending_data = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/inb/exp/final_inbending_exclusive.pkl"
outbending_data = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/outb/exp/final_outbending_exclusive.pkl"
rec_in = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/inb/rec/"#dvpip_events_norad_10000_20230705_1043_Fall_2018_Inbending_45nA_recon.pkl"
rec_in_rad = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/inb/rec_rad/"
rec_in_reproc = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/inb/rec_reproc/dvpip_events_norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon.pkl"
rec_out = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/outb/rec/norad_10000_20230703_1814_Fall_2018_Outbending_100_50nA_recon_fid_corr_smear.pkl"
inb = "/mnt/d/GLOBUS/CLAS12/Thesis/1_potential_dvpip_events/inb/exp/20220511_f18_in_combined_157_cor.pkl"
gen = "/mnt/d/GLOBUS/CLAS12/Thesis/1_potential_dvpip_events/gen/lund_10000_20230624_1457/_mnt_d_GLOBUS_CLAS12_Simulation_staging_lund_10000_20230624_1457_lund_files__1.pkl"
#Display nevents in each bin also

rec_in_epgg = "/mnt/d/GLOBUS/CLAS12/Thesis/1_potential_dvpip_events/inb/rec/norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon.pkl"

df = pd.read_pickle(rec_in_reproc)

configs = ["Inb."]
files = [rec_in_reproc]
output_dirs =["/mnt/d/GLOBUS/CLAS12/Thesis/plots/2_dvpip_distros/inb/rec/bin_migrations/"]

xb_bin_edges, q2_bin_edges,t2_bin_edges, phi1_bin_edges = fs.xBbins, fs.Q2bins, fs.tbins, fs.phibins

phi_bin_edges = phi1_bin_edges
t_bin_edges = t2_bin_edges

# # # for i in range(len(configs)):
# # #     config = configs[i]
# # #     df = pd.read_pickle(files[i])
# # #     out_dir = output_dirs[i]

# # # # # 
# # #     # plot_2dhist(df["GenxB"],df["GenQ2"],["$x_B$","$Q^2$"],[[0.05,0.8,100],[0.5,11,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
# # #     #             plot_title="$x_B$ vs $Q^2$, Exp. {}".format(config),units=["None","GeV$^2$"],xbq2plot=True,saveplot=False,pics_dir=out_dir)

# # #     # plot_2dhist(df["Genphi1"],df["Gent1"],["$\phi$","t"],[[0,360,50],[0,1.8,50]],colorbar=True,xaxis_lines=phi_bin_edges,yaxis_lines=t_bin_edges,
# # #     #             plot_title="$\phi$ vs Momentum Transfer t, Exp. {}".format(config),units=["None","GeV$^2$"],saveplot=False,pics_dir=out_dir)

# # #     # df = df[(df["GenxB"] > 0.2) & (df["GenxB"] < 0.25) & (df["GenQ2"] > 3.5) & (df["GenQ2"] <4)]
# # #     # df = df[(df["Gent1"] > 0.2) & (df["Gent1"] < 0.3) & (df["Genphi1"] > 36) & (df["Genphi1"] <72)]
# # #     # print(df)

# # #     for i in range(len(xb_bin_edges)-1):
# # #         for j in range(len(q2_bin_edges)-1):
# # #             # Filter the dataframe for the current xb and Q2 bin
# # #             bin_df = df[(df[prefix+"xB"] > xb_bin_edges[i]) & 
# # #                         (df[prefix+"xB"] < xb_bin_edges[i+1]) & 
# # #                         (df[prefix+"Q2"] > q2_bin_edges[j]) & 
# # #                         (df[prefix+"Q2"] < q2_bin_edges[j+1])]
            
# # #             if len(bin_df) > 1:

# # #                 print("Bin {}-{} x {}-{} Q2".format(xb_bin_edges[i],xb_bin_edges[i+1],q2_bin_edges[j],q2_bin_edges[j+1]))
# # #                 # Plot a 2D histogram of 'p' vs 't' for the filtered dataframe
                
# # #                 # Set the title based on the bin ranges and the number of events in the bin
# # #                 plot_title = "$\phi$ vs Momentum Transfer t, Exp. {}, {}<$x_B$<{},{}<$Q^2$<{}, N={}".format(config,xb_bin_edges[i],xb_bin_edges[i+1],q2_bin_edges[j],q2_bin_edges[j+1], len(bin_df))

# # #                 print(bin_df[prefix+"phi1"])
# # #                 print(bin_df[prefix+"t2"])
# # #                 plot_2dhist(bin_df[prefix+"phi1"],bin_df[prefix+"t2"],["$\phi$","t"],[[0,360,50],[0,1.8,50]],colorbar=True,xaxis_lines=phi_bin_edges,yaxis_lines=t_bin_edges,
# # #                         plot_title=plot_title,units=["None","GeV$^2$"],saveplot=True,pics_dir=out_dir,figsize=(30,18))

directory = rec_in



for i in range(len(configs)):
    config = configs[i]
    out_dir = output_dirs[i]
    # if out_dir doesn't exist, make it:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    

# # 
    # plot_2dhist(df["GenxB"],df["GenQ2"],["$x_B$","$Q^2$"],[[0.05,0.8,100],[0.5,11,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
    #             plot_title="$x_B$ vs $Q^2$, Exp. {}".format(config),units=["None","GeV$^2$"],xbq2plot=True,saveplot=False,pics_dir=out_dir)

    # plot_2dhist(df["Genphi1"],df["Gent1"],["$\phi$","t"],[[0,360,50],[0,1.8,50]],colorbar=True,xaxis_lines=phi_bin_edges,yaxis_lines=t_bin_edges,
    #             plot_title="$\phi$ vs Momentum Transfer t, Exp. {}".format(config),units=["None","GeV$^2$"],saveplot=False,pics_dir=out_dir)

    # df = df[(df["GenxB"] > 0.2) & (df["GenxB"] < 0.25) & (df["GenQ2"] > 3.5) & (df["GenQ2"] <4)]
    # df = df[(df["Gent1"] > 0.2) & (df["Gent1"] < 0.3) & (df["Genphi1"] > 36) & (df["Genphi1"] <72)]
    # print(df)








    plot_all_purities = 0

    for t_option in ["t1","t2"]:
        mag_option,rad_option = "inb","norad"

        if plot_all_purities:
            df = pd.read_pickle("bin_purities_{}_{}_{}.pkl".format(mag_option,rad_option,t_option))
            print(df)
            print(min(df["purity"]))


            # # Convert the string representation of the tuple into actual tuple
            # df['xB_bin'] = df['xB_bin'].apply(lambda x: eval(x))
            # df['Q2_bin'] = df['Q2_bin'].apply(lambda x: eval(x))
            # df['t2_bin'] = df['t2_bin'].apply(lambda x: eval(x))
            # df['phi1_bin'] = df['phi1_bin'].apply(lambda x: eval(x))

            # Create new min and max columns for each bin
            df['xmin'], df['xmax'] = zip(*df['xB_bin'])
            df['qmin'], df['qmax'] = zip(*df['Q2_bin'])
            df['tmin'], df['tmax'] = zip(*df['{}_bin'.format("t2")])
            df['pmin'], df['pmax'] = zip(*df['phi1_bin'])

            # Drop the original bin columns
            df = df.drop(columns=['xB_bin', 'Q2_bin', '{}_bin'.format("t2"), 'phi1_bin'])

            print(df.head())  # display the first few rows of the modified dataframe

            plt.rcParams["font.size"] = "18"

            for xbin_val in fs.xBbins:
                for qbin_val in fs.Q2bins:
                    # xbin_val = 0.2
                    # qbin_val = 3
                    dff =  df[(df['xmin'] == xbin_val) & (df['qmin'] == qbin_val)]

                    # test if any value of purity is not -1
                    if len(dff[dff['purity'] != -1]) == 0:
                        print("this bitch empty, YEET")
                    else:
                        
                        print("DFF")
                        print(dff)

                        # Create 2D histogram bins
                        p_bins_x = np.array(fs.xBbins)
                        t_bins_x = np.array(fs.Q2bins)

                        p_bins_p = np.array(fs.phibins)
                        t_bins_p = np.array(fs.tbins)

                        # Calculate bin centres
                        p_bin_centres_x = (p_bins_x[:-1] + p_bins_x[1:]) / 2
                        t_bin_centres_x = (t_bins_x[:-1] + t_bins_x[1:]) / 2

                        p_bin_centres_p = (p_bins_p[:-1] + p_bins_p[1:]) / 2
                        t_bin_centres_p = (t_bins_p[:-1] + t_bins_p[1:]) / 2

                        # Create a grid of bin centres
                        p_grid_x, t_grid_x = np.meshgrid(p_bins_x, t_bins_x)
                        p_grid_p, t_grid_p = np.meshgrid(p_bins_p, t_bins_p)

                        # Create a 2D array to hold purity values
                        purity_grid_x = np.zeros(p_grid_x.shape)
                        purity_grid_p = np.zeros(p_grid_p.shape)




                        for _, row in dff.iterrows():
                            p_idx = np.digitize((row['pmin'] + row['pmax']) / 2, p_bins_p) - 1
                            t_idx = np.digitize((row['tmin'] + row['tmax']) / 2, t_bins_p) - 1
                            purity_grid_p[t_idx, p_idx] = row['purity']

                        # remove bins with purity = -1
                        purity_grid_p[purity_grid_p == -1] = np.nan
                        
                        p_ave = purity_grid_p
                        #remove zeros
                        p_ave[p_ave == 0] = np.nan
                        #get average of all purity values in purity_grid_p
                        avg_purity = np.nanmean(p_ave)

                                            # Populate the purity_grid with purity values from the DataFrame
                        for _, row in dff.iterrows():
                            p_idx = np.digitize((row['xmin'] + row['xmax']) / 2, p_bins_x) - 1
                            t_idx = np.digitize((row['qmin'] + row['qmax']) / 2, t_bins_x) - 1
                            #print(row)
                            purity_grid_x[t_idx, p_idx] = avg_purity#row['purity']

                        # print(purity_grid_x)
                        # sys.exit()

                        # remove bins with purity = -1
                        purity_grid_x[purity_grid_x == -1] = np.nan

                        # Create figure and axes
                        import matplotlib.gridspec as gridspec

                        fig = plt.figure(figsize=(24, 10))

                        gs = gridspec.GridSpec(1, 3)  # we create a grid with 1 row and 3 columns

                        ax1 = plt.subplot(gs[0, 0])  # ax1 occupies the first column of the grid
                        ax2 = plt.subplot(gs[0, 1:])  # ax2 occupies the second and third column of the grid

                        axs = [ax1, ax2]
                                            # Plot pcolormesh for first grid
                        pcmesh_x = axs[0].pcolormesh(p_grid_x, t_grid_x, purity_grid_x, cmap='viridis', vmin=0.3, vmax=.9, norm=mpl.colors.LogNorm())
                        #fig.colorbar(pcmesh_x, ax=axs[0], label='Purity')
                        axs[0].set_xlabel('$x_B$')
                        axs[0].set_ylabel('$Q^2$ (GeV$^2$))')
                        #set title as bin values
                        axs[0].set_title("{}<$x_B$<{}, {}<$Q^2$<{}".format(xbin_val,fs.xBbins[fs.xBbins.index(xbin_val)+1],qbin_val,fs.Q2bins[fs.Q2bins.index(qbin_val)+1]))

                        # Plot lines and set limits for first grid
                        #... # Code for plotting lines on the first grid

                        xaxis_lines = fs.xBbins
                        yaxis_lines = fs.Q2bins
                        xbq2plot = True
                        plotlines = 1
                        ymin_x = 1
                        ymax_x = 11
                        xmin_x = 0.05
                        xmax_x = 0.8
                        if plotlines: 
                            for xbin in xaxis_lines:   
                                if xbq2plot:
                                    print("plotting")
                                    axs[0].vlines(xbin,ymin=ymin_x,ymax=ymax_x, color='lightgray')
                                    axs[0].vlines(xbin,ymin=np.max([bottom_line(xbin),1]),ymax=top_line(xbin), color='r')
                                else:
                                    axs[0].vlines(xbin,ymin=ymin_x,ymax=ymax_x, color='r')

                            for ybin in yaxis_lines:
                                print(ybin)
                                if xbq2plot:
                                    axs[0].hlines(ybin,xmin=xmin_x, xmax=xmax_x, color='lightgray')
                                    axs[0].hlines(ybin,xmin=left_line(ybin), xmax=right_line(ybin), color='r')
                                else:
                                    axs[0].hlines(ybin,xmin=xmin_x, xmax=xmax_x, color='r')


                            if xbq2plot:
                                axs[0].hlines(1,xmin=left_line(1), xmax=right_line(1), color='orange', linewidth=5) #Q2>1 GeV^2 cut
                                
                                #make x range for plotting
                                x_range = np.linspace(left_line(1),xmax_x,1000)
                                #populate yvals with the y values for the fit
                                yvals = top_line(x_range)
                                #plot the fit with thick line
                                axs[0].plot(x_range,yvals,'b',linewidth=5)
                                #make x range for plotting
                                #populate yvals with the y values for the fit
                                x_range2 = np.linspace(right_line(1),xmax_x,1000)

                                yvals_bottom = bottom_line(x_range2)
                                #plot the fit
                                axs[0].plot(x_range2,yvals_bottom,'k',linewidth=5)


                        axs[0].set_xlim(xmin_x,xmax_x)
                        axs[0].set_ylim(ymin_x,ymax_x)

                        # Plot pcolormesh for second grid
                        pcmesh_p = axs[1].pcolormesh(p_grid_p, t_grid_p, purity_grid_p, cmap='viridis', vmin=0.3, vmax=.9)
                        fig.colorbar(pcmesh_p, ax=axs[1], label='Purity')
                        axs[1].set_xlabel('$\phi$')
                        axs[1].set_ylabel('Momentum Transfer t (GeV$^2$))')
                        #set title as bin values
                        axs[1].set_title("Bin Purity, {} {} $\phi$ and {}, {}<$x_B$<{}, {}<$Q^2$<{}".format(mag_option,rad_option,t_option,xbin_val,fs.xBbins[fs.xBbins.index(xbin_val)+1],qbin_val,fs.Q2bins[fs.Q2bins.index(qbin_val)+1]))

                        # Plot lines and set limits for second grid
                        #... # Code for plotting lines on the second grid


                        xaxis_lines = fs.phibins
                        yaxis_lines = fs.tbins
                        xbq2plot = True
                        plotlines = 1
                        ymin_p = 0
                        ymax_p = 2
                        xmin_p = 0.09
                        xmax_p = 360
                        xbq2plot = False
                        if plotlines: 
                            for xbin in xaxis_lines:   
                                if xbq2plot:
                                    plt.vlines(xbin,ymin=ymin_p,ymax=ymax_p, color='lightgray')
                                    plt.vlines(xbin,ymin=np.max([bottom_line(xbin),1]),ymax=top_line(xbin), color='r')
                                else:
                                    plt.vlines(xbin,ymin=ymin_p,ymax=ymax_p, color='r')

                            for ybin in yaxis_lines:
                                print(ybin)
                                if xbq2plot:
                                    plt.hlines(ybin,xmin=xmin_p, xmax=xmax_p, color='lightgray')
                                    plt.hlines(ybin,xmin=left_line(ybin), xmax=right_line(ybin), color='r')
                                else:
                                    plt.hlines(ybin,xmin=xmin_p, xmax=xmax_p, color='r')



                        axs[1].set_xlim(xmin_p,xmax_p)
                        axs[1].set_ylim(ymin_p,ymax_p)

                        # generate name for plot file
                        output_name = "bin_purity_{}_{}_{}_{}".format(xbin_val,fs.xBbins[fs.xBbins.index(xbin_val)+1],qbin_val,fs.Q2bins[fs.Q2bins.index(qbin_val)+1])

                        plt.savefig("bin_purity_studies_{}_{}_{}/{}.png".format(mag_option,rad_option,t_option,output_name))
                        print("Figure {} saved to {}".format(output_name,"bin_purity_studies_{}_{}_{}/".format(mag_option,rad_option,t_option)))







    calc_all_purities = 0
    directory = rec_in_rad
    if calc_all_purities:
        # List all pickle files in the directory
        t_option = "t1"
        mag_option = "inb"
        rad_option = "rad"
        files = [f for f in os.listdir(directory) if f.endswith('.pkl')]

        # Initialize an empty list to store individual dataframes
        dfs = []
        
        # Loop through each file
        for file in files:
            # Create a full file path
            filepath = os.path.join(directory, file)

            # Load the DataFrame from the pickle file and append it to the list
            df = pd.read_pickle(filepath)
            dfs.append(df)

        # Concatenate all dataframes in the list into one dataframe
        combined_df = pd.concat(dfs, ignore_index=True)


        df = combined_df


        # Assume t2_bin_edges and phi1_bin_edges are defined.
        # for example:
        # t2_bin_edges = np.arange(0, 2, 0.1)
        # phi1_bin_edges = np.arange(0, 360, 10)

        # Initialize the results
        results = []

        for i in range(len(xb_bin_edges)-1):
            for j in range(len(q2_bin_edges)-1):
                for k in range(len(t2_bin_edges)-1):
                    print(k)
                    for l in range(len(phi1_bin_edges)-1):
                        prefix = ""
                        # How many events were found in this bin?
                        bin_rec_df = df[(df[prefix+"xB"] > xb_bin_edges[i]) & 
                                    (df[prefix+"xB"] < xb_bin_edges[i+1]) & 
                                    (df[prefix+"Q2"] > q2_bin_edges[j]) & 
                                    (df[prefix+"Q2"] < q2_bin_edges[j+1]) &
                                    (df[prefix+"{}".format(t_option)] > t2_bin_edges[k]) & 
                                    (df[prefix+"{}".format(t_option)] < t2_bin_edges[k+1]) & 
                                    (df[prefix+"phi1"] > phi1_bin_edges[l]) & 
                                    (df[prefix+"phi1"] < phi1_bin_edges[l+1])
                                    ]
                        if len(bin_rec_df) > -1:
                            prefix = "Gen"
                            # How many events originated in this bin?
                            bin_gen_df = df[(df[prefix+"xB"] > xb_bin_edges[i]) & 
                                        (df[prefix+"xB"] < xb_bin_edges[i+1]) & 
                                        (df[prefix+"Q2"] > q2_bin_edges[j]) & 
                                        (df[prefix+"Q2"] < q2_bin_edges[j+1]) &
                                        (df[prefix+"{}".format(t_option)] > t2_bin_edges[k]) & 
                                        (df[prefix+"{}".format(t_option)] < t2_bin_edges[k+1]) & 
                                        (df[prefix+"phi1"] > phi1_bin_edges[l]) & 
                                        (df[prefix+"phi1"] < phi1_bin_edges[l+1])
                                        ]

                            prefix = "Gen"
                            # Filter the dataframe for the current xb and Q2 bin
                            #Of the events that were found in this bin, how many originated there?
                            bin_rec_gen_df = bin_rec_df[(bin_rec_df[prefix+"xB"] > xb_bin_edges[i]) & 
                                        (bin_rec_df[prefix+"xB"] < xb_bin_edges[i+1]) & 
                                        (bin_rec_df[prefix+"Q2"] > q2_bin_edges[j]) & 
                                        (bin_rec_df[prefix+"Q2"] < q2_bin_edges[j+1]) &
                                        (bin_rec_df[prefix+"{}".format(t_option)] > t2_bin_edges[k]) & 
                                        (bin_rec_df[prefix+"{}".format(t_option)] < t2_bin_edges[k+1]) & 
                                        (bin_rec_df[prefix+"phi1"] > phi1_bin_edges[l]) & 
                                        (bin_rec_df[prefix+"phi1"] < phi1_bin_edges[l+1])
                                        ]


                            N_pure = len(bin_rec_gen_df)
                            N_found = len(bin_rec_df)
                            N_generated = len(bin_gen_df)
                            if N_found > 0:
                                bin_purity = N_pure/N_found
                            else:
                                bin_purity = -1
                            if N_generated > 0:
                                bin_efficiency = N_pure/N_generated
                            else:
                                bin_efficiency = -1

                            # Append the results
                            results.append({
                                "xB_bin": (xb_bin_edges[i], xb_bin_edges[i+1]),
                                "Q2_bin": (q2_bin_edges[j], q2_bin_edges[j+1]),
                                "{}_bin".format(t_option): (t2_bin_edges[k], t2_bin_edges[k+1]),
                                "phi1_bin": (phi1_bin_edges[l], phi1_bin_edges[l+1]),
                                "N_pure": N_pure,
                                "N_found": N_found,
                                "N_generated": N_generated,
                                "purity": bin_purity,
                                "efficiency": bin_efficiency
                            })

        # Convert the results to a DataFrame for easier analysis
        results_df = pd.DataFrame(results)
        #save to pickle
        results_df.to_pickle("bin_purities_{}_{}_{}.pkl".format(mag_option,rad_option,t_option))

                    #Of the events that originated in this bin, how many were found in another bin?

                    # print("Bin {}-{} x {}-{} Q2".format(xb_bin_edges[i],xb_bin_edges[i+1],q2_bin_edges[j],q2_bin_edges[j+1]))
                    # # Plot a 2D histogram of 'p' vs 't' for the filtered dataframe
                    
                    # # Set the title based on the bin ranges and the number of events in the bin

                    # print(bin_rec_df[prefix+"phi1"])
                    # print(bin_rec_df[prefix+"t2"])

                    # #bin purity: of the number found in this bin, how many originated there?
                    # #bin efficiency: of the number that originated in this bin, how many were found?
                    # #bin migration: of the number that originated in this bin, how many were found in another bin?
                    # #bin contamination: of the number found in this bin, how many originated in another bin?
                    # #bin background: of the number found in this bin, how many originated in another bin?


                    # title_prefixes = ["Gen","Rec"]
                    # bin_purity = int(bin_purity*100)

                    # for ind,prefix in enumerate(["Gen",""]):
                    #     title_prefix = title_prefixes[ind]
                    #     #turn bin purity into a percentage with no decimal places
                        
                    #     plot_title = "$x_B$ vs $Q^2$, Sim. {}, {}<$x_B$<{},{}<$Q^2$<{}, N={}, $P${}%".format(title_prefix,xb_bin_edges[i],xb_bin_edges[i+1],q2_bin_edges[j],q2_bin_edges[j+1], N_found,bin_purity)

                    #     print(prefix)
                    #     print(title_prefix)
                    #     plot_2dhist(bin_rec_df[prefix+"xB"],bin_rec_df[prefix+"Q2"],["$x_B$","$Q^2$"],[[0.05,0.9,200],[0.5,12,200]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
                    #                 plot_title=plot_title,units=["None","GeV$^2$"],xbq2plot=True,saveplot=True,pics_dir=out_dir)

                    #     plot_title = "$\phi$ vs Momentum Transfer t, Sim. {}, {}<$x_B$<{},{}<$Q^2$<{}, N={}, $P$={}%".format(title_prefix,xb_bin_edges[i],xb_bin_edges[i+1],q2_bin_edges[j],q2_bin_edges[j+1],N_found, bin_purity)

                    #     plot_2dhist(bin_rec_df[prefix+"phi1"],bin_rec_df[prefix+"t2"],["$\phi$","t"],[[0,360,100],[0,1.8,100]],colorbar=True,xaxis_lines=phi_bin_edges,yaxis_lines=t_bin_edges,
                    #             plot_title=plot_title,units=["None","GeV$^2$"],saveplot=True,pics_dir=out_dir,figsize=(30,18))


    plot_bin_mig_example = 1

    if plot_bin_mig_example:
        for i in range(len(xb_bin_edges)-1):
            for j in range(len(q2_bin_edges)-1):
                prefix = ""
                # How many events were found in this bin?
                bin_rec_df = df[(df[prefix+"xB"] > xb_bin_edges[i]) & 
                            (df[prefix+"xB"] < xb_bin_edges[i+1]) & 
                            (df[prefix+"Q2"] > q2_bin_edges[j]) & 
                            (df[prefix+"Q2"] < q2_bin_edges[j+1]) &
                            (df[prefix+"t2"] > 0.3) & 
                            (df[prefix+"t2"] < 0.4) & 
                            (df[prefix+"phi1"] > 144) & 
                            (df[prefix+"phi1"] < 180)
                            ]
                if len(bin_rec_df) > 1:
                
                    
                    prefix = "Gen"
                    # How many events originated in this bin?
                    bin_gen_df = df[(df[prefix+"xB"] > xb_bin_edges[i]) & 
                                (df[prefix+"xB"] < xb_bin_edges[i+1]) & 
                                (df[prefix+"Q2"] > q2_bin_edges[j]) & 
                                (df[prefix+"Q2"] < q2_bin_edges[j+1]) &
                                (df[prefix+"t2"] > 0.3) & 
                                (df[prefix+"t2"] < 0.4) & 
                                (df[prefix+"phi1"] > 144) & 
                                (df[prefix+"phi1"] < 180)
                                ]
                    
                    prefix = "Gen"
                    # Filter the dataframe for the current xb and Q2 bin
                    #Of the events that were found in this bin, how many originated there?
                    bin_rec_gen_df = bin_rec_df[(bin_rec_df[prefix+"xB"] > xb_bin_edges[i]) & 
                                (bin_rec_df[prefix+"xB"] < xb_bin_edges[i+1]) & 
                                (bin_rec_df[prefix+"Q2"] > q2_bin_edges[j]) & 
                                (bin_rec_df[prefix+"Q2"] < q2_bin_edges[j+1]) &
                                (bin_rec_df[prefix+"t2"] > 0.3) & 
                                (bin_rec_df[prefix+"t2"] < 0.4) & 
                                (bin_rec_df[prefix+"phi1"] > 144) & 
                                (bin_rec_df[prefix+"phi1"] < 180)
                                ]
                    

                    N_pure = len(bin_rec_gen_df)
                    N_found = len(bin_rec_df)
                    N_generated = len(bin_gen_df)
                    bin_purity = N_pure/N_found


                    #Of the events that originated in this bin, how many were found in another bin?

                    print("Bin {}-{} x {}-{} Q2".format(xb_bin_edges[i],xb_bin_edges[i+1],q2_bin_edges[j],q2_bin_edges[j+1]))
                    # Plot a 2D histogram of 'p' vs 't' for the filtered dataframe
                    
                    # Set the title based on the bin ranges and the number of events in the bin

                    print(bin_rec_df[prefix+"phi1"])
                    print(bin_rec_df[prefix+"t2"])

                    #bin purity: of the number found in this bin, how many originated there?
                    #bin efficiency: of the number that originated in this bin, how many were found?
                    #bin migration: of the number that originated in this bin, how many were found in another bin?
                    #bin contamination: of the number found in this bin, how many originated in another bin?
                    #bin background: of the number found in this bin, how many originated in another bin?


                    title_prefixes = ["Gen","Rec"]
                    bin_purity = int(bin_purity*100)

                    for ind,prefix in enumerate(["Gen",""]):
                        title_prefix = title_prefixes[ind]
                        #turn bin purity into a percentage with no decimal places
                        
                        plot_title = "$x_B$ vs $Q^2$, Sim. {}, {}<$x_B$<{},{}<$Q^2$<{}, N={}, $P${}%".format(title_prefix,xb_bin_edges[i],xb_bin_edges[i+1],q2_bin_edges[j],q2_bin_edges[j+1], N_found,bin_purity)

                        print(prefix)
                        print(title_prefix)
                        plot_2dhist(bin_rec_df[prefix+"xB"],bin_rec_df[prefix+"Q2"],["$x_B$","$Q^2$"],[[0.05,0.9,200],[0.5,12,200]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
                                    plot_title=plot_title,units=["None","GeV$^2$"],xbq2plot=True,saveplot=True,pics_dir=out_dir)

                        plot_title = "$\phi$ vs Momentum Transfer t, Sim. {}, {}<$x_B$<{},{}<$Q^2$<{}, N={}, $P$={}%".format(title_prefix,xb_bin_edges[i],xb_bin_edges[i+1],q2_bin_edges[j],q2_bin_edges[j+1],N_found, bin_purity)

                        plot_2dhist(bin_rec_df[prefix+"phi1"],bin_rec_df[prefix+"t2"],["$\phi$","t"],[[0,360,100],[0,1.8,100]],colorbar=True,xaxis_lines=phi_bin_edges,yaxis_lines=t_bin_edges,
                                plot_title=plot_title,units=["None","GeV$^2$"],saveplot=False,pics_dir=out_dir,figsize=(30,18))


    sys.exit()
    matplotlib.rcParams.update({'font.size': 14})

    df = df.sample(frac=0.05)


    prefix = "Gen"#"Gen"



    # Calculate the W2 values
    #W2 = np.sin(df[prefix+"Q2"] / df[prefix+"xB"] - df[prefix+"Q2"] - 0.938**2)
    #tp = np.sin(df[prefix+"phi1"]*3.14159/180*5)#df[prefix+"t2"]
    tp = np.sin(df[prefix+"t1"]*3.14159/180*1000)#df[prefix+"t2"]

    # # Normalize the W2 values to range between 0 and 1
    #norm = plt.Normalize(W2.min(), W2.max())

    prefix = ""#"Gen"
    tp_norm = plt.Normalize(tp.min(), tp.max())
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Left subplot for "gen"
    axs[0].scatter(df["Gen"+"phi1"], df["Gen"+"t1"], marker="o", c=tp, cmap='viridis', norm=tp_norm)
    axs[0].set_xlim([0, 360])
    axs[0].set_ylim([0, 1.8])
    axs[0].set_title('Generated Distribution')
    axs[0].set_xlabel('$\phi$')
    axs[0].set_ylabel('t2 ($GeV^2$)' )
    #fig.colorbar(plt.cm.ScalarMappable(norm=tp_norm, cmap='viridis'), ax=axs[0], label='tp')

    # Right subplot for "rec"
    axs[1].scatter(df["phi1"], df["t1"], marker="o", c=tp, cmap='viridis', norm=tp_norm)
    axs[1].set_xlim([0, 360])
    axs[1].set_ylim([0, 1.8])
    axs[1].set_title('Reconstructed Distribution')
    axs[1].set_xlabel('$\phi$')
    axs[1].set_ylabel('t1 ($GeV^2$)' )
    #fig.colorbar(plt.cm.ScalarMappable(norm=tp_norm, cmap='viridis'), ax=axs[1], label='tp')

    plt.tight_layout()


    """/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/inb/rec
# #     dvpip_events_norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon.pkl
# # dvpip_events_norad_10000_20230704_1143_Fall_2018_Inbending_50nA_recon_high_cuts.pkl
# # dvpip_events_norad_10000_20230705_1041_Fall_2018_Inbending_50nA_recon.pkl
# # dvpip_events_norad_10000_20230705_1043_Fall_2018_Inbending_45nA_recon.pkl
# # dvpip_events_norad_10000_20230705_1046_Fall_2018_Inbending_55nA_recon.pkl
    
# #     sample_df = df.sample(frac=0.1)
# #     """
# #     # #Select data where 0.2<xB<0.25 and 1<Q2<1.5
# #     #df = df[(df["xB"] > 0.2) & (df["xB"] < 0.25) & (df["Q2"] > 3.5) & (df["Q2"] <4)]
# #     # #select data where 0.3<t<0.4 and 72<phi<108
# #     # df = df[(df["t1"] > 0.2) & (df["t1"] < 0.3) & (df["phi1"] > 36) & (df["phi1"] <72)]
# #     # print(df)




# #     # Create a scatter plot with colors varying according to W2
# #     # plt.scatter(df[prefix+"xB"], df[prefix+"Q2"], marker="o", c=W2, cmap='rainbow', norm=norm)

# #     # # Add a colorbar
# #     # plt.colorbar(label='W2')
# #     # plt.xlim([0, 1])
# #     # plt.ylim([0, 11])

# #     # plt.show()
    

# #     # plot_2dhist(df["xB"],df["GenxB"],["$x_B$","$Q^2$"],[[0.05,0.9,100],[0.05,0.9,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
# #     #              plot_title="$x_B$ vs $Q^2$, Exp. {}".format(config),units=["None","GeV$^2$"],xbq2plot=True,saveplot=False,pics_dir=out_dir)

# #     # plot_2dhist(df["Q2"],df["GenQ2"],["$x_B$","$Q^2$"],[[1,11,100],[1,11,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
# #     #              plot_title="$x_B$ vs $Q^2$, Exp. {}".format(config),units=["None","GeV$^2$"],xbq2plot=True,saveplot=False,pics_dir=out_dir)

# #     # plot_2dhist(df["t1"],df["Gent1"],["$x_B$","$Q^2$"],[[0,1.8,100],[0,1.8,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
# #     #              plot_title="$x_B$ vs $Q^2$, Exp. {}".format(config),units=["None","GeV$^2$"],xbq2plot=True,saveplot=False,pics_dir=out_dir)

# #     # plot_2dhist(df["t2"],df["Gent2"],["$x_B$","$Q^2$"],[[0,1.8,100],[0,1.8,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
# #     #              plot_title="$x_B$ vs $Q^2$, Exp. {}".format(config),units=["None","GeV$^2$"],xbq2plot=True,saveplot=False,pics_dir=out_dir)

# #     # plot_2dhist(df["phi1"],df["Genphi1"],["$x_B$","$Q^2$"],[[0,360,100],[0,360,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
# #     #              plot_title="$x_B$ vs $Q^2$, Exp. {}".format(config),units=["None","GeV$^2$"],xbq2plot=True,saveplot=False,pics_dir=out_dir)

# #     plot_2dhist(df["xB"],df["GenxB"]-df['xB'],["$x_B$","Gen$x_B$ -$x_B$"],[[0.05,0.8,100],[-.05,0.05,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
# #                  plot_title="$x_B$ vs $\Delta x_B$",units=["None","None"],xbq2plot=True,saveplot=True,pics_dir="migrations/")

# #     plot_2dhist(df["Q2"],df["GenQ2"]-df['Q2'],["$Q^2$","Gen$Q^2$-$Q^2$"],[[1,11,100],[-.2,.2,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
# #                  plot_title="$Q^2$ vs $\Delta Q^2$",units=["$GeV^2$","$GeV^2$"],xbq2plot=True,saveplot=True,pics_dir="migrations/")

# #     plot_2dhist(df["t1"],df["Gent1"]-df['t1'],["$t1$","Gent1-t1"],[[0,1.7,100],[-.5,.5,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
# #                  plot_title="t1 vs $\Delta$ t1",units=["$GeV^2$","$GeV^2$"],xbq2plot=True,saveplot=True,pics_dir="migrations/")

# #     plot_2dhist(df["t2"],df["Gent2"]-df['t2'],["$t2$","Gent2-t2"],[[0,1.7,100],[-.5,.5,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
# #                  plot_title="t1 vs $\Delta$ t2",units=["$GeV^2$","$GeV^2$"],xbq2plot=True,saveplot=True,pics_dir="migrations/")

# #     plot_2dhist(df["phi1"],df["Genphi1"]-df['phi1'],["$\phi$","Gen$\phi$ - $\phi$"],[[0,360,100],[-20,20,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
# #                  plot_title="$\phi$ vs $\Delta \phi$",units=["None","None"],xbq2plot=True,saveplot=True,pics_dir="migrations/")

# #     plot_2dhist(df["phi2"],df["Genphi2"]-df['phi2'],["$\phi$2","Gen$\phi$2 - $\phi$2"],[[0,360,100],[-20,20,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
# #                  plot_title="$\phi$2 vs $\Delta \phi$2",units=["None","None"],xbq2plot=True,saveplot=True,pics_dir="migrations/")

# #     import matplotlib.pyplot as plt
# #     import matplotlib.cm as cm
# #     import matplotlib

# #     matplotlib.rcParams.update({'font.size': 14})

# #     df = df.sample(frac=0.05)




# #     prefix = "Gen"#"Gen"



# #     # Calculate the W2 values
# #     #W2 = np.sin(df[prefix+"Q2"] / df[prefix+"xB"] - df[prefix+"Q2"] - 0.938**2)
# #     #tp = np.sin(df[prefix+"phi1"]*3.14159/180*5)#df[prefix+"t2"]
# #     tp = np.sin(df[prefix+"t1"]*3.14159/180*1000)#df[prefix+"t2"]

# #     # # Normalize the W2 values to range between 0 and 1
# #     #norm = plt.Normalize(W2.min(), W2.max())

# #     prefix = ""#"Gen"
# #     tp_norm = plt.Normalize(tp.min(), tp.max())
# #     fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# #     # Left subplot for "gen"
# #     axs[0].scatter(df["Gen"+"phi1"], df["Gen"+"t1"], marker="o", c=tp, cmap='viridis', norm=tp_norm)
# #     axs[0].set_xlim([0, 360])
# #     axs[0].set_ylim([0, 1.8])
# #     axs[0].set_title('Generated Distribution')
# #     axs[0].set_xlabel('$\phi$')
# #     axs[0].set_ylabel('t2 ($GeV^2$)' )
# #     #fig.colorbar(plt.cm.ScalarMappable(norm=tp_norm, cmap='viridis'), ax=axs[0], label='tp')

# #     # Right subplot for "rec"
# #     axs[1].scatter(df["phi1"], df["t1"], marker="o", c=tp, cmap='viridis', norm=tp_norm)
# #     axs[1].set_xlim([0, 360])
# #     axs[1].set_ylim([0, 1.8])
# #     axs[1].set_title('Reconstructed Distribution')
# #     axs[1].set_xlabel('$\phi$')
# #     axs[1].set_ylabel('t1 ($GeV^2$)' )
# #     #fig.colorbar(plt.cm.ScalarMappable(norm=tp_norm, cmap='viridis'), ax=axs[1], label='tp')

# #     plt.tight_layout()
# #     #plt.show()
# #     plt.savefig("t1.png")
# #     #plt.scatter(df[prefix+"xB"],df[prefix+"Q2"],marker="o")
# #     #, bins =[xb_bin_edges, q2_bin_edges])
# #     #plt.show()
# #     # plot_2dhist(df[prefix+"xB"],df[prefix+"Q2"],["$x_B$","$Q^2$"],[[0.05,0.9,100],[0.5,12,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
# #     #             plot_title="$x_B$ vs $Q^2$, Exp. {}".format(config),units=["None","GeV$^2$"],xbq2plot=True,saveplot=True,pics_dir=out_dir)

# #     # plot_2dhist(df[prefix+"phi1"],df[prefix+"t1"],["$\phi$","t"],[[0,360,50],[0,1.8,50]],colorbar=True,xaxis_lines=phi_bin_edges,yaxis_lines=t_bin_edges,
# #     #             plot_title="$\phi$ vs Momentum Transfer t, Exp. {}".format(config),units=["None","GeV$^2$"],saveplot=True,pics_dir=out_dir)

# #     sys.exit()