


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
rec_in = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/inb/rec/norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon_fid_corr_smear.pkl"
rec_out = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/outb/rec/norad_10000_20230703_1814_Fall_2018_Outbending_100_50nA_recon_fid_corr_smear.pkl"
inb = "/mnt/d/GLOBUS/CLAS12/Thesis/1_potential_dvpip_events/inb/exp/20220511_f18_in_combined_157_cor.pkl"

#Display nevents in each bin also

configs = ["Inb.","Outb."]
files = [rec_in,rec_out]
output_dirs =["/mnt/d/GLOBUS/CLAS12/Thesis/plots/3_binned_distros/inb/exp/",
              "/mnt/d/GLOBUS/CLAS12/Thesis/plots/3_binned_distros/outb/exp/"]

xb_bin_edges, q2_bin_edges,t_bin_edges, phi_bin_edges = fs.xBbins, fs.Q2bins, fs.tbins, fs.phibins

for i in range(len(configs)):
    config = configs[i]
    df = pd.read_pickle(files[i])
    out_dir = output_dirs[i]

    #Select data where 0.2<xB<0.25 and 1<Q2<1.5
    df = df[(df["xB"] > 0.2) & (df["xB"] < 0.25) & (df["Q2"] > 3.5) & (df["Q2"] <4)]
    #select data where 0.3<t<0.4 and 72<phi<108
    df = df[(df["t1"] > 0.2) & (df["t1"] < 0.3) & (df["phi1"] > 36) & (df["phi1"] <72)]
    print(df)


    plot_2dhist(df["xB"],df["Q2"],["$x_B$","$Q^2$"],[[0.05,0.9,100],[0.5,12,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
                plot_title="$x_B$ vs $Q^2$, Exp. {}".format(config),units=["None","GeV$^2$"],xbq2plot=True,saveplot=False,pics_dir=out_dir)

    plot_2dhist(df["phi1"],df["t1"],["$\phi$","t"],[[0,360,50],[0,1.8,50]],colorbar=True,xaxis_lines=phi_bin_edges,yaxis_lines=t_bin_edges,
                plot_title="$\phi$ vs Momentum Transfer t, Exp. {}".format(config),units=["None","GeV$^2$"],saveplot=False,pics_dir=out_dir)

    plot_2dhist(df["GenxB"],df["GenQ2"],["$x_B$","$Q^2$"],[[0.05,0.8,100],[0.5,11,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
                plot_title="$x_B$ vs $Q^2$, Exp. {}".format(config),units=["None","GeV$^2$"],xbq2plot=True,saveplot=False,pics_dir=out_dir)

    plot_2dhist(df["Genphi1"],df["Gent1"],["$\phi$","t"],[[0,360,50],[0,1.8,50]],colorbar=True,xaxis_lines=phi_bin_edges,yaxis_lines=t_bin_edges,
                plot_title="$\phi$ vs Momentum Transfer t, Exp. {}".format(config),units=["None","GeV$^2$"],saveplot=False,pics_dir=out_dir)

    df = df[(df["GenxB"] > 0.2) & (df["GenxB"] < 0.25) & (df["GenQ2"] > 3.5) & (df["GenQ2"] <4)]
    df = df[(df["Gent1"] > 0.2) & (df["Gent1"] < 0.3) & (df["Genphi1"] > 36) & (df["Genphi1"] <72)]
    print(df)

    # for i in range(len(xb_bin_edges)-1):
    #     for j in range(len(q2_bin_edges)-1):
    #         # Filter the dataframe for the current xb and Q2 bin
    #         bin_df = df[(df["xB"] > xb_bin_edges[i]) & 
    #                     (df["xB"] < xb_bin_edges[i+1]) & 
    #                     (df["Q2"] > q2_bin_edges[j]) & 
    #                     (df["Q2"] < q2_bin_edges[j+1])]
            
    #         if len(bin_df) > 0:

    #             # Plot a 2D histogram of 'p' vs 't' for the filtered dataframe
                
    #             # Set the title based on the bin ranges and the number of events in the bin
    #             plot_title = "$\phi$ vs Momentum Transfer t, Exp. {}, {}<$x_B$<{},{}<$Q^2$<{}, N={}".format(config,xb_bin_edges[i],xb_bin_edges[i+1],q2_bin_edges[j],q2_bin_edges[j+1], len(bin_df))

    #             plot_2dhist(bin_df["phi1"],bin_df["t2"],["$\phi$","t"],[[0,360,50],[0,1.8,50]],colorbar=True,xaxis_lines=phi_bin_edges,yaxis_lines=t_bin_edges,
    #                     plot_title=plot_title,units=["None","GeV$^2$"],saveplot=True,pics_dir=out_dir,figsize=(30,18))

