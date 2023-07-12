


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

    plotlines = 0
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
rec_in = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/inb/rec/dvpip_events_norad_10000_20230705_1043_Fall_2018_Inbending_45nA_recon.pkl"
rec_out = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/outb/rec/norad_10000_20230703_1814_Fall_2018_Outbending_100_50nA_recon_fid_corr_smear.pkl"
inb = "/mnt/d/GLOBUS/CLAS12/Thesis/1_potential_dvpip_events/inb/exp/20220511_f18_in_combined_157_cor.pkl"
gen = "/mnt/d/GLOBUS/CLAS12/Thesis/1_potential_dvpip_events/gen/lund_10000_20230624_1457/_mnt_d_GLOBUS_CLAS12_Simulation_staging_lund_10000_20230624_1457_lund_files__1.pkl"
#Display nevents in each bin also

rec_in_epgg = "/mnt/d/GLOBUS/CLAS12/Thesis/1_potential_dvpip_events/inb/rec/norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon.pkl"

configs = ["Inb.","Outb."]
files = [rec_in,rec_out]
output_dirs =["/mnt/d/GLOBUS/CLAS12/Thesis/plots/2_dvpip_distros/inb/rec/binned/",
              "/mnt/d/GLOBUS/CLAS12/Thesis/plots/3_binned_distros/outb/gen/"]

xb_bin_edges, q2_bin_edges,t_bin_edges, phi_bin_edges = fs.xBbins, fs.Q2bins, fs.tbins, fs.phibins

for i in range(len(configs)):
    config = configs[i]
    df = pd.read_pickle(files[i])
    out_dir = output_dirs[i]

    """/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/inb/rec
    dvpip_events_norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon.pkl
dvpip_events_norad_10000_20230704_1143_Fall_2018_Inbending_50nA_recon_high_cuts.pkl
dvpip_events_norad_10000_20230705_1041_Fall_2018_Inbending_50nA_recon.pkl
dvpip_events_norad_10000_20230705_1043_Fall_2018_Inbending_45nA_recon.pkl
dvpip_events_norad_10000_20230705_1046_Fall_2018_Inbending_55nA_recon.pkl
    
    sample_df = df.sample(frac=0.1)
    """
    # #Select data where 0.2<xB<0.25 and 1<Q2<1.5
    #df = df[(df["xB"] > 0.2) & (df["xB"] < 0.25) & (df["Q2"] > 3.5) & (df["Q2"] <4)]
    # #select data where 0.3<t<0.4 and 72<phi<108
    # df = df[(df["t1"] > 0.2) & (df["t1"] < 0.3) & (df["phi1"] > 36) & (df["phi1"] <72)]
    # print(df)




    # Create a scatter plot with colors varying according to W2
    # plt.scatter(df[prefix+"xB"], df[prefix+"Q2"], marker="o", c=W2, cmap='rainbow', norm=norm)

    # # Add a colorbar
    # plt.colorbar(label='W2')
    # plt.xlim([0, 1])
    # plt.ylim([0, 11])

    # plt.show()
    

    # plot_2dhist(df["xB"],df["GenxB"],["$x_B$","$Q^2$"],[[0.05,0.9,100],[0.05,0.9,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
    #              plot_title="$x_B$ vs $Q^2$, Exp. {}".format(config),units=["None","GeV$^2$"],xbq2plot=True,saveplot=False,pics_dir=out_dir)

    # plot_2dhist(df["Q2"],df["GenQ2"],["$x_B$","$Q^2$"],[[1,11,100],[1,11,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
    #              plot_title="$x_B$ vs $Q^2$, Exp. {}".format(config),units=["None","GeV$^2$"],xbq2plot=True,saveplot=False,pics_dir=out_dir)

    # plot_2dhist(df["t1"],df["Gent1"],["$x_B$","$Q^2$"],[[0,1.8,100],[0,1.8,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
    #              plot_title="$x_B$ vs $Q^2$, Exp. {}".format(config),units=["None","GeV$^2$"],xbq2plot=True,saveplot=False,pics_dir=out_dir)

    # plot_2dhist(df["t2"],df["Gent2"],["$x_B$","$Q^2$"],[[0,1.8,100],[0,1.8,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
    #              plot_title="$x_B$ vs $Q^2$, Exp. {}".format(config),units=["None","GeV$^2$"],xbq2plot=True,saveplot=False,pics_dir=out_dir)

    # plot_2dhist(df["phi1"],df["Genphi1"],["$x_B$","$Q^2$"],[[0,360,100],[0,360,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
    #              plot_title="$x_B$ vs $Q^2$, Exp. {}".format(config),units=["None","GeV$^2$"],xbq2plot=True,saveplot=False,pics_dir=out_dir)

    # plot_2dhist(df["xB"],df["GenxB"]-df['xB'],["$x_B$","Gen$x_B$ -$x_B$"],[[0.05,0.8,100],[-.05,0.05,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
    #              plot_title="$x_B$ vs $\Delta x_B$",units=["None","None"],xbq2plot=True,saveplot=True,pics_dir="migrations/")

    # plot_2dhist(df["Q2"],df["GenQ2"]-df['Q2'],["$Q^2$","Gen$Q^2$-$Q^2$"],[[1,11,100],[-.2,.2,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
    #              plot_title="$Q^2$ vs $\Delta Q^2$",units=["$GeV^2$","$GeV^2$"],xbq2plot=True,saveplot=True,pics_dir="migrations/")

    # plot_2dhist(df["t1"],df["Gent1"]-df['t1'],["$t1$","Gent1-t1"],[[0,1.7,100],[-.5,.5,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
    #              plot_title="t1 vs $\Delta$ t1",units=["$GeV^2$","$GeV^2$"],xbq2plot=True,saveplot=True,pics_dir="migrations/")

    # plot_2dhist(df["t2"],df["Gent2"]-df['t2'],["$t2$","Gent2-t2"],[[0,1.7,100],[-.5,.5,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
    #              plot_title="t1 vs $\Delta$ t2",units=["$GeV^2$","$GeV^2$"],xbq2plot=True,saveplot=True,pics_dir="migrations/")

    plot_2dhist(df["phi1"],df["Genphi1"]-df['phi1'],["$\phi$","Gen$\phi$ - $\phi$"],[[0,360,100],[-20,20,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
                 plot_title="$\phi$ vs $\Delta \phi$",units=["None","None"],xbq2plot=True,saveplot=True,pics_dir="migrations/")

    plot_2dhist(df["phi2"],df["Genphi2"]-df['phi2'],["$\phi$2","Gen$\phi$2 - $\phi$2"],[[0,360,100],[-20,20,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
                 plot_title="$\phi$2 vs $\Delta \phi$2",units=["None","None"],xbq2plot=True,saveplot=True,pics_dir="migrations/")

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib

    matplotlib.rcParams.update({'font.size': 14})

    df = df.sample(frac=0.05)




    prefix = "Gen"#"Gen"



    # Calculate the W2 values
    #W2 = np.sin(df[prefix+"Q2"] / df[prefix+"xB"] - df[prefix+"Q2"] - 0.938**2)
    #tp = np.sin(df[prefix+"phi1"]*3.14159/180*5)#df[prefix+"t2"]
    tp = np.sin(df[prefix+"t2"]*3.14159/180*1000)#df[prefix+"t2"]

    # # Normalize the W2 values to range between 0 and 1
    #norm = plt.Normalize(W2.min(), W2.max())

    prefix = ""#"Gen"
    tp_norm = plt.Normalize(tp.min(), tp.max())
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Left subplot for "gen"
    axs[0].scatter(df["Gen"+"phi1"], df["Gen"+"t2"], marker="o", c=tp, cmap='viridis', norm=tp_norm)
    axs[0].set_xlim([0, 360])
    axs[0].set_ylim([0, 1.8])
    axs[0].set_title('Generated Distribution')
    axs[0].set_xlabel('$\phi$')
    axs[0].set_ylabel('t2 ($GeV^2$)' )
    #fig.colorbar(plt.cm.ScalarMappable(norm=tp_norm, cmap='viridis'), ax=axs[0], label='tp')

    # Right subplot for "rec"
    axs[1].scatter(df["phi1"], df["t2"], marker="o", c=tp, cmap='viridis', norm=tp_norm)
    axs[1].set_xlim([0, 360])
    axs[1].set_ylim([0, 1.8])
    axs[1].set_title('Reconstructed Distribution')
    axs[1].set_xlabel('$\phi$')
    axs[1].set_ylabel('t2 ($GeV^2$)' )
    #fig.colorbar(plt.cm.ScalarMappable(norm=tp_norm, cmap='viridis'), ax=axs[1], label='tp')

    plt.tight_layout()
    plt.show()
    #plt.scatter(df[prefix+"xB"],df[prefix+"Q2"],marker="o")
    #, bins =[xb_bin_edges, q2_bin_edges])
    #plt.show()
    # plot_2dhist(df[prefix+"xB"],df[prefix+"Q2"],["$x_B$","$Q^2$"],[[0.05,0.9,100],[0.5,12,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
    #             plot_title="$x_B$ vs $Q^2$, Exp. {}".format(config),units=["None","GeV$^2$"],xbq2plot=True,saveplot=True,pics_dir=out_dir)

    # plot_2dhist(df[prefix+"phi1"],df[prefix+"t1"],["$\phi$","t"],[[0,360,50],[0,1.8,50]],colorbar=True,xaxis_lines=phi_bin_edges,yaxis_lines=t_bin_edges,
    #             plot_title="$\phi$ vs Momentum Transfer t, Exp. {}".format(config),units=["None","GeV$^2$"],saveplot=True,pics_dir=out_dir)

    sys.exit()
    # plot_2dhist(df["GenxB"],df["GenQ2"],["$x_B$","$Q^2$"],[[0.05,0.8,100],[0.5,11,100]],colorbar=True,xaxis_lines=xb_bin_edges,yaxis_lines=q2_bin_edges,
    #             plot_title="$x_B$ vs $Q^2$, Exp. {}".format(config),units=["None","GeV$^2$"],xbq2plot=True,saveplot=False,pics_dir=out_dir)

    # plot_2dhist(df["Genphi1"],df["Gent1"],["$\phi$","t"],[[0,360,50],[0,1.8,50]],colorbar=True,xaxis_lines=phi_bin_edges,yaxis_lines=t_bin_edges,
    #             plot_title="$\phi$ vs Momentum Transfer t, Exp. {}".format(config),units=["None","GeV$^2$"],saveplot=False,pics_dir=out_dir)

    # df = df[(df["GenxB"] > 0.2) & (df["GenxB"] < 0.25) & (df["GenQ2"] > 3.5) & (df["GenQ2"] <4)]
    # df = df[(df["Gent1"] > 0.2) & (df["Gent1"] < 0.3) & (df["Genphi1"] > 36) & (df["Genphi1"] <72)]
    # print(df)

    for i in range(len(xb_bin_edges)-1):
        for j in range(len(q2_bin_edges)-1):
            # Filter the dataframe for the current xb and Q2 bin
            bin_df = df[(df[prefix+"xB"] > xb_bin_edges[i]) & 
                        (df[prefix+"xB"] < xb_bin_edges[i+1]) & 
                        (df[prefix+"Q2"] > q2_bin_edges[j]) & 
                        (df[prefix+"Q2"] < q2_bin_edges[j+1])]
            
            if len(bin_df) > 1:

                print("Bin {}-{} x {}-{} Q2".format(xb_bin_edges[i],xb_bin_edges[i+1],q2_bin_edges[j],q2_bin_edges[j+1]))
                # Plot a 2D histogram of 'p' vs 't' for the filtered dataframe
                
                # Set the title based on the bin ranges and the number of events in the bin
                plot_title = "$\phi$ vs Momentum Transfer t, Exp. {}, {}<$x_B$<{},{}<$Q^2$<{}, N={}".format(config,xb_bin_edges[i],xb_bin_edges[i+1],q2_bin_edges[j],q2_bin_edges[j+1], len(bin_df))

                print(bin_df[prefix+"phi1"])
                print(bin_df[prefix+"t2"])
                plot_2dhist(bin_df[prefix+"phi1"],bin_df[prefix+"t2"],["$\phi$","t"],[[0,360,50],[0,1.8,50]],colorbar=True,xaxis_lines=phi_bin_edges,yaxis_lines=t_bin_edges,
                        plot_title=plot_title,units=["None","GeV$^2$"],saveplot=True,pics_dir=out_dir,figsize=(30,18))

