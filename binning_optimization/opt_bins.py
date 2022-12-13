import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.patches import Rectangle

# 1.) Necessary imports.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import argparse
import sys
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib as mpl
from icecream import ic

df = pd.read_pickle("data/df.pkl")

ic(df)

#df.head(10).to_pickle("data/df_head.pkl")

#df = pd.read_pickle("data/df.pkl")

#df = pd.read_pickle("data/df_head.pkl")
ic(df["Q2"])
ic(df["xB"])

#plt.plot(df["Q2"],df["xB"],'o')
#plt.show()

df_type = "Real"
prefix = "Gen" if df_type=="Gen" else ""
df_np = df[["{}Q2".format(prefix),"{}xB".format(prefix).format(prefix)]].copy().to_numpy()

num_cols = df_np.shape[1]
blank_bin_edges = [-1000,1000]
initalized_bin_edges = [blank_bin_edges]*num_cols

q2_bin_edges,xb_bin_edges = [1,2,3,4,6,10],[0.2,.3,0.4,.6,0.8]

initalized = [blank_bin_edges]*num_cols

initalized[0] = q2_bin_edges
initalized[1] = xb_bin_edges

number_of_counts_bin_values, edges = np.histogramdd(df_np, bins=initalized)
weighted_q2_values, edges = np.histogramdd(df_np, bins=initalized,weights=df_np[:,0])
weighted_xB_values, edges = np.histogramdd(df_np, bins=initalized,weights=df_np[:,1])
q2_bin_averages = np.divide(weighted_q2_values,number_of_counts_bin_values).T.reshape(-1,1)
xb_bin_averages_unshaped = np.divide(weighted_xB_values,number_of_counts_bin_values)
xb_bin_averages = xb_bin_averages_unshaped.T.reshape(-1,1)
number_of_counts_bin_values_reshaped = number_of_counts_bin_values.T.reshape(-1,1)

q2_min = edges[0][:-1]
q2_max = edges[0][1:]
xb_min = edges[1][:-1]
xb_max = edges[1][1:]
num_of_binning_vars = 2
all_min = np.array(np.meshgrid(xb_min,q2_min)).T.reshape(-1,num_of_binning_vars)
all_max = np.array(np.meshgrid(xb_max,q2_max)).T.reshape(-1,num_of_binning_vars)


all_together_now = np.concatenate((all_min, all_max), axis=1)

aves_counts_together = np.concatenate((xb_bin_averages, q2_bin_averages,number_of_counts_bin_values_reshaped), axis=1)

ic(aves_counts_together)

all_together_now1 = np.concatenate((all_together_now, aves_counts_together), axis=1)

#all_together_now1 = np.concatenate((all_together_now, xb_bin_averages, q2_bin_averages,number_of_counts_bin_values_reshaped), axis=1)


df_minibin = pd.DataFrame(all_together_now1, columns = ['xmin','qmin','xmax','qmax','xave','qave',str(prefix)+'counts'])

ic(df['Q2'])
ic(df['xB'])

ic(df_minibin)

x = df_minibin["xave"]
y = df_minibin["qave"]
#plt.pcolormesh(x,y,colorsx,cmap=cmap)#norm=mpl.colors.LogNorm())

def plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
            saveplot=False,pics_dir="none",plot_title="none",logger=False,first_label="rad",
            filename="ExamplePlot",units=["",""],extra_data=None):
    
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
    fig, ax = plt.subplots(figsize =(18, 10)) 
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

    for xbin in xb_bin_edges:   
        plt.vlines(xbin,ymin=.2,ymax=14, color='r')

    for ybin in q2_bin_edges:
        plt.hlines(ybin,xmin=.01, xmax=1, color='r')

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



plot_2dhist(df["xB"],df["Q2"],["xB","Q2"],[[0,1,100],[0,10,100]],colorbar=True,)