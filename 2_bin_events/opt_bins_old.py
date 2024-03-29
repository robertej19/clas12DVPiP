import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.patches import Rectangle
from scipy.optimize import root_scalar


import scipy.integrate as integrate

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
from utils import filestruct

fs = filestruct.fs()


inbending_data = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/inb/exp/final_inbending_exclusive.pkl"
outbending_data = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/outb/exp/final_outbending_exclusive.pkl"

inb = "/mnt/d/GLOBUS/CLAS12/Thesis/1_potential_dvpip_events/inb/exp/20220511_f18_in_combined_157_cor.pkl"

df = pd.read_pickle(inb)
# print(df)

# #find events where Q2 is greater than 6
# #df = df[df["Q2"]>8]
# #print(df)

# #make a 1d histogram of t distribution

# #define top bounding function over range x
def top_acceptance_bound(x, a, b, c):
    #return a*x**2/(b**2 + x**2) + c
    return (10.604-2) / (10.604)*2*0.938*x*10.604

#define bottom bounding function over range x
def bottom_acceptance_bound(x, a, b, c):
    #return a*x**2/(b**2 + x**2) + c
    #return exponential of x:
    return (4-0.938**2)/(1/x-1)
    #return (4-0.938**2)/(1/x-1)
    #Q = (4 - m) / (1/x - 1)
    
def max_over_ybin_max(ybin_max,x,a_param1,b_param1,c_param1):
    return max(ybin_max,bottom_acceptance_bound(x,a_param1,b_param1,c_param1))



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


xb_bin_edges, q2_bin_edges = [0.175,0.25,.3,0.35,.4,0.45,.5,.6],	[0,1,1.5,2,2.5,3,3.5,4,5,6,8,11]
        
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

def calc_bin_vol_corr(xbin_min,xbin_max,ybin_min,ybin_max):

    whole_bin_outside = False
    bin_vol = (xbin_max-xbin_min)*(ybin_max-ybin_min)
    print(bin_vol)




    # try:
    #     root = root_scalar(bottom_acceptance_bound,args=(a_param1,b_param1,c_param1-ybin_min), bracket=[xbin_min, xbin_max])
    # # if root cannot be found, print error message
    # except ValueError:
    #     print("Root not found")
    #     sys.exit()
    



    # from scipy.optimize import fmin_tnc
    # from scipy import optimize as opt

    # #g_min = fmin_tnc(bottom_acceptance_bound,x0=((xbin_min+xbin_max)/2),approx_grad=True,args=(a_param1,b_param1,c_param1), bounds = ((xbin_min,xbin_max),))
    # #g_max = fmin_tnc(-bottom_acceptance_bound,x0=((xbin_min+xbin_max)/2),approx_grad=True,args=(a_param1,b_param1,c_param1), bounds = ((xbin_min,xbin_max),))

    # g_min = opt.fminbound(lambda x: bottom_acceptance_bound(x,a_param1,b_param1,c_param1), xbin_min,xbin_max,xtol=1e-12)
    # g_max = opt.fminbound(lambda x: -bottom_acceptance_bound(x,a_param1,b_param1,c_param1), xbin_min,xbin_max,xtol=1e-12)


    

    x_int_min = xbin_min
    x_int_max = xbin_max



    # total_vol = integrate.quad(lambda x: (top_acceptance_bound(x,a_param,b_param,c_param)-bottom_acceptance_bound(x,a_param1,b_param1,c_param1)), x_int_min, x_int_max)
    # print(total_vol)
    # top_vol = integrate.quad(lambda x: (top_acceptance_bound(x,a_param,b_param,c_param)-ybin_max), x_int_min, x_int_max)
    
    #ymin_intercept = 0.6
    # 
    # bot_vol0 = integrate.quad(lambda x: (ybin_min-bottom_acceptance_bound(x,a_param1,b_param1,c_param1)), x_int_min, ymin_intercept)
    # bot_vol1 = integrate.quad(lambda x: (bottom_acceptance_bound(x,a_param1,b_param1,c_param1)), ymin_intercept, x_int_max)

    # bot_vol = bot_vol0+bot_vol1

    # bin_vol_int = total_vol[0]-top_vol[0]#-bot_vol[0]
    # print(bin_vol_int)


    total_vol = integrate.quad(lambda x: (top_acceptance_bound(x,a_param,b_param,c_param)-bottom_acceptance_bound(x,a_param1,b_param1,c_param1)), x_int_min, x_int_max)
    
    
    min_f_val = top_acceptance_bound(xbin_min,a_param,b_param,c_param)
    max_f_val = top_acceptance_bound(xbin_max,a_param,b_param,c_param) 
    ic(min_f_val)
    ic(ybin_min)
    if min_f_val < ybin_min:
        if max_f_val < ybin_min:
            whole_bin_outside = True
        else:
            #Need to subtract part of f less than ybin min
            ic("only integrating over range where f(x) is less than y bin lower bound")
            integration_bound = root_scalar(top_acceptance_bound,args=(a_param,b_param,c_param-ybin_min), bracket=[xbin_min, xbin_max])
            ic(integration_bound.root)
            top_vol = integrate.quad(lambda x: (top_acceptance_bound(x,a_param,b_param,c_param)-ybin_min), xbin_min, integration_bound.root)
            #make top_vol negative
            #top_vol = [-top_vol[0],top_vol[1]]
        
    elif min_f_val < ybin_max:        
        if max_f_val < ybin_max:
            #Don't need to subtract anything            
            top_vol = [0]
        else:
            #Need to subtract part of f greater than ybin max
            ic("only integrating over range where f(x) is greater than y bin upper bound")
            integration_bound = root_scalar(top_acceptance_bound,args=(a_param,b_param,c_param-ybin_max), bracket=[xbin_min, xbin_max])
            ic(integration_bound.root)
            top_vol = integrate.quad(lambda x: (top_acceptance_bound(x,a_param,b_param,c_param)-ybin_max), integration_bound.root, xbin_max)
    
    else:
        #else subtract over whole range
        top_vol = integrate.quad(lambda x: (top_acceptance_bound(x,a_param,b_param,c_param)-max(ybin_max,bottom_acceptance_bound(x,a_param1,b_param1,c_param1))), x_int_min, x_int_max)


    min_g_val = bottom_acceptance_bound(xbin_min,a_param1,b_param1,c_param1)
    max_g_val = bottom_acceptance_bound(xbin_max,a_param1,b_param1,c_param1) 

    
    ic(min_g_val)
    ic(max_g_val)

    if min_g_val < ybin_min:        
        if max_g_val < ybin_min:
            #integrate over whole x range
            bot_vol = integrate.quad(lambda x: (ybin_min-bottom_acceptance_bound(x,a_param1,b_param1,c_param1)), x_int_min, x_int_max)
        else:
            # we must stop integration where g(x) = ybin_min
            ic("only integrating over range where g(x) is less than y bin lower bound")
            integration_bound = root_scalar(bottom_acceptance_bound,args=(a_param1,b_param1,c_param1-ybin_min), bracket=[xbin_min, xbin_max])
            ic(integration_bound.root)
            bot_vol = integrate.quad(lambda x: (ybin_min-bottom_acceptance_bound(x,a_param1,b_param1,c_param1)), x_int_min, integration_bound.root)
    else:
        # if min_g_val > ybin_max:
        #     whole_bin_outside = True
        # else:
        #     #else don't need to subtract anything
            bot_vol = [0]


    if whole_bin_outside:
        bin_vol_int = 0
    else:
        bin_vol_int = total_vol[0]-top_vol[0]-bot_vol[0]

    #print out range of bin:
    print("x range: ",xbin_min,xbin_max)
    print("y range: ",ybin_min,ybin_max)
    print("bin volume: ratio ",bin_vol_int/bin_vol)


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


    #make x range for plotting
    x_range = np.linspace(xmin,xmax,1000)
    #populate yvals with the y values for the fit
    yvals = top_line(x_range,a_param,b_param,c_param)
    #plot the fit
    plt.plot(x_range,yvals,'b')
    #make x range for plotting
    #populate yvals with the y values for the fit
    yvals_bottom = plot_line(x_range,a_param1,b_param1,c_param1)
    #plot the fit
    plt.plot(x_range,yvals_bottom,'k')


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



    


#iterate over all xb q2 bins:
for xmin,xmax in zip(xb_bin_edges[:-1],xb_bin_edges[1:]):
    for ymin,ymax in zip(q2_bin_edges[:-1],q2_bin_edges[1:]):
        calc_bin_vol_corr(xmin,xmax,ymin,ymax)


plot_2dhist(df["xB"],df["Q2"],["xB","Q2"],[[0,1,200],[0,11,200]],colorbar=True,)
