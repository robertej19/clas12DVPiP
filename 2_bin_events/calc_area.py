


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
from scipy.integrate import dblquad
import numpy as np

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






# turn ice cream messages off
ic.disable()
def bottom_acceptance_bound(x,y):
    return (4-0.938**2)/(1/x-1)-y 
    
    #y = Q^2/(2ME xB) < (10.604-2) / (10.604)
def top_acceptance_bound(x,y):
    return (10.604-2) / (10.604)*2*0.938*x*10.604-y

    
def max_over_ybin_max(ybin_max,x,):
    return max(ybin_max,bottom_acceptance_bound(x,0))


def calc_bin_vol_corr(xbin_min,xbin_max,ybin_min,ybin_max):
    whole_bin_outside = False
    bin_vol = (xbin_max-xbin_min)*(ybin_max-ybin_min)

    x_int_min = xbin_min
    x_int_max = xbin_max

    total_vol = integrate.quad(lambda x: (top_acceptance_bound(x,0)-bottom_acceptance_bound(x,0)), x_int_min, x_int_max)
    
    left_top_cutoff = top_acceptance_bound(xbin_min,0)
    right_top_cutoff = top_acceptance_bound(xbin_max,0) 
    ic(left_top_cutoff)
    ic(ybin_min)
    if left_top_cutoff < ybin_min:
        if right_top_cutoff < ybin_min:
            whole_bin_outside = True
        else:
            #Need to subtract part of f less than ybin min
            ic("only integrating over range where f(x) is less than y bin lower bound")
            integration_bound = root_scalar(top_acceptance_bound,args=(ybin_min), bracket=[xbin_min, xbin_max])
            ic(integration_bound.root)
            top_vol = integrate.quad(lambda x: (top_acceptance_bound(x,0)-ybin_min), xbin_min, integration_bound.root)
        
    elif left_top_cutoff < ybin_max:        
        if right_top_cutoff < ybin_max:
            #Don't need to subtract anything            
            top_vol = [0]
        else:
            #Need to subtract part of f greater than ybin max
            ic("only integrating over range where f(x) is greater than y bin upper bound")
            integration_bound = root_scalar(top_acceptance_bound,args=(ybin_max), bracket=[xbin_min, xbin_max])
            ic(integration_bound.root)
            top_vol = integrate.quad(lambda x: (top_acceptance_bound(x,0)-ybin_max), integration_bound.root, xbin_max)
    
    else:
        #else subtract over whole range
        top_vol = integrate.quad(lambda x: (top_acceptance_bound(x,0)-max(ybin_max,bottom_acceptance_bound(x,0))), x_int_min, x_int_max)


    left_bottom_cutoff = bottom_acceptance_bound(xbin_min,0)
    right_bottom_cutoff = bottom_acceptance_bound(xbin_max,0)

    if left_bottom_cutoff < ybin_min:        
        if right_bottom_cutoff < ybin_min:
            #integrate over whole x range
            bot_vol = integrate.quad(lambda x: (ybin_min-bottom_acceptance_bound(x,0)), x_int_min, x_int_max)
        else:
            # we must stop integration where g(x) = ybin_min
            ic("only integrating over range where g(x) is less than y bin lower bound")
            integration_bound = root_scalar(bottom_acceptance_bound,args=(ybin_min), bracket=[xbin_min, xbin_max])
            ic(integration_bound.root)
            bot_vol = integrate.quad(lambda x: (ybin_min-bottom_acceptance_bound(x,0)), x_int_min, integration_bound.root)
    else:
            bot_vol = [0]


    if whole_bin_outside:
        bin_vol_int = 0
    else:
        bin_vol_int = total_vol[0]-top_vol[0]-bot_vol[0]

    print("x range: ",xbin_min,xbin_max)
    print("y range: ",ybin_min,ybin_max)
    # print ratio with only 2 significant figures
    print("bin volume ratio {:.2f}".format(bin_vol_int/bin_vol))

    return bin_vol_int, bin_vol, bin_vol_int/bin_vol

    

def calc_bin_vol_corr_array(row):
    xbin_min = row['xmin']
    xbin_max = row['xmax']
    ybin_min = row['qmin']
    ybin_max = row['qmax']

    tp_bin_volume = (row['tmax']-row['tmin'])*(row['pmax']-row['pmin'])

    whole_bin_outside = False
    bin_vol = (xbin_max-xbin_min)*(ybin_max-ybin_min)

    x_int_min = xbin_min
    x_int_max = xbin_max

    total_vol = integrate.quad(lambda x: (top_acceptance_bound(x,0)-bottom_acceptance_bound(x,0)), x_int_min, x_int_max)
    
    left_top_cutoff = top_acceptance_bound(xbin_min,0)
    right_top_cutoff = top_acceptance_bound(xbin_max,0) 
    ic(left_top_cutoff)
    ic(ybin_min)
    if left_top_cutoff < ybin_min:
        if right_top_cutoff < ybin_min:
            whole_bin_outside = True
        else:
            #Need to subtract part of f less than ybin min
            ic("only integrating over range where f(x) is less than y bin lower bound")
            integration_bound = root_scalar(top_acceptance_bound,args=(ybin_min), bracket=[xbin_min, xbin_max])
            ic(integration_bound.root)
            top_vol = integrate.quad(lambda x: (top_acceptance_bound(x,0)-ybin_min), xbin_min, integration_bound.root)
        
    elif left_top_cutoff < ybin_max:        
        if right_top_cutoff < ybin_max:
            #Don't need to subtract anything            
            top_vol = [0]
        else:
            #Need to subtract part of f greater than ybin max
            ic("only integrating over range where f(x) is greater than y bin upper bound")
            integration_bound = root_scalar(top_acceptance_bound,args=(ybin_max), bracket=[xbin_min, xbin_max])
            ic(integration_bound.root)
            top_vol = integrate.quad(lambda x: (top_acceptance_bound(x,0)-ybin_max), integration_bound.root, xbin_max)
    
    else:
        #else subtract over whole range
        top_vol = integrate.quad(lambda x: (top_acceptance_bound(x,0)-max(ybin_max,bottom_acceptance_bound(x,0))), x_int_min, x_int_max)


    left_bottom_cutoff = bottom_acceptance_bound(xbin_min,0)
    right_bottom_cutoff = bottom_acceptance_bound(xbin_max,0)

    if left_bottom_cutoff < ybin_min:        
        if right_bottom_cutoff < ybin_min:
            #integrate over whole x range
            bot_vol = integrate.quad(lambda x: (ybin_min-bottom_acceptance_bound(x,0)), x_int_min, x_int_max)
        else:
            # we must stop integration where g(x) = ybin_min
            ic("only integrating over range where g(x) is less than y bin lower bound")
            integration_bound = root_scalar(bottom_acceptance_bound,args=(ybin_min), bracket=[xbin_min, xbin_max])
            ic(integration_bound.root)
            bot_vol = integrate.quad(lambda x: (ybin_min-bottom_acceptance_bound(x,0)), x_int_min, integration_bound.root)
    else:
            bot_vol = [0]

    if whole_bin_outside:
        bin_vol_int = 0
    else:
        bin_vol_int = total_vol[0]-top_vol[0]-bot_vol[0]

    print("x range: ",xbin_min,xbin_max)
    print("y range: ",ybin_min,ybin_max)
    # print ratio with only 2 significant figures
    print("bin volume ratio {:.2f}".format(bin_vol_int/bin_vol))

    if bin_vol_int < 0:
        print("bin_vol_int is negative due to being last xb-q2 bin, using manual estimator instaed")
        manual_bin_vol_estimated_ratio =  .63 #how much is within acceptancefor0.58<xb<0.9, 7.0<q2<11.0
        bin_vol_int = bin_vol* manual_bin_vol_estimated_ratio

    true_total_vol = tp_bin_volume*bin_vol_int
    return pd.Series([bin_vol_int, bin_vol, bin_vol_int/bin_vol,tp_bin_volume,true_total_vol])

    

# Assume df is your DataFrame with columns 'x_min', 'x_max', 'y_min', 'y_max'
fs = filestruct.fs()
self_calc = False
if self_calc:
    print("self calcing")
    xb_bin_edges = fs.xBbins
    q2_bin_edges = fs.Q2bins

    df = pd.DataFrame()#columns=['xmin', 'xmax', 'ymin', 'ymax','nominal_bin_volume','true_bin_volume', 'volume_ratio'])

    #iterate over all xb q2 bins:
    for xmin,xmax in zip(xb_bin_edges[:-1],xb_bin_edges[1:]):
        for ymin,ymax in zip(q2_bin_edges[:-1],q2_bin_edges[1:]):
            bin_vol_int, bin_vol, volume_ratio = calc_bin_vol_corr(xmin,xmax,ymin,ymax)

                    # Add xbin_min, xbin_max, ybin_min, ybin_max, bin_vol_int/bin_vol to the DataFrame
            df_temp = pd.DataFrame({'xmin': [xmin],
                            'xmax': [xmax],
                            'ymin': [ymin],
                            'ymax': [ymax],
                            'nominal_bin_volume': [bin_vol],
                            'true_bin_volume': [bin_vol_int],
                            'volume_ratio': [volume_ratio]})

            df = pd.concat([df, df_temp], ignore_index=True)

    print(df)
else:
    
    #input_dir = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/exp/singles_t2/"
    input_dir = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/exp/singles_t1/"

    #get a list of files in input_dir
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    #iterate over all files
    for file in files:

        print("Binning on {}".format(input_dir+file))
        df = pd.read_pickle(input_dir+file)

        # add a column true_bin_volume
        results = df.apply(calc_bin_vol_corr_array, axis=1)
        results.columns = ['true_xbq2_bin_volume', 'nominal_xbq2_bin_volume', 'volume_ratio','tp_bin_volume','true_total_vol']
        df = pd.concat([df, results], axis=1)
        print(df)
        #save df as pickle file in same directory with "with_area" added to filename
        df.to_pickle(input_dir+"with_area_"+file)


# #Print each row of the dataframe:
# for index, row in df.iterrows():
#     print(row['x_min'], row['x_max'], row['y_min'], row['y_max'], row['volume_ratio'])