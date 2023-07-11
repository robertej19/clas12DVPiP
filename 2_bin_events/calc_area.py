


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

    #print bin values
    #print("x range: ",xbin_min,xbin_max)
    #print("y range: ",ybin_min,ybin_max)
    #print("bin volume: ",bin_vol)


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

    
    ic(left_bottom_cutoff)
    ic(right_bottom_cutoff)

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

# for x in np.linspace(0.1,0.9,10):
#     for y in np.linspace(1,10,10):
#         xbin_min, xbin_max, ybin_min, ybin_max, bin_vol_int,bin_vol  = calc_bin_vol_corr(x,x+0.1,y,y+0.1)
        
        #print out range of bin:
    print("x range: ",xbin_min,xbin_max)
    print("y range: ",ybin_min,ybin_max)
    # print ratio with only 2 significant figures
    print("bin volume ratio {:.2f}".format(bin_vol_int/bin_vol))

    return bin_vol_int, bin_vol, bin_vol_int/bin_vol
    # add xbin_min, xbin_max, ybin_min, ybin_max, bin_vol_int/bin_vol to a dataframe

    

fs = filestruct.fs()
xb_bin_edges = fs.xBbins
q2_bin_edges = fs.Q2bins

df = pd.DataFrame(columns=['xmin', 'xmax', 'ymin', 'ymax','nominal_bin_volume','true_bin_volume', 'volume_ratio'])

#iterate over all xb q2 bins:
for xmin,xmax in zip(xb_bin_edges[:-1],xb_bin_edges[1:]):
    for ymin,ymax in zip(q2_bin_edges[:-1],q2_bin_edges[1:]):
        bin_vol_int, bin_vol, volume_ratio = calc_bin_vol_corr(xmin,xmax,ymin,ymax)

                # Add xbin_min, xbin_max, ybin_min, ybin_max, bin_vol_int/bin_vol to the DataFrame
        df_temp = pd.DataFrame({'x_min': [xmin],
                        'x_max': [xmax],
                        'y_min': [ymin],
                        'y_max': [ymax],
                        'nominal_bin_volume': [bin_vol],
                        'true_bin_volume': [bin_vol_int],
                        'volume_ratio': [volume_ratio]})

        df = pd.concat([df, df_temp], ignore_index=True)

#Print each row of the dataframe:
for index, row in df.iterrows():
    print(row['x_min'], row['x_max'], row['y_min'], row['y_max'], row['volume_ratio'])