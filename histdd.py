import uproot
import pandas as pd
import numpy as np
import argparse
import os, sys
from icecream import ic


from copy import copy

from utils import filestruct
pd.set_option('mode.chained_assignment', None)


M = 0.938272081 # target mass
me = 0.5109989461 * 0.001 # electron mass
ebeam = 10.604 # beam energy
pbeam = np.sqrt(ebeam * ebeam - me * me) # beam electron momentum
beam = [0, 0, pbeam] # beam vector
target = [0, 0, 0] # target vector
alpha = 1/137 #Fund const
mp = 0.938 #Mass proton
prefix = alpha/(8*np.pi)
E = 10.6



def bin_df_fast(df,df_type="real"):
    prefix = "Gen" if df_type=="Gen" else ""


    df2 = df[["Q2","xB","t1","phi1","y"]].copy().head(6)
    print(df2)


    dfnp = df2.to_numpy()




    #Get number of columns
    num_cols = dfnp.shape[1]
    blank_bin_edges = [-100,1000]
    q2_bin_edges = [2,3,50]
    xb_bin_edges = [0.1,0.3,0.9]
    t1_bin_edges = [0.1,0.3,9]
    phi1_bin_edges = [0,90,360]

    initalized = [blank_bin_edges]*num_cols

    initalized[0] = q2_bin_edges
    initalized[1] = xb_bin_edges
    initalized[2] = t1_bin_edges
    initalized[3] = phi1_bin_edges


    number_of_counts_bin_values, edges = np.histogramdd(dfnp, bins=initalized)

    weighted_q2_values, edges = np.histogramdd(dfnp, bins=initalized,weights=dfnp[:,0])
    weighted_xB_values, edges = np.histogramdd(dfnp, bins=initalized,weights=dfnp[:,1])
    weighted_t1_values, edges = np.histogramdd(dfnp, bins=initalized,weights=dfnp[:,2])
    weighted_phi1_values, edges = np.histogramdd(dfnp, bins=initalized,weights=dfnp[:,3])
    weighted_y_values, edges = np.histogramdd(dfnp, bins=initalized,weights=dfnp[:,4])



    q2_bin_averages = np.divide(weighted_q2_values,number_of_counts_bin_values).reshape(-1,1)
    xb_bin_averages = np.divide(weighted_xB_values,number_of_counts_bin_values).reshape(-1,1)
    t1_bin_averages = np.divide(weighted_t1_values,number_of_counts_bin_values).reshape(-1,1)
    phi1_bin_averages = np.divide(weighted_phi1_values,number_of_counts_bin_values).reshape(-1,1)
    y_bin_averages = np.divide(weighted_y_values,number_of_counts_bin_values).reshape(-1,1)


    number_of_counts_bin_values_reshaped = number_of_counts_bin_values.reshape(-1,1)



    q2_min = edges[0][:-1]
    q2_max = edges[0][1:]
    xb_min = edges[1][:-1]
    xb_max = edges[1][1:]
    t1_min = edges[2][:-1]
    t1_max = edges[2][1:]
    phi1_min = edges[3][:-1]
    phi1_max = edges[3][1:]


    num_of_binning_vars = 4


    #The ordering is important and non - obvious
    all_min = np.array(np.meshgrid(t1_min,phi1_min,xb_min,q2_min)).T.reshape(-1,num_of_binning_vars)
    all_max = np.array(np.meshgrid(t1_max,phi1_max,xb_max,q2_max)).T.reshape(-1,num_of_binning_vars)
    all_together_now = np.concatenate((all_min, all_max), axis=1)


    all_together_now1 = np.concatenate((all_together_now,   t1_bin_averages, phi1_bin_averages,xb_bin_averages, q2_bin_averages, y_bin_averages,number_of_counts_bin_values_reshaped), axis=1)


    # df_minibin = pd.DataFrame(all_together_now1, columns = ['qmin','xmin','tmin','pmin','qmax','xmax','tmax','pmax','qave','yave','xave','tave','pave',str(prefix)+'counts'])
    df_minibin = pd.DataFrame(all_together_now1, columns = ['tmin','pmin','xmin','qmin','tmax','pmax','xmax','qmax','tave','pave','xave','qave','yave',str(prefix)+'counts'])

    print(df_minibin)

    #print(xb_bin_averages[:,0])





    # for q2_index,(q2_bin_min,q2_bin_max) in enumerate(zip(q2_bin_edges[0:-1],q2_bin_edges[1:])):
    #     print("Q2 bin range of {} to {}".format(q2_bin_min,q2_bin_max))
    #     for xb_index,(xb_bin_min,xb_bin_max) in enumerate(zip(xb_bin_edges[0:-1],xb_bin_edges[1:])):
    #         print("XB bin range of {} to {}".format(xb_bin_min,xb_bin_max))
    #         print(QQQ_bin_values[q2_index][xb_index])




df = pd.read_pickle("/mnt/d/GLOBUS/CLAS12/APS2022/pickled_dvpip/raw_data_f2018_inbending_20220113_dvpip_exp.pkl")
bin_df_fast(df)