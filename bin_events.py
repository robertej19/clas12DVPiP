import uproot
import pandas as pd
import numpy as np
import argparse
import os, sys
from icecream import ic
import matplotlib.pyplot as plt
from copy import copy
from utils.utils import dot
from utils.utils import mag
from utils.utils import mag2
from utils.utils import cosTheta
from utils.utils import angle
from utils.utils import cross
from utils.utils import vecAdd
from utils.utils import pi0Energy
from utils.utils import pi0InvMass
from utils.utils import getPhi
from utils.utils import getTheta
from utils.utils import getEnergy
from utils.utils import readFile
from utils import make_histos
from utils import histo_plotting
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
fs = filestruct.fs()

def bin_df(df_in,df_type="real"):
    prefix = "Gen" if df_type=="Gen" else ""
    df = df_in
    num_counts = []

    print("Binning df: {}".format(df))

    #args.test = False
    q2bins,xBbins, tbins, phibins = fs.q2bins, fs.xBbins, fs.tbins, fs.phibins
    #if args.test:
    #        q2bins,xBbins, tbins, phibins = fs.q2bins_test, fs.xBbins_test, fs.tbins_test, fs.phibins_test

    qrange = [q2bins[0], q2bins[-1]]
    xBrange = [xBbins[0], xBbins[-1]]
    trange = [tbins[0], tbins[-1]]

    if prefix=="Gen":                               
        total_num = df.query('GenQ2>{} and GenQ2<{} and GenxB>{} and GenxB<{} and Gent1>{} and Gent1<{}'.format(*qrange, *xBrange, *trange)).shape[0]
    else:
        total_num = df.query('Q2>{} and Q2<{} and xB>{} and xB<{} and t1>{} and t1<{}'.format(*qrange, *xBrange, *trange)).shape[0]

    for qmin,qmax in zip(q2bins[0:-1],q2bins[1:]):
        print(" \n Q2 bin: {} to {}".format(qmin,qmax))
        query = "{}Q2 > {} and {}Q2 < {}".format(prefix,qmin,prefix,qmax)
        df_q = df.query(query)

        print(df_q)
        for xmin,xmax in zip(xBbins[0:-1],xBbins[1:]):
            #print("        xB bin: {} to {}".format(xmin,xmax))
            query = "{}xB > {} and {}xB < {}".format(prefix,xmin,prefix,xmax)
            df_qx = df_q.query(query)

            for tmin,tmax in zip(tbins[0:-1],tbins[1:]):
                query = "{}t1 > {} and {}t1 < {}".format(prefix,tmin,prefix,tmax)
                df_qxt = df_qx.query(query)

                for pmin,pmax in zip(phibins[0:-1],phibins[1:]):
                    query = "{}phi1 > {} and {}phi1 < {}".format(prefix,pmin,prefix,pmax)
                    df_qxtp =  df_qxt.query(query)
                    

                    mean_q2 =    df_qxtp["{}Q2".format(prefix)].mean(axis=0)
                    mean_xb =    df_qxtp["{}xB".format(prefix)].mean(axis=0)
                    mean_t1 =    df_qxtp["{}t1".format(prefix)].mean(axis=0)
                    mean_phi =  df_qxtp["{}phi1".format(prefix)].mean(axis=0)
                    mean_y =  df_qxtp["{}y".format(prefix)].mean(axis=0)


                    num_counts.append([qmin,xmin,tmin,pmin,qmax,xmax,tmax,pmax,mean_q2,mean_y,mean_xb,mean_t1,mean_phi,len(df_qxtp.index)])



    df_minibin = pd.DataFrame(num_counts, columns = ['qmin','xmin','tmin','pmin','qmax','xmax','tmax','pmax','qave','yave','xave','tave','pave',prefix+'counts'])
    print("Total number of binned events: {}".format(df_minibin[prefix+'counts'].sum()))
    print("Total number of original events: {}".format(total_num))
    return df_minibin


def bin_df_fast(df,df_type="real"):
    prefix = "Gen" if df_type=="Gen" else ""


    df_np = df[["{}Q2".format(prefix),"{}xB".format(prefix),"{}t1".format(prefix),"{}phi1".format(prefix),"{}y".format(prefix)]].copy().to_numpy()

    num_cols = df_np.shape[1]
    blank_bin_edges = [-1000,1000]
    initalized_bin_edges = [blank_bin_edges]*num_cols

    q2_bin_edges,xb_bin_edges, t1_bin_edges, phi1_bin_edges = fs.q2bins, fs.xBbins, fs.tbins, fs.phibins

    print("BIN EDGES ARE:")
    ic(q2_bin_edges)
    ic(xb_bin_edges)
    ic(t1_bin_edges)
    ic(phi1_bin_edges)

    #Get number of columns
    num_cols = df_np.shape[1]


    initalized = [blank_bin_edges]*num_cols

    initalized[0] = q2_bin_edges
    initalized[1] = xb_bin_edges
    initalized[2] = t1_bin_edges
    initalized[3] = phi1_bin_edges


    number_of_counts_bin_values, edges = np.histogramdd(df_np, bins=initalized)

    weighted_q2_values, edges = np.histogramdd(df_np, bins=initalized,weights=df_np[:,0])
    weighted_xB_values, edges = np.histogramdd(df_np, bins=initalized,weights=df_np[:,1])
    weighted_t1_values, edges = np.histogramdd(df_np, bins=initalized,weights=df_np[:,2])
    weighted_phi1_values, edges = np.histogramdd(df_np, bins=initalized,weights=df_np[:,3])
    weighted_y_values, edges = np.histogramdd(df_np, bins=initalized,weights=df_np[:,4])



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


    all_min = np.array(np.meshgrid(q2_min,xb_min,t1_min,phi1_min)).T.reshape(-1,num_of_binning_vars)
    all_max = np.array(np.meshgrid(q2_max,xb_max,t1_max,phi1_max)).T.reshape(-1,num_of_binning_vars)



    all_together_now = np.concatenate((all_min, all_max), axis=1)

    #print(all_together_now.shape)
   # print(xb_bin_averages.shape)

    all_together_now1 = np.concatenate((all_together_now, q2_bin_averages, y_bin_averages,xb_bin_averages, t1_bin_averages, phi1_bin_averages,number_of_counts_bin_values_reshaped), axis=1)

  #  print(all_together_now1)
 #   print(all_together_now1.shape)

    #    print(number_of_counts_bin_values_reshaped)

    df_minibin = pd.DataFrame(all_together_now1, columns = ['qmin','xmin','tmin','pmin','qmax','xmax','tmax','pmax','qave','yave','xave','tave','pave',str(prefix)+'counts'])



    ########## FIX THIS - just include a logic check ##############
    #print("Total number of binned events: {}".format(df_minibin[prefix+'counts'].sum()))
    #print("Total number of original events: {}".format(total_num))
    return df_minibin


if __name__ == "__main__":
    df = pd.read_pickle("pickled_dvpip/f18_bkmrg_in_dvpp_rec_noseccut.pkl")
    df_binned = bin_df(df,df_type="real")
    df_binned.to_pickle("binned_dvpip/f18_bkmrg_in_dvpp_rec_noseccut_binned.pkl")

# df = pd.read_pickle("pickled_data/f18_in_gen.pkl")
# df_binned = bin_df(df,df_type="Gen")
# df_binned.to_pickle("binned_dvpip/f18_in_dvpp_gen_binned.pkl")

# df = pd.read_pickle("pickled_dvpip/f18_in_dvpp_rec_noseccut.pkl")
# df_binned = bin_df(df,df_type="real")
# df_binned.to_pickle("binned_dvpip/f18_in_dvpp_rec_noseccut_binned.pkl")

# df = pd.read_pickle("pickled_dvpip/f18_in_dvpp_exp_noseccut.pkl")
# df_binned = bin_df(df,df_type="real")
# df_binned.to_pickle("binned_dvpip/f18_in_dvpp_exp_noseccut_binned.pkl")

