import uproot
import pandas as pd
import numpy as np
import argparse
import itertools
import os, sys
from icecream import ic
import matplotlib
#matplotlib.use('Agg') 

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
from convert_root_to_pickle import convert_GEN_NORAD_root_to_pkl
from convert_root_to_pickle import convert_GEN_RAD_root_to_pkl
from convert_root_to_pickle import convert_RECON_NORAD_root_to_pkl
from convert_root_to_pickle import convert_RECON_RAD_root_to_pkl
import pickle_analysis
from root2pickleEpggRec import root2pickle
pd.set_option('mode.chained_assignment', None)

import random 
import sys
import os, subprocess
import argparse
import shutil
import time
from datetime import datetime 
import json
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


g = True

def get_gamma(x,q2):
    a8p = 1/137*(1/(8*3.14159))
    #print(a8p)
    energies = [10.604]
    for e in energies:
        y = q2/(2*x*e*mp)
        num = 1-y-q2/(4*e*e)
        denom = 1- y + y*y/2 + q2/(4*e*e)
        #print(y,q2,e,num,denom)
        epsi = num/denom
        gamma = 1/(e*e)*(1/(1-epsi))*(1-x)/(x*x*x)*a8p*q2/(0.938*.938)

    return [gamma, epsi]

if g:
    df6 = pd.read_csv("xs_clas6.csv")
    print(df6)

    #for index,row in df6.iterrows():
    #    print(row)
    #    if row.q<2:
    #        row.q = 2
    

    q2bins,xBbins, tbins, phibins = [fs.q2bins, fs.xBbins, fs.tbins, fs.phibins]
    var_bin_arr =  [q2bins,xBbins, tbins, phibins]
    var_arr = ['q','x','t','p']

    for var, varbins in zip(var_arr, var_bin_arr):
        for varmin,varmax in zip(varbins[0:-1],varbins[1:]):
            df6.loc[((df6[var] > varmin) & (df6[var]<=varmax)) , '{}min'.format(var)] = varmin



    df6.to_pickle("xs_clas6_binned.pkl")
    print(df6)
    sys.exit()

    for col in df6[["q","x","t","p"]].columns:
        print(col)
        print(df6[col].unique())


    df6 = df6.query("q<4 and q>3.5 and x>0.4 and x<0.5 and t>0.6 and t<1")
    df6 = df6[["p","dsdtdp","stat","sys"]]
    print(df6)
    
    df = pd.read_pickle("F18_In_binned_events.pkl")
    df = df.query("qmin==3.5 and xmin==0.45 and tmin==0.6 ")
    print(df)

    sample = pd.read_pickle("/mnt/d/GLOBUS/CLAS12/simulations/production/F2018_In_Norad/rec_and_gen_binned_events_meta.pkl")
    sample = sample.query("qmin==3.5 and xmin==0.45 and tmin==0.6")
    sample = sample[["qmin","xmin","tmin","pmin","rec_sum","gen_sum"]]
    dfout = pd.merge(df,sample,how='inner', on=['qmin','xmin','tmin','pmin'])
    dfout.loc[:,"acc_corr"] = dfout["rec_sum"]/dfout["gen_sum"]
    dfout.loc[:,"N_corr"] = dfout["counts"]/dfout["acc_corr"]
    dfout.loc[:,"Lumi"] = 5.5E40
    dfout.loc[:,"qmax"] = dfout["qmin"]+0.5
    dfout.loc[:,"xmax"] = dfout["xmin"]+0.05
    dfout.loc[:,"tmax"] = dfout["tmin"]+0.2
    dfout.loc[:,"pmax"] = dfout["pmin"]+18
    dfout.loc[:,"binvol"] = (dfout["qmax"]-dfout["qmin"])*(dfout["xmax"]-dfout["xmin"])*(dfout["tmax"]-dfout["tmin"])*(dfout["pmax"]-dfout["pmin"])*3.14159/180
    dfout.loc[:,"xsec"] = dfout["N_corr"]/dfout["Lumi"]/dfout["binvol"]
    dfout.loc[:,"gamma"] = get_gamma((dfout["xmin"]+dfout["xmax"])/2,(dfout["qmin"]+dfout["qmax"])/2)[0]
    dfout.loc[:,"epsi"] = get_gamma((dfout["xmin"]+dfout["xmax"])/2,(dfout["qmin"]+dfout["qmax"])/2)[1]
    dfout.loc[:,"red_xsec"] = dfout["xsec"]/dfout["gamma"]
    dfout.loc[:,"red_xsecnb"] = dfout["red_xsec"]*1E33
    dfout.loc[:,"red_xsecnb_err"] = dfout["red_xsecnb"]*np.sqrt(1/dfout["counts"]+1/dfout["rec_sum"] + 1/dfout["gen_sum"])
    dfout.loc[:,"frac_err"] = dfout["red_xsecnb_err"]/dfout["red_xsecnb"]
    dfout.loc[:,"p"] = dfout["pmin"]+9
    dfout = dfout[["p","acc_corr","red_xsecnb","red_xsecnb_err"]]
    print(dfout)

    dffinal = pd.merge(dfout,df6,how='inner', on='p')
    dffinal = dffinal[dffinal['acc_corr'] > 0.005]
    print(dffinal)

    fig, ax = plt.subplots(figsize =(16, 10)) 

    x = dffinal["p"]
    y1 = dffinal["red_xsecnb"]
    y_err1 =   dffinal["red_xsecnb_err"]
    y2 = dffinal["dsdtdp"]
    y_err2 =   dffinal["stat"]

    plt.rcParams["font.size"] = "20"


    ax.bar(x, y2,  yerr=y_err2, align='center', alpha=0.5, color="black", ecolor="black", capsize=10, width=16, label="CLAS6")
    ax.bar(x, y1, yerr=y_err1, align='center', alpha=0.5, color="red", ecolor='red', capsize=10, width=16, label="CLAS12")

    
    #ax.set_ylim(0,100)
    #ax.set_xticks(p_pos[::2])
    #ax.set_xticklabels(p_labels[::2])
    #ax.set_title('Simualtions: Gen and Rec Counts, $Q^2$ bin: {} GeV$^2$, $x_B$ bin: {}, t bin: {} GeV$^2$'.format(q_bin, x_bin, t_bin))
    #ax.yaxis.grid(True)

    #if float(q_bin)<10:
    #    q_label = "0"+str(q_bin)

    #print(q_label)


    plt_title = "Reduced Cross Section, Q2= {}, xB = {}, -t = {}".format(2.25,0.34,0.2)
    ax.set_ylabel('Reduced Cross Section (nb/GeV$^2$)')
    plt.rcParams["font.size"] = "20"
    ax.yaxis.label.set_size(20) 
    ax.set_title(plt_title)

    ax.set_xlabel('Lepton-Hadron Plane Angle')
    plt.rcParams["font.size"] = "20"
    ax.xaxis.label.set_size(20) 

    ax.legend(loc="best")
    #plt_title = base_plot_dir+str(t_bin)+"/x_{}_q_{}_acc_inv.png".format(str(x_bin),q_label)
    plt.show()
    #plt.savefig(plt_title)

    # ax.bar(x, y, align='center', alpha=1, color="black", capsize=10, width=16)
    # ax.bar(x, yg, align='center', alpha=0.5, color="red", capsize=10, width=16)

    # ax.set_ylabel('Simualtions: Gen and Rec Counts')
    # #ax.set_ylim(0,100)
    # #ax.set_xticks(p_pos[::2])
    # #ax.set_xticklabels(p_labels[::2])
    # ax.set_title('Simualtions: Gen and Rec Counts, $Q^2$ bin: {} GeV$^2$, $x_B$ bin: {}, t bin: {} GeV$^2$'.format(q_bin, x_bin, t_bin))
    # #ax.yaxis.grid(True)

    # if float(q_bin)<10:
    #     q_label = "0"+str(q_bin)

    # print(q_label)

    # plt_title = base_plot_dir+str(t_bin)+"/x_{}_q_{}_acc_inv.png".format(str(x_bin),q_label)
    # plt.savefig(plt_title)






    





#     pd.concat(
#     objs,
#     axis=0,
#     join="outer",
#     ignore_index=False,
#     keys=None,
#     levels=None,
#     names=None,
#     verify_integrity=False,
#     copy=True,
# )
    print(dfout)
sys.exit()


#outname = recon_file.split(".")[0]
#output_loc_event_pkl_after_cuts = dirname+run+"/binned_pickles/"+outname+"_reconstructed_events_after_cuts.pkl"
df = pd.read_pickle(output_loc_event_pkl_after_cuts)
#df = df.query("Q2 > 2 and Q2 < 2.5 and xB < 0.38 and xB>0.3 and t>0.2 and t<0.3")

# print(df.shape)

# x_data = df["phi1"]
# plot_title = "F 2018 Inbending, epgg, all exclusivity cuts"

# #plot_title = "F 2018 Inbending, epgg, no exclusivity cuts"

# vars = ["XB (GeV)"]
# make_histos.plot_1dhist(x_data,vars,ranges="none",second_x="none",logger=False,first_label="F18IN",second_label="norad",
#             saveplot=False,pics_dir="none",plot_title=plot_title,first_color="blue",sci_on=False)

# sys.exit()

df_gen = pd.read_pickle(output_loc_event_pkl_all_gen_events)
#df = pd.read_pickle(save_base_dir+"100_20211103_1524_merged_Fall_2018_Inbending_gen_all_generated_events_all_generated_events.pkl")
for col in df.columns:
    print(col)

df['t1'] = df['t']
orginial_sum = df.shape[0]


