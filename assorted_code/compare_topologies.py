import uproot
import pandas as pd
import numpy as np
import argparse
import itertools
import os, sys
from icecream import ic
import matplotlib
matplotlib.use('Agg') 


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
from utils import get_integrated_lumi
import matplotlib as mpl

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
import matplotlib as mpl

# 1.) Necessary imports.    
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import argparse
import sys 
import pandas as pd
from matplotlib.patches import Rectangle

from matplotlib import pylab as plt

import numpy as np
from scipy.optimize import minimize

from utils import filestruct
pd.set_option('mode.chained_assignment', None)

import random 
import sys
import os, subprocess
import argparse
import shutil
import time
from datetime import datetime 
import json

from convert_root_to_pickle import convert_GEN_NORAD_root_to_pkl
from convert_root_to_pickle import convert_GEN_RAD_root_to_pkl
from convert_root_to_pickle import convert_REC_RAD_root_to_pkl
from convert_root_to_pickle import convert_REC_NORAD_root_to_pkl
from convert_root_to_pickle import convert_real_to_pkl



from utils import filestruct
pd.set_option('mode.chained_assignment', None)

import random 
import sys
import os, subprocess
import argparse
import shutil
import time
from datetime import datetime 
import json

# For analysis flow
from make_dvpip_cuts import makeDVpi0
from bin_events import bin_df





M = 0.938272081 # target mass
me = 0.5109989461 * 0.001 # electron mass
ebeam = 10.604 # beam energy
pbeam = np.sqrt(ebeam * ebeam - me * me) # beam electron momentum
beam = [0, 0, pbeam] # beam vector
target = [0, 0, 0] # target vector
alpha = 1/137 #Fund const
mp = 0.938 #Mass proton
prefix = alpha/(8*np.pi)
fs = filestruct.fs()
E = 10.604





"""
TO DO:
include logging for exclusivty cut usage
include logic switch for GenQ2>1, GenW>2
fix plotting ranges on 1D histos
"""


def get_gamma(x,q2,BeamE):
    a8p = 1/137*(1/(8*3.14159))
    energies = [BeamE]
    for e in energies:
        y = q2/(2*x*e*mp)
        num = 1-y-q2/(4*e*e)
        denom = 1- y + y*y/2 + q2/(4*e*e)
        #print(y,q2,e,num,denom)
        epsi = num/denom
        gamma = 1/(e*e)*(1/(1-epsi))*(1-x)/(x*x*x)*a8p*q2/(0.938*.938)

    return [gamma, epsi]


def expand_clas6(df):
    q2bins,xBbins, tbins, phibins = fs.q2bins, fs.xBbins, fs.tbins, fs.phibins
    print(df)

    qrange = [q2bins[0], q2bins[-1]]
    xBrange = [xBbins[0], xBbins[-1]]
    trange = [tbins[0], tbins[-1]]

    data_vals = []

    for qmin,qmax in zip(q2bins[0:-1],q2bins[1:]):
        print(" \n Q2 bin: {} to {}".format(qmin,qmax))
        query = "qmin=={}".format(qmin)
        df_q = df.query(query)

        for xmin,xmax in zip(xBbins[0:-1],xBbins[1:]):
            print("        xB bin: {} to {}".format(xmin,xmax))
            query = "xmin=={}".format(xmin)
            df_qx = df_q.query(query)

            for tmin,tmax in zip(tbins[0:-1],tbins[1:]):
                #print("                 t bin: {} to {}".format(tmin,tmax))
                query = "tmin=={}".format(tmin)
                df_qxt = df_qx.query(query)

                for pmin,pmax in zip(phibins[0:-1],phibins[1:]):
                    #print("                             p bin: {} to {}".format(pmin,pmax))
                    query = "pmin=={}".format(pmin)
                    df_qxtp =  df_qxt.query(query)
                    if df_qxtp.empty:
                        data_vals.append([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,qmin,xmin,tmin,pmin,qmax,xmax,tmax,pmax])
                    else:                    
                        list_value = [df_qxtp["q"].values[0],df_qxtp["x"].values[0],df_qxtp["t"].values[0],df_qxtp["p"].values[0],df_qxtp["dsdtdp"].values[0],df_qxtp["stat"].values[0],df_qxtp["sys"].values[0],qmin,xmin,tmin,pmin,qmax,xmax,tmax,pmax]
                        print(list_value)
                        data_vals.append(list_value)

    df_spaced = pd.DataFrame(data_vals, columns = ['q','x','t','p','dsdtdp','stat','sys','qmin','xmin','tmin','pmin','qmax','xmax','tmax','pmax'])
    # df_minibin = pd.DataFrame(num_counts, columns = ['qmin','xmin','tmin','pmin','qave','xave','tave','pave',prefix+'counts'])
    # print("Total number of binned events: {}".format(df_minibin[prefix+'counts'].sum()))
    # print("Total number of original events: {}".format(total_num))
    return df_spaced



fs = filestruct.fs()

########################################################################
########################################################################
########################################################################
#Analysis topology defintion
generator_type = "norad"
generator_type = "rad"
mag_config = "inbending"
mag_config = "outbending"

########################################################################
########################################################################
########################################################################
#Exclusivity cuts


########################################################################
########################################################################
########################################################################
#Analysis step definitions
# convert_roots = False
# make_exclusive_cuts = True
# plot_initial_distros = False
# plot_final_distros = True
# bin_all_events = True
# bin_gen = True
# calc_xsection = True
# plot_reduced_xsec_and_fit = True
convert_roots = False
make_exclusive_cuts = True
plot_initial_distros = True
plot_final_distros = True
bin_all_events = True
bin_gen = True
calc_xsection = True
plot_reduced_xsec_and_fit = True

########################################################################
########################################################################
########################################################################
# Directory definitions
unique_identifyer = "with_gp_cut"
run_identifiyer = mag_config+"_"+generator_type+"_"+unique_identifyer

datafile_base_dir = "/mnt/d/GLOBUS/CLAS12/APS2022/"
roots_dir = "raw_roots/"
raw_data_dir = "pickled_data/"

dvpip_data_dir = "pickled_dvpip/"
binned_data_dir = "binned_dvpip/"
final_xsec_dir = "final_data_files/"

initial_distros_dir = "initial_distros/"
final_distros_dir = "final_distros/"

initial_exp_plots_dir = "initial_exp_plots_"+mag_config+"/"
initial_rec_plots_dir = "initial_rec_plots_"+run_identifiyer+"/"
initial_rec_exp_plots_dir = "initial_rec_exp_plots_"+run_identifiyer+"/"
initial_gen_plots_dir = "initial_gen_plots_"+run_identifiyer+"/"

final_exp_plots_dir = "final_exp_plots_"+mag_config+"/"
final_rec_plots_dir = "final_rec_plots_"+run_identifiyer+"/"
final_rec_exp_plots_dir = "final_rec_exp_plots_"+run_identifiyer+"/"
final_gen_plots_dir = "final_gen_plots_"+run_identifiyer+"/"
reduced_xsection_plots_dir = "reduced_xsection_plots/"


merged_data_name = "merged_total_"+run_identifiyer
final_output_name = "full_xsection_"+run_identifiyer

exp_common_name = "fall 2018 {} exp {}".format(mag_config,unique_identifyer)
rec_common_name = "sim rec {} {} {}".format(mag_config,generator_type,unique_identifyer)


if plot_reduced_xsec_and_fit:
    def resid(pars):
        return ((y-fit_function(x,pars))**2).sum()

    def resid_weighted(pars):
        return (((y-fit_function(x,pars))**2)/sigma).sum()

    def fit_function(phi,A,B,C):
        #A + B*np.cos(2*phi) +C*np.cos(phi)
        rads = phi*np.pi/180
        #return (A * np.exp(-x/beta) + B * np.exp(-1.0 * (x - mu)**2 / (2 * sigma**2)))
        #A = T+L, B=TT, C=LT
        #A = black, B=blue, C=red
        return A + B*np.cos(2*rads) + C*np.cos(rads)

    def bin_clas6_sf(df):
        q2bins,xBbins, tbins, phibins = fs.q2bins, fs.xBbins, fs.tbins, fs.phibins

        qrange = [q2bins[0], q2bins[-1]]
        xBrange = [xBbins[0], xBbins[-1]]
        trange = [tbins[0], tbins[-1]]

        data_vals = []

        for qmin,qmax in zip(q2bins[0:-1],q2bins[1:]):
            print(" \n Q2 bin: {} to {}".format(qmin,qmax))
            query = "q2 >{} and q2 < {}".format(qmin,qmax)
            df_q = df.query(query)

            for xmin,xmax in zip(xBbins[0:-1],xBbins[1:]):
                #print("        xB bin: {} to {}".format(xmin,xmax))
                query = "xb>{} and xb<{}".format(xmin,xmax)
                df_qx = df_q.query(query)

                for tmin,tmax in zip(tbins[0:-1],tbins[1:]):
                    #print("                 t bin: {} to {}".format(tmin,tmax))
                    query = "t>{} and t<{}".format(tmin,tmax)
                    df_qxt = df_qx.query(query)

                    if df_qxt.empty:
                        data_vals.append([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,qmin,xmin,tmin,qmax,xmax,tmax])
                    else:                    
                        list_value = [df_qxt["q2"].values[0],df_qxt["xb"].values[0],df_qxt["t"].values[0],df_qxt["tel"].values[0],df_qxt["tel-stat"].values[0],df_qxt["tel-sys"].values[0],df_qxt["lt"].values[0],df_qxt["lt-stat"].values[0],df_qxt["lt-sys"].values[0],df_qxt["tt"].values[0],df_qxt["tt-stat"].values[0],df_qxt["tt-sys"].values[0],qmin,xmin,tmin,qmax,xmax,tmax]
                        print(list_value)
                        data_vals.append(list_value)


        df_spaced = pd.DataFrame(data_vals, columns = ['q','x','t','tel','tel-stat','tel-sys','lt','lt-stat','lt-sys','tt','tt-stat','tt-sys','qmin','xmin','tmin','qmax','xmax','tmax'])
        # df_minibin = pd.DataFrame(num_counts, columns = ['qmin','xmin','tmin','pmin','qave','xave','tave','pave',prefix+'counts'])
        # print("Total number of binned events: {}".format(df_minibin[prefix+'counts'].sum()))
        # print("Total number of original events: {}".format(total_num))
        return df_spaced

    xmax = 360
    xspace = np.linspace(0, xmax, 1000)

    # # df_inbending = pd.read_pickle(datafile_base_dir + final_xsec_dir+"full_xsection_inbending_rad_all.pkl")
    # # df_outbending = pd.read_pickle(datafile_base_dir + final_xsec_dir+"full_xsection_outbending_rad_all.pkl")


    df_inbending = pd.read_pickle(datafile_base_dir + final_xsec_dir+"full_xsection_inbending_rad_All_All_All_low_and_hi_Gp_cuts.pkl")
    df_outbending = pd.read_pickle(datafile_base_dir + final_xsec_dir+"full_xsection_outbending_rad_All_All_All_low_and_hi_Gp_cuts.pkl")

    # df_inbending = pd.read_pickle(datafile_base_dir + final_xsec_dir+"full_xsection_inbending_rad_All_All_All_no_Gp_cuts.pkl")
    # df_outbending = pd.read_pickle(datafile_base_dir + final_xsec_dir+"full_xsection_outbending_rad_All_All_All_no_Gp_cuts.pkl")


    df_sf_binned = pd.read_pickle('final_data_files/clas6_structure_funcs_binned.pkl')
    df_sf_binned = df_sf_binned.apply(pd.to_numeric)


    #inbending

    df_inbending.loc[:,"xsec_corr_nb_gamma"] = df_inbending["xsec_corr"]*1E33/df_inbending["gamma_exp"]

    df_inbending.loc[:,"tot_clas6_uncert"] = np.sqrt(np.square(df_inbending["stat"]/df_inbending["dsdtdp"]) + np.square(df_inbending["sys"]/df_inbending["dsdtdp"]))*df_inbending["dsdtdp"]

    df_inbending.loc[:,"epsi_clas6"] = get_gamma(df_inbending["x"],df_inbending["q"],5.776)[1] #get_gamma((df_inbendingout["xmin"]+df_inbendingout["xmax"])/2,(df_inbendingout["qmin"]+df_inbendingout["qmax"])/2)[1]

    df_inbending.loc[:,"uncert_xsec_corr_nb_gamma"] = np.sqrt(  np.square(df_inbending["uncert_xsec"]/df_inbending["xsec"]) + np.square(df_inbending["uncert_acc_corr"]/df_inbending["acc_corr"]))*df_inbending["xsec_corr_nb_gamma"]
    df_inbending.loc[:,"c_12_uncert_ratio"] = df_inbending['uncert_xsec_corr_nb_gamma']/df_inbending['xsec_corr_nb_gamma']

    df_inbending.loc[(df_inbending.acc_corr < 0.01),'xsec_corr_nb_gamma']=np.nan


    #outbending

    df_outbending.loc[:,"xsec_corr_nb_gamma"] = df_outbending["xsec_corr"]*1E33/df_outbending["gamma_exp"]

    df_outbending.loc[:,"tot_clas6_uncert"] = np.sqrt(np.square(df_outbending["stat"]/df_outbending["dsdtdp"]) + np.square(df_outbending["sys"]/df_outbending["dsdtdp"]))*df_outbending["dsdtdp"]

    df_outbending.loc[:,"epsi_clas6"] = get_gamma(df_outbending["x"],df_outbending["q"],5.776)[1] #get_gamma((df_outbendingout["xmin"]+df_outbendingout["xmax"])/2,(df_outbendingout["qmin"]+df_outbendingout["qmax"])/2)[1]

    df_outbending.loc[:,"uncert_xsec_corr_nb_gamma"] = np.sqrt(  np.square(df_outbending["uncert_xsec"]/df_outbending["xsec"]) + np.square(df_outbending["uncert_acc_corr"]/df_outbending["acc_corr"]))*df_outbending["xsec_corr_nb_gamma"]
    df_outbending.loc[:,"c_12_uncert_ratio"] = df_outbending['uncert_xsec_corr_nb_gamma']/df_outbending['xsec_corr_nb_gamma']

    df_outbending.loc[(df_outbending.acc_corr < 0.01),'xsec_corr_nb_gamma']=np.nan





    q2bins,xBbins, tbins, phibins = fs.q2bins, fs.xBbins, fs.tbins, fs.phibins


    qrange = [q2bins[0], q2bins[-1]]
    xBrange = [xBbins[0], xBbins[-1]]
    trange = [tbins[0], tbins[-1]]


    sf_data_vals = []

    reduced_plot_dir = datafile_base_dir+reduced_xsection_plots_dir+run_identifiyer+ "in_out_comp_/"
    if not os.path.exists(reduced_plot_dir):
        os.makedirs(reduced_plot_dir)

    for qmin,qmax in zip(q2bins[0:-1],q2bins[1:]):
        print(" \n Q2 bin: {} to {}".format(qmin,qmax))
        for xmin,xmax in zip(xBbins[0:-1],xBbins[1:]):
            for tmin,tmax in zip(tbins[0:-1],tbins[1:]):

                query = "qmin == {} and xmin == {} and tmin == {}".format(qmin,xmin,tmin)

                df_inbending_small = df_inbending.query(query)
                df_outbending_small = df_outbending.query(query)

                df_sf_binned_small = df_sf_binned.query(query)

                df_inbending_check = df_inbending_small[df_inbending_small["xsec_corr_nb_gamma"].notnull()]
                df_outbending_check = df_outbending_small[df_outbending_small["xsec_corr_nb_gamma"].notnull()]


                if np.isnan(df_sf_binned_small["tel"].values[0]) or (df_inbending_check.empty or df_outbending_check.empty) or (df_inbending_small[df_inbending_small["xsec_corr_nb_gamma"].notnull()].shape[0]<3 or df_outbending_small[df_outbending_small["xsec_corr_nb_gamma"].notnull()].shape[0]<3):
                    pass
                else:
                    epsi_mean_c6 = df_inbending_small["epsi_clas6"].mean()

                    epsi_mean_c12 = df_inbending_small["epsi_exp"].mean()
                    epsi_mean_c12_out = df_outbending_small["epsi_exp"].mean()

                    mean_xsec_uncer_ratio_c12 = df_inbending_small['c_12_uncert_ratio'].mean()
                    mean_xsec_uncer_ratio_c12_out = df_outbending_small['c_12_uncert_ratio'].mean()



                    binscenters_c12 = df_inbending_small["pave_exp"]
                    data_entries_c12 = df_inbending_small["xsec_corr_nb_gamma"]
                    sigma_c12 = df_inbending_small["uncert_xsec_corr_nb_gamma"]

                    binscenters_c12_out = df_outbending_small["pave_exp"]
                    data_entries_c12_out = df_outbending_small["xsec_corr_nb_gamma"]
                    sigma_c12_out = df_outbending_small["uncert_xsec_corr_nb_gamma"]

                    binscenters = df_inbending_small["p"]
                    data_entries = df_inbending_small["dsdtdp"]
                    sigma = df_inbending_small["tot_clas6_uncert"]

                    x = binscenters
                    y = data_entries

                    def resid_weighted_c12(pars):
                        return (((y-fit_function(x,pars))**2)/sigma_c12).sum()

                    def resid_weighted_c12_out(pars):
                        return (((y-fit_function(x,pars))**2)/sigma_c12_out).sum()

                    def constr0(pars):
                        return fit_function(0,pars)
                    
                    def constr180(pars):
                        return fit_function(180,pars)

                    con1 = {'type': 'ineq', 'fun': constr0}
                    con2 = {'type': 'ineq', 'fun': constr180}
                    # con3 = {'type': 'ineq', 'fun': constr270}
                    cons = [con1,con2]

                    x = binscenters_c12
                    y = data_entries_c12
                    valid = ~(np.isnan(x) | np.isnan(y))

                    popt_0, pcov = curve_fit(fit_function, xdata=x[valid], ydata=y[valid], p0=[100,-60,-11],
                        sigma=sigma_c12[valid], absolute_sigma=True)

                    popt, pcov = curve_fit(fit_function, xdata=x[valid], ydata=y[valid], p0=[popt_0[0],popt_0[1],popt_0[2]],
                                sigma=sigma_c12[valid], absolute_sigma=True)

                    a,b,c = popt[0],popt[1],popt[2]
                    
                    a_err = np.sqrt(pcov[0][0])#*qmod
                    b_err = np.sqrt(pcov[1][1])#*qmod
                    c_err = np.sqrt(pcov[2][2])#*qmod

                    x = binscenters_c12_out
                    y = data_entries_c12_out
                    valid = ~(np.isnan(x) | np.isnan(y))

                    popt_0, pcov = curve_fit(fit_function, xdata=x[valid], ydata=y[valid], p0=[100,-60,-11],
                        sigma=sigma_c12_out[valid], absolute_sigma=True)

                    popt, pcov = curve_fit(fit_function, xdata=x[valid], ydata=y[valid], p0=[popt_0[0],popt_0[1],popt_0[2]],
                                sigma=sigma_c12_out[valid], absolute_sigma=True)

                    a0,b0,c0 = popt[0],popt[1],popt[2]
                    
                    a0_err = np.sqrt(pcov[0][0])#*qmod
                    b0_err = np.sqrt(pcov[1][1])#*qmod
                    c0_err = np.sqrt(pcov[2][2])#*qmod



                    ###A +    Bcos(2x) + Ccos(x)
                    ###TEL +   ep*TT   + sqr*LT
                    
                    pub_tel =  df_sf_binned_small['tel'].values[0]
                    pub_tt =  df_sf_binned_small['tt'].values[0]
                    pub_lt =  df_sf_binned_small['lt'].values[0]

                    rev_a =pub_tel/6.28
                    rev_b = pub_tt/6.28*epsi_mean_c6
                    rev_c = pub_lt/6.28*np.sqrt(2*epsi_mean_c6*(1+epsi_mean_c6))

                    fit_y_data_weighted = fit_function(xspace,rev_a,rev_b,rev_c)

                    a_c12,b_c12,c_c12 = a,b,c 
                    a_c12_out,b_c12_out,c_c12_out = a0,b0,c0 


                    tel_c12 = a_c12*6.28
                    tt_c12 = b_c12/epsi_mean_c12*6.28
                    lt_c12 = c_c12/np.sqrt(2*epsi_mean_c12*(1+epsi_mean_c12))*6.28

                    tel_c12_out = a_c12_out*6.28
                    tt_c12_out = b_c12_out/epsi_mean_c12_out*6.28
                    lt_c12_out = c_c12_out/np.sqrt(2*epsi_mean_c12_out*(1+epsi_mean_c12_out))*6.28

                    tel_c12_err = tel_c12*a_err/a
                    tt_c12_err = tt_c12*b_err/b
                    lt_c12_err = lt_c12*c_err/c

                    tel_c12_out_err = tel_c12_out*a0_err/a0
                    tt_c12_out_err = tt_c12_out*b0_err/b0
                    lt_c12_out_err = lt_c12_out*c0_err/c0

                    fit_y_data_weighted_new_c12 = fit_function(xspace, a_c12,b_c12,c_c12)
                    fit_y_data_weighted_new_c12_out = fit_function(xspace, a_c12_out,b_c12_out,c_c12_out)



                    q_mean_c12 = df_inbending_small['qave_exp'].mean()
                    x_mean_c12 = df_inbending_small['xave_exp'].mean()
                    t_mean_c12 = df_inbending_small['tave_exp'].mean()

                    sf_data_vals.append([df_sf_binned_small['q'].values[0],
                        df_sf_binned_small['x'].values[0],
                        df_sf_binned_small['t'].values[0],
                        df_sf_binned_small['tel'].values[0],
                        df_sf_binned_small['tel-stat'].values[0],
                        df_sf_binned_small['tel-sys'].values[0],
                        df_sf_binned_small['tt'].values[0],
                        df_sf_binned_small['tt-stat'].values[0],
                        df_sf_binned_small['tt-sys'].values[0],
                        df_sf_binned_small['lt'].values[0],
                        df_sf_binned_small['lt-stat'].values[0],
                        df_sf_binned_small['lt-sys'].values[0],
                        q_mean_c12,x_mean_c12,t_mean_c12,                 
                        tel_c12,tt_c12,lt_c12,
                        mean_xsec_uncer_ratio_c12,
                        qmin,xmin,qmax,xmax,
                        tel_c12_err,tt_c12_err,lt_c12_err,])

                    plt.rcParams["font.size"] = "20"

                    fig, ax = plt.subplots(figsize =(14, 10)) 

                    plt.errorbar(binscenters, data_entries, yerr=sigma, color="red",fmt="o")#,label="CLAS6 Data")

                    plt.errorbar(binscenters_c12, data_entries_c12, yerr=sigma_c12, color="blue",fmt="x")#,label="CLAS12 Data in.")
                    plt.errorbar(binscenters_c12_out, data_entries_c12_out, yerr=sigma_c12_out, color="green",fmt="x")#,label="CLAS12 Data out.")


                    plt.rcParams["font.size"] = "20"

                    fit2, = ax.plot(xspace, fit_y_data_weighted, color='red', linewidth=2.5, label='CLAS6 Fit:         t+l:{:.0f} tt:{:.0f} lt:{:.0f}'.format(pub_tel,pub_tt,pub_lt))
                    fit4, = ax.plot(xspace, fit_y_data_weighted_new_c12, color='blue', linewidth=2.5, label='CLAS12 in.: t+l:{:.0f} tt:{:.0f} lt:{:.0f}'.format(tel_c12,tt_c12,lt_c12))
                    fit6, = ax.plot(xspace, fit_y_data_weighted_new_c12_out, color='green', linewidth=2.5, label='CLAS12 out.: t+l:{:.0f} tt:{:.0f} lt:{:.0f}'.format(tel_c12_out,tt_c12_out,lt_c12_out))
                    
                    ax.legend(loc="best")
                    ax.set_xlabel("Phi")  
                    ax.set_ylabel('Reduced Cross Section (nb/GeV$^2$)')  
                    title = "Reduced Cross Section Fit Over Phi, Q$^2$ = {:.2f}, x$_B$ = {:.2f}, t = {:.1f}".format(df_sf_binned_small['q'].values[0],df_sf_binned_small['x'].values[0],df_sf_binned_small['t'].values[0])
                    plt.title(title)


                    plt.savefig(reduced_plot_dir+title.replace("$","").replace(".","").replace("^","").replace(" ","").replace("=","").replace(",","_")+".png")

                    plt.close()

    df_out = pd.DataFrame(sf_data_vals, columns = ['qC6','xC6','tC6','telC6','tel-statC6','tel-sysC6','ltC6','lt-statC6','lt-sysC6','ttC6','tt-statC6',
                                    'tt-sysC6','qC12','xC12','tC12','tel_c12','tt_c12','lt_c12','mean_uncert_c12','qmin','xmin',
                                    'qmax','xmax','tel_c12_err','tt_c12_err','lt_c12_err'])


    df_out.to_pickle(datafile_base_dir + final_xsec_dir+"struct_funcs_combined.pkl")


