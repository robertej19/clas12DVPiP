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
from convert_root_to_pickle import new_convert_real_to_pkl
from convert_root_to_pickle import new_convert_rec_to_pkl


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
#from make_dvpip_cuts import makeDVpi0
#from exclusivity_cuts.sangbaek_exclusivity_cuts import makeDVpi0P
from exclusivity_cuts.new_exclusivity_cuts import makeDVpi0P
from exclusivity_cuts.new_exclusivity_cuts import calc_ex_cut_mu_sigma


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
convert_roots = 0
make_exclusive_cuts = 1
plot_initial_distros = 0
plot_final_distros = 1
bin_all_events = 0
bin_gen = 0
calc_xsection = 0
plot_reduced_xsec_and_fit = 0
plot_1_D_hists = 0




########################################################################
########################################################################
########################################################################



def run_analysis(mag_config,generator_type,unique_identifyer="",
                det_proton_loc="All",det_photon1_loc="All",det_photon2_loc="All",
                convert_roots = 0,
                make_exclusive_cuts = 1,
                plot_initial_distros = 0,
                plot_final_distros = 1,
                bin_all_events = 0,
                bin_gen = 0,
                calc_xsection = 0,
                plot_reduced_xsec_and_fit = 0,
                plot_1_D_hists = 1,
                calc_xsection_c12_only = 0,
                emergency_stop = 0,
                qxt_cuts = [[0,100],[0,1],[0,100]],
                simple_exclusivity_cuts=False,
                plot_reduced_xsec_and_fit_c12_only=0,
                comp_2_config=False,
                gen_ex_cut_table=False,
                sigma_multiplier=3):


    run_identifiyer = mag_config+"_"+generator_type+"_"+det_proton_loc+"_"+det_photon1_loc+"_"+det_photon2_loc+"_"+unique_identifyer+"excut_sigma_{}".format(sigma_multiplier)
    inb_run_id = "inbending"+"_"+generator_type+"_"+det_proton_loc+"_"+det_photon1_loc+"_"+det_photon2_loc+"_"+unique_identifyer
    outb_run_id = "outbending"+"_"+generator_type+"_"+det_proton_loc+"_"+det_photon1_loc+"_"+det_photon2_loc+"_"+unique_identifyer

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

    final_exp_plots_dir = "final_exp_plots_"+run_identifiyer+"/"
    final_rec_plots_dir = "final_rec_plots_"+run_identifiyer+"/"
    final_rec_exp_plots_dir = "final_rec_exp_plots_"+run_identifiyer+"/"
    final_gen_plots_dir = "final_gen_plots_"+run_identifiyer+"/"
    reduced_xsection_plots_dir = "reduced_xsection_plots/"


    merged_data_name = "merged_total_"+run_identifiyer
    final_output_name = "full_xsection_"+run_identifiyer

    exp_common_name = "fall 2018 {} exp {}".format(mag_config,unique_identifyer)
    rec_common_name = "sim rec {} {} {}".format(mag_config,generator_type,unique_identifyer)

    df_final_config_1 = datafile_base_dir+final_xsec_dir + "full_xsection_"+outb_run_id+".pkl"
    df_final_config_2  = datafile_base_dir+final_xsec_dir + "full_xsection_"+inb_run_id+".pkl"

    if not convert_roots:
        convert_root_exp = False
        convert_root_rec = False
        convert_root_gen = False
        calc_lumi = False
    else:
        convert_root_exp = True
        convert_root_rec = True
        convert_root_gen = True
        calc_lumi = False

    if not plot_initial_distros:
        plot_initial_exp_distros = False
        plot_initial_rec_distros = False
        plot_initial_rec_exp_distros = False
        plot_initial_gen_distros = False
    else:
        plot_initial_exp_distros = True
        plot_initial_rec_distros = True
        plot_initial_rec_exp_distros = True
        plot_initial_gen_distros = True

    if not make_exclusive_cuts:
        make_ex_cut_exp = False
        make_ex_cut_gen = False
        make_ex_cut_rec = False
    else:
        make_ex_cut_exp = True
        make_ex_cut_gen = True
        make_ex_cut_rec = True


    if not plot_final_distros:
        plot_final_exp_distros = False
        plot_final_rec_distros = False
        plot_final_rec_exp_distros = False
        plot_final_gen_distros = False
    else:
        plot_final_exp_distros = 1
        plot_final_rec_distros = 1
        plot_final_rec_exp_distros = 1
        plot_final_gen_distros = 1

    ########################################################################
    ########################################################################
    ########################################################################
    #Choose functions from config parameters
    if generator_type == "rad":
        generator_type_keyword = "pi0rad"
        gen_converter = convert_GEN_RAD_root_to_pkl
        if mag_config == "inbending":
            path_to_exp_root = fs.path_to_exp_inbending_root
            path_to_rec_root = fs.path_to_rec_inbending_rad_root
            path_to_gen_root = fs.path_to_gen_inbending_rad_root
        elif mag_config == "outbending":
            path_to_exp_root = fs.path_to_exp_outbending_root
            path_to_rec_root = fs.path_to_rec_outbending_rad_root
            path_to_gen_root = fs.path_to_gen_outbending_rad_root
    elif generator_type == "norad":
        generator_type_keyword = "pi0norad"
        gen_converter = convert_GEN_NORAD_root_to_pkl
        if mag_config == "inbending":
            path_to_exp_root = fs.path_to_exp_inbending_root
            path_to_rec_root = fs.path_to_rec_inbending_norad_root
            path_to_gen_root = fs.path_to_gen_inbending_norad_root
        elif mag_config == "outbending":
            path_to_exp_root = fs.path_to_exp_outbending_root
            path_to_rec_root = fs.path_to_rec_outbending_norad_root
            path_to_gen_root = fs.path_to_gen_outbending_norad_root

    exp_file_base = os.listdir(datafile_base_dir+roots_dir+path_to_exp_root)[0].split(".")[0]
    rec_file_base = os.listdir(datafile_base_dir+roots_dir+path_to_rec_root)[0].split(".")[0]
    gen_file_base = os.listdir(datafile_base_dir+roots_dir+path_to_gen_root)[0].split(".")[0]
    Clas12_exp_luminosity = 5.5e+40 if (mag_config == "inbending") else 4.651647453735352e+40
                            # new number: 5.039797538543702e+40
                 
    ########################################################################
    ########################################################################
    ########################################################################
    #Convert root to pkl
    if convert_roots:

        if calc_lumi:
            converter_exp = new_convert_real_to_pkl.root2pickle(
            datafile_base_dir+roots_dir+path_to_exp_root+exp_file_base+".root",
            pol=mag_config,
            logistics=True)
            df_exp_with_logi  = converter_exp.df_epgg
            df_exp_with_logi.to_pickle(datafile_base_dir+raw_data_dir+exp_file_base+"_with_logi"+".pkl")
            Clas12_exp_luminosity = get_integrated_lumi.get_integrated_lumi(df_exp_with_logi,bad_runs_list=[])[0]
            print(Clas12_exp_luminosity)

        if convert_root_exp:
            converter_exp = new_convert_real_to_pkl.root2pickle(
                datafile_base_dir+roots_dir+path_to_exp_root+exp_file_base+".root",
                pol=mag_config,
                logistics=False)

            df_exp  = converter_exp.df_epgg
            df_exp.to_pickle(datafile_base_dir+raw_data_dir+exp_file_base+".pkl")

        if convert_root_rec:
            converter_rec = new_convert_rec_to_pkl.root2pickle(
                datafile_base_dir+roots_dir+path_to_rec_root+rec_file_base+".root",
                pol=mag_config,gen=generator_type_keyword)

            df_rec = converter_rec.df
            df_rec.to_pickle(datafile_base_dir+raw_data_dir+rec_file_base+".pkl")

        if convert_root_gen:
            print("CONVERING ROOT GEN")
            df_gen = gen_converter.readEPGG(
                datafile_base_dir+roots_dir+path_to_gen_root+gen_file_base+".root")
            print("SAVING GEN ROOT FILE TO {}".format(datafile_base_dir+raw_data_dir+gen_file_base+".pkl"))
            
            df_gen.to_pickle(datafile_base_dir+raw_data_dir+gen_file_base+".pkl")




    if make_exclusive_cuts:
        #### APPLY EXCLUSIVITY CUTS
        print("Applying exclusive cuts to dataframe...")

        ########################################
        if make_ex_cut_exp:
            try:
                df_exp
            except NameError:
                df_exp = pd.read_pickle(datafile_base_dir+raw_data_dir+exp_file_base+".pkl")

            df_exp_epgg = df_exp

            df_exp_epgg = df_exp_epgg.query("Q2 > {} and Q2 < {} and xB > {} and xB < {} and t1 > {} and t1 < {}".format(qxt_cuts[0][0],qxt_cuts[0][1],qxt_cuts[1][0],qxt_cuts[1][1],qxt_cuts[2][0],qxt_cuts[2][1]))

            df_exp_epgg.to_pickle(datafile_base_dir+raw_data_dir+exp_file_base+"_ex_cut"+".pkl")

            df_exp_epgg.loc[:,'y'] = df_exp_epgg['nu']/10.604
            
            if plot_initial_exp_distros:
                title_dir = datafile_base_dir+initial_distros_dir+initial_exp_plots_dir
                histo_plotting.make_all_histos(df_exp_epgg,datatype="exp",hists_2d=True,hists_1d=plot_1_D_hists,
                        first_label='exp',hists_overlap=False,saveplots=True,output_dir = title_dir)

            print("There are {} exp epgg events".format(df_exp_epgg.shape[0]))

            datafilename=datafile_base_dir+raw_data_dir+exp_file_base+"_table_of_ex_cut_"


        ########################################
        #WORK HERE!!!
            if gen_ex_cut_table:
                print("CALCULATING EXCLUSIVE CUT TABLE for {}".format(datafile_base_dir+raw_data_dir+exp_file_base))
                calc_ex_cut_mu_sigma(df_exp_epgg,datafilename=datafile_base_dir+raw_data_dir+exp_file_base+"_table_of_ex_cut_",unique_identifyer=unique_identifyer)
            #df_ex_cut_ranges = pd.read_pickle(datafile_base_dir+raw_data_dir+exp_file_base+"_table_of_ex_cut"+".pkl",unique_identifyer=unique_identifyer)
            
            print("CALCULATING EXCLUSIVE CUTS exp for {} with sigma {}".format(datafile_base_dir+raw_data_dir+exp_file_base,sigma_multiplier))

            df_dvpip_exp = makeDVpi0P(df_exp_epgg, sigma_multiplier,datafilename=datafilename, unique_identifyer = unique_identifyer)
            
            #df_dvpip_exp = pd.read_pickle("{}/{}_dvpip_exp.pkl".format(datafile_base_dir+dvpip_data_dir,exp_file_base))
            df_dvpip_exp.to_pickle("{}/{}_dvpip_exp.pkl".format(datafile_base_dir+dvpip_data_dir,exp_file_base))
            print("There are {} exp dvpip events".format(df_dvpip_exp.shape[0]))

            df_dvpip_exp = df_dvpip_exp.query("Q2 > {} and Q2 < {} and xB > {} and xB < {} and t1 > {} and t1 < {}".format(qxt_cuts[0][0],qxt_cuts[0][1],qxt_cuts[1][0],qxt_cuts[1][1],qxt_cuts[2][0],qxt_cuts[2][1]))

            if plot_final_exp_distros:
                title_dir = datafile_base_dir+final_distros_dir+final_exp_plots_dir
                print("plotting final exp distros at {}".format(title_dir))
                histo_plotting.make_all_histos(df_dvpip_exp,datatype="exp",hists_2d=True,hists_1d=plot_1_D_hists,
                        first_label='exp',hists_overlap=False,saveplots=True,output_dir = title_dir)

        ########################################
        if make_ex_cut_rec:
            try:
                df_rec
            except NameError:
                df_rec = pd.read_pickle(datafile_base_dir+raw_data_dir+rec_file_base+".pkl")

            df_rec_epgg = df_rec
            df_rec_epgg = df_rec_epgg.query("Q2 > {} and Q2 < {} and xB > {} and xB < {} and t1 > {} and t1 < {}".format(qxt_cuts[0][0],qxt_cuts[0][1],qxt_cuts[1][0],qxt_cuts[1][1],qxt_cuts[2][0],qxt_cuts[2][1]))

            df_rec_epgg.loc[:,'y'] = df_rec_epgg['nu']/10.604

            
            if plot_initial_rec_distros:
                title_dir = datafile_base_dir+initial_distros_dir+initial_rec_plots_dir
                histo_plotting.make_all_histos(df_rec_epgg,datatype="rec",hists_2d=True,hists_1d=plot_1_D_hists,
                        first_label='rec',hists_overlap=False,saveplots=True,output_dir = title_dir)

            print("There are {} rec epgg events".format(df_rec_epgg.shape[0]))

            print("CALCULATING EXCLUSIVE CUTS rec for {} with sigma {}".format(datafile_base_dir+raw_data_dir+exp_file_base,sigma_multiplier))

            df_dvpip_rec = makeDVpi0P(df_rec_epgg, sigma_multiplier,datafilename=datafilename,unique_identifyer=unique_identifyer)

            #df_dvpip_rec = makeDVpi0P(df_rec_epgg,data_type="rec",proton_loc=det_proton_loc,photon1_loc=det_photon1_loc,photon2_loc=det_photon2_loc,pol = mag_config,simple_exclusivity_cuts=simple_exclusivity_cuts)
            #df_dvpip_rec = pd.read_pickle("{}/{}_dvpip_rec.pkl".format(datafile_base_dir+dvpip_data_dir,rec_file_base))
            df_dvpip_rec.to_pickle("{}/{}_dvpip_rec.pkl".format(datafile_base_dir+dvpip_data_dir,rec_file_base))

            print("There are {} rec dvpip events".format(df_dvpip_rec.shape[0]))

            if plot_final_rec_distros:
                title_dir = datafile_base_dir+final_distros_dir+final_rec_plots_dir
                histo_plotting.make_all_histos(df_dvpip_rec,datatype="rec",hists_2d=True,hists_1d=plot_1_D_hists,
                        first_label='rec',hists_overlap=False,saveplots=True,output_dir = title_dir)


        if plot_initial_rec_exp_distros:
            title_dir = datafile_base_dir+initial_distros_dir+initial_rec_exp_plots_dir
            histo_plotting.make_all_histos(df_exp_epgg,datatype="exp",hists_2d=False,hists_1d=False,hists_overlap=True,saveplots=True,output_dir = title_dir,
                                            df_2=df_rec_epgg,first_label=exp_common_name,second_label=rec_common_name)
        if plot_final_rec_exp_distros:
            title_dir = datafile_base_dir+final_distros_dir+final_rec_exp_plots_dir
            # histo_plotting.make_all_histos(df_dvpip_exp,datatype="rec",hists_2d=False,hists_1d=False,hists_overlap=True,saveplots=True,output_dir = title_dir,
            #                                 df_2=df_dvpip_rec,first_label=exp_common_name,second_label=rec_common_name,plot_title_identifiyer=run_identifiyer)
            histo_plotting.make_all_histos(df_dvpip_exp,datatype="rec",hists_2d=False,hists_1d=False,hists_overlap=True,saveplots=True,output_dir = title_dir,
                                            df_2=df_dvpip_rec,first_label="exp",second_label="sim",plot_title_identifiyer=run_identifiyer)

        ########################################
        if make_ex_cut_gen:
            try:
                df_gen
            except NameError:
                df_gen = pd.read_pickle(datafile_base_dir+raw_data_dir+gen_file_base+".pkl")

            df_gen_epgg = df_gen

            
            if plot_initial_gen_distros:
                title_dir = datafile_base_dir+initial_distros_dir+initial_gen_plots_dir
                histo_plotting.make_all_histos(df_gen_epgg,datatype="Gen",hists_2d=True,hists_1d=plot_1_D_hists,
                        first_label='gen',hists_overlap=False,saveplots=True,output_dir = title_dir)

            print("There are {} gen epgg events".format(df_gen_epgg.shape[0]))
            df_dvpip_gen = df_gen_epgg.query("GenQ2>1 and GenW>2")
            df_dvpip_gen.to_pickle("{}/{}_dvpip_gen.pkl".format(datafile_base_dir+dvpip_data_dir,gen_file_base))

            print("There are {} gen dvpip events".format(df_dvpip_gen.shape[0]))

            if plot_final_gen_distros:
                title_dir = datafile_base_dir+final_distros_dir+final_gen_plots_dir
                histo_plotting.make_all_histos(df_dvpip_gen,datatype="Gen",hists_2d=True,hists_1d=plot_1_D_hists,
                        first_label='gen',hists_overlap=False,saveplots=True,output_dir = title_dir)

    if emergency_stop:
        print("Emergency stop, exiting function")
        return

    #### BIN EVENTS
    if bin_all_events:
        print("Binning events...")

        if bin_gen:
            df_dvpip_gen = pd.read_pickle("{}/{}_dvpip_gen.pkl".format(datafile_base_dir+dvpip_data_dir,gen_file_base))

            print("BINNING GENERATED EVENTS")
            df_gen_binned = bin_df(df_dvpip_gen, "Gen")
            df_gen_binned.to_pickle(datafile_base_dir+binned_data_dir + gen_file_base+"_binned"+".pkl")
            df_gen = df_gen_binned
        else:
            df_gen = pd.read_pickle(datafile_base_dir+binned_data_dir + gen_file_base+"_binned"+".pkl")

        df_dvpip_exp = pd.read_pickle("{}/{}_dvpip_exp.pkl".format(datafile_base_dir+dvpip_data_dir,exp_file_base))
        df_dvpip_rec = pd.read_pickle("{}/{}_dvpip_rec.pkl".format(datafile_base_dir+dvpip_data_dir,rec_file_base))

        #df_dvpip_exp = pd.read_pickle("new_exp_dvpi0p_outbend.pkl")
        #df_dvpip_rec = pd.read_pickle("new_rec_dvpi0p_outbend_rad.pkl")


        df_exp_binned = bin_df(df_dvpip_exp, "exp")
        df_rec_binned = bin_df(df_dvpip_rec, "rec")


        df_exp_binned.to_pickle(datafile_base_dir+binned_data_dir+exp_file_base+run_identifiyer+"_binned"+".pkl")
        df_rec_binned.to_pickle(datafile_base_dir+binned_data_dir+rec_file_base+run_identifiyer+"_binned"+".pkl")

        df_exp = df_exp_binned
        df_rec = df_rec_binned
        df_exp.to_csv("test_df_exp.csv")
        
                                

    if calc_xsection:
        if not bin_all_events:
            df_gen = pd.read_pickle(datafile_base_dir+binned_data_dir + gen_file_base+"_binned"+".pkl")
            df_exp = pd.read_pickle(datafile_base_dir+binned_data_dir+exp_file_base+run_identifiyer+"_binned"+".pkl")
            df_rec = pd.read_pickle(datafile_base_dir+binned_data_dir+rec_file_base+run_identifiyer+"_binned"+".pkl")

        space_clas6 = False
        if space_clas6:
            df_clas6 = pd.read_pickle(binned_data_dir + "xs_clas6_binned.pkl")
            df_clas6 = expand_clas6(df_clas6)
            df_clas6.to_pickle(base_dir + "xs_clas6_binned_expanded.pkl")
        else:
            df_clas6 = pd.read_pickle(binned_data_dir + "xs_clas6_binned_expanded.pkl")



        df_exp = df_exp.rename(columns={"qave": "qave_exp", "xave": "xave_exp","tave": "tave_exp", "pave": "pave_exp","counts":"counts_exp"})
        df_rec = df_rec.rename(columns={"qave": "qave_rec", "xave": "xave_rec","tave": "tave_rec", "pave": "pave_rec","counts":"counts_rec"})
        df_gen = df_gen.rename(columns={'qave': 'qave_gen', 'xave': 'xave_gen', 'tave': 'tave_gen', 'pave': 'pave_gen', 'Gencounts': 'counts_gen'})


        df_merged_1 = pd.merge(df_exp,df_rec,how='inner', on=['qmin','xmin','tmin','pmin'])
        df_merged_2 = pd.merge(df_merged_1,df_gen,how='inner', on=['qmin','xmin','tmin','pmin'])
        df_merged_total = pd.merge(df_merged_2,df_clas6,how='inner', on=['qmin','xmin','tmin','pmin'])

        print(df_merged_total)
        df_merged_total.to_pickle(datafile_base_dir+binned_data_dir + merged_data_name+".pkl")


        # Calc x-section:

        base_dir = datafile_base_dir+binned_data_dir
        df = df_merged_total


        df.loc[:,"gamma_exp"] = get_gamma(df["xave_exp"],df["qave_exp"],10.604)[0] #get_gamma((dfout["xmin"]+dfout["xmax"])/2,(dfout["qmin"]+dfout["qmax"])/2)[0]
        df.loc[:,"epsi_exp"] =  get_gamma(df["xave_exp"],df["qave_exp"],10.604)[1] #get_gamma((dfout["xmin"]+dfout["xmax"])/2,(dfout["qmin"]+dfout["qmax"])/2)[1]

        df.loc[:,"binvol"] = (df["qmax"]-df["qmin"])*(df["xmax"]-df["xmin"])*(df["tmax"]-df["tmin"])*(df["pmax"]-df["pmin"])*3.14159/180

        df.loc[:,"acc_corr"] = df["counts_rec"]/df["counts_gen"]

        df.loc[:,"xsec"] = df["counts_exp"]/Clas12_exp_luminosity/df["binvol"]
        df.loc[:,"xsec_corr"] = df["xsec"]/df["acc_corr"]
        df.loc[:,"xsec_corr_red"] = df["xsec_corr"]/df["gamma_exp"]
        df.loc[:,"xsec_corr_red_nb"] = df["xsec_corr_red"]*1E33

        df.loc[:,"xsec_ratio_exp"] = df["xsec_corr_red_nb"]/df["dsdtdp"]


        df.loc[:,"uncert_counts_exp"] = np.sqrt(df["counts_exp"])
        df.loc[:,"uncert_counts_rec"] = np.sqrt(df["counts_rec"])
        df.loc[:,"uncert_counts_gen"] = np.sqrt(df["counts_gen"])

        df.loc[:,"uncert_xsec"] = df["uncert_counts_exp"]/df["counts_exp"]*df["xsec"]
        df.loc[:,"uncert_acc_corr"] = np.sqrt(  np.square(df["uncert_counts_rec"]/df["counts_rec"]) + np.square(df["uncert_counts_gen"]/df["counts_gen"]))*df["acc_corr"]
        df.loc[:,"uncert_xsec_corr_red_nb"] = np.sqrt(  np.square(df["uncert_xsec"]/df["xsec"]) + np.square(df["uncert_acc_corr"]/df["acc_corr"]))*df["xsec_corr_red_nb"]

        df.loc[:,"uncert_xsec_ratio_exp"] = np.sqrt(  np.square(df["uncert_xsec_corr_red_nb"]/df["xsec_corr_red_nb"]) + np.square(df["stat"]/df["dsdtdp"]) + np.square(df["sys"]/df["dsdtdp"]) )*df["xsec_ratio_exp"]



        df.to_pickle(datafile_base_dir + final_xsec_dir+final_output_name+".pkl")
        try:
            df.to_csv(datafile_base_dir + final_xsec_dir+final_output_name+".csv")
        except:
            print("Error saving CSV, continuing")

        print("Output pickle file save to {}".format(datafile_base_dir + final_xsec_dir+final_output_name+".pkl"))

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

        df = pd.read_pickle(datafile_base_dir + final_xsec_dir+final_output_name+".pkl")
                                            
        df_sf_binned = pd.read_pickle('final_data_files/clas6_structure_funcs_binned.pkl')
        df_sf_binned = df_sf_binned.apply(pd.to_numeric)

        #for col in df.columns:
        #    print(col)
        #print(df_sf_binned.head(3))
        #sys.exit()



        df.loc[:,"xsec_corr_nb_gamma"] = df["xsec_corr"]*1E33/df["gamma_exp"]

        df.loc[:,"tot_clas6_uncert"] = np.sqrt(np.square(df["stat"]/df["dsdtdp"]) + np.square(df["sys"]/df["dsdtdp"]))*df["dsdtdp"]

        df.loc[:,"epsi_clas6"] = get_gamma(df["x"],df["q"],5.776)[1] #get_gamma((dfout["xmin"]+dfout["xmax"])/2,(dfout["qmin"]+dfout["qmax"])/2)[1]

        df.loc[:,"uncert_xsec_corr_nb_gamma"] = np.sqrt(  np.square(df["uncert_xsec"]/df["xsec"]) + np.square(df["uncert_acc_corr"]/df["acc_corr"]))*df["xsec_corr_nb_gamma"]
        df.loc[:,"c_12_uncert_ratio"] = df['uncert_xsec_corr_nb_gamma']/df['xsec_corr_nb_gamma']

        df.loc[(df.acc_corr < 0.01),'xsec_corr_nb_gamma']=np.nan

        q2bins,xBbins, tbins, phibins = fs.q2bins, fs.xBbins, fs.tbins, fs.phibins

        qrange = [q2bins[0], q2bins[-1]]
        xBrange = [xBbins[0], xBbins[-1]]
        trange = [tbins[0], tbins[-1]]


        sf_data_vals = []

        reduced_plot_dir = datafile_base_dir+reduced_xsection_plots_dir+run_identifiyer+"/"
        if not os.path.exists(reduced_plot_dir):
            os.makedirs(reduced_plot_dir)

        for qmin,qmax in zip(q2bins[0:-1],q2bins[1:]):
            print(" \n Q2 bin: {} to {}".format(qmin,qmax))
            for xmin,xmax in zip(xBbins[0:-1],xBbins[1:]):
                for tmin,tmax in zip(tbins[0:-1],tbins[1:]):

                    query = "qmin == {} and xmin == {} and tmin == {}".format(qmin,xmin,tmin)

                    df_small = df.query(query)
                    df_sf_binned_small = df_sf_binned.query(query)

                    df_check = df_small[df_small["xsec_corr_nb_gamma"].notnull()]

                    if np.isnan(df_sf_binned_small["tel"].values[0]) or df_check.empty or df_small[df_small["xsec_corr_nb_gamma"].notnull()].shape[0]<3:
                        pass
                    else:
                        epsi_mean_c6 = df_small["epsi_clas6"].mean()

                        epsi_mean_c12 = df_small["epsi_exp"].mean()
                        mean_xsec_uncer_ratio_c12 = df_small['c_12_uncert_ratio'].mean()

                       

                        binscenters_c12 = df_small["pave_exp"]
                        data_entries_c12 = df_small["xsec_corr_nb_gamma"]
                        sigma_c12 = df_small["uncert_xsec_corr_nb_gamma"]
                        binscenters = df_small["p"]
                        data_entries = df_small["dsdtdp"]
                        sigma = df_small["tot_clas6_uncert"]

                        x = binscenters
                        y = data_entries

                        def resid_weighted_c12(pars):
                            return (((y-fit_function(x,pars))**2)/sigma_c12).sum()

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

                        tel_c12 = a_c12*6.28
                        tt_c12 = b_c12/epsi_mean_c12*6.28
                        lt_c12 = c_c12/np.sqrt(2*epsi_mean_c12*(1+epsi_mean_c12))*6.28

                        tel_c12_err = tel_c12*a_err/a
                        tt_c12_err = tt_c12*b_err/b
                        lt_c12_err = lt_c12*c_err/c

                        fit_y_data_weighted_new_c12 = fit_function(xspace, a_c12,b_c12,c_c12)


                        q_mean_c12 = df_small['qave_exp'].mean()
                        x_mean_c12 = df_small['xave_exp'].mean()
                        t_mean_c12 = df_small['tave_exp'].mean()

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
                            qmin,xmin,tmin,qmax,xmax,tmax,
                            tel_c12_err,tt_c12_err,lt_c12_err,])

                        plt.rcParams["font.size"] = "20"

                        fig, ax = plt.subplots(figsize =(14, 10)) 

                        #plt.errorbar(binscenters, data_entries, yerr=sigma, color="red",fmt="o",label='CLAS6 Data')#. Bin Averages: Q2: {:.2f} xB: {:.2f} t: {:.2f}'.format(df_sf_binned_small['q'].values[0],df_sf_binned_small['x'].values[0],df_sf_binned_small['t'].values[0]))

                        plt.errorbar(binscenters_c12, data_entries_c12, yerr=sigma_c12, color="blue",fmt="x",label='CLAS12 Data')#. Bin Averages: Q2: {:.2f} xB: {:.2f} t: {:.2f}'.format(df_small["qave_exp"].mean(),df_small["xave_exp"].mean(),df_small["tave_exp"].mean()))

                        plt.rcParams["font.size"] = "20"

                        #fit2, = ax.plot(xspace, fit_y_data_weighted, color='red', linewidth=2.5, label='CLAS6 Fit:         t+l:{:.0f} tt:{:.0f} lt:{:.0f}'.format(pub_tel,pub_tt,pub_lt))
                        #fit2, = ax.plot(xspace, fit_y_data_weighted, color='red', linewidth=2.5)#, label='CLAS6 Fit')        
                        
                        #fit4, = ax.plot(xspace, fit_y_data_weighted_new_c12, color='blue', linewidth=2.5, label='CLAS12 Fit: t+l:{:.0f} tt:{:.0f} lt:{:.0f}'.format(tel_c12,tt_c12,lt_c12))
                        fit4, = ax.plot(xspace, fit_y_data_weighted_new_c12, color='blue', linewidth=2.5)#, label='CLAS12 Fit')     
                        
                        ax.legend(loc="best")
                        ax.set_xlabel("Phi")  
                        ax.set_ylabel('Reduced Cross Section (nb/GeV$^2$)')  
                        title = "Reduced Cross Section Fit Over Phi, Q$^2$ = {:.2f}, x$_B$ = {:.2f}, t = {:.1f}".format(df_sf_binned_small['q'].values[0],df_sf_binned_small['x'].values[0],df_sf_binned_small['t'].values[0])
                        plt.title(title)

                        plt.savefig(reduced_plot_dir+title.replace("$","").replace(".","").replace("^","").replace(" ","").replace("=","").replace(",","_")+".png")

                        plt.close()

        df_out = pd.DataFrame(sf_data_vals, columns = ['qC6','xC6','tC6','telC6','tel-statC6','tel-sysC6','ltC6','lt-statC6','lt-sysC6','ttC6','tt-statC6',
                                        'tt-sysC6','qC12','xC12','tC12','tel_c12','tt_c12','lt_c12','mean_uncert_c12','qmin','xmin','tmin',
                                        'qmax','xmax','tmax','tel_c12_err','tt_c12_err','lt_c12_err'])


        df_out.to_pickle(datafile_base_dir + final_xsec_dir+"struct_funcs_"+run_identifiyer+".pkl")


    if calc_xsection_c12_only:
        if not bin_all_events:
            df_gen = pd.read_pickle(datafile_base_dir+binned_data_dir + gen_file_base+"_binned"+".pkl")
            df_exp = pd.read_pickle(datafile_base_dir+binned_data_dir+exp_file_base+run_identifiyer+"_binned"+".pkl")
            df_rec = pd.read_pickle(datafile_base_dir+binned_data_dir+rec_file_base+run_identifiyer+"_binned"+".pkl")

        

        df_exp = df_exp.rename(columns={"qave": "qave_exp", "xave": "xave_exp","tave": "tave_exp", "pave": "pave_exp","counts":"counts_exp"})
        df_rec = df_rec.rename(columns={"qave": "qave_rec", "xave": "xave_rec","tave": "tave_rec", "pave": "pave_rec","counts":"counts_rec"})
        df_gen = df_gen.rename(columns={'qave': 'qave_gen', 'xave': 'xave_gen', 'tave': 'tave_gen', 'pave': 'pave_gen', 'Gencounts': 'counts_gen'})


        df_merged_1 = pd.merge(df_exp,df_rec,how='inner', on=['qmin','xmin','tmin','pmin'])
        df_merged_total = pd.merge(df_merged_1,df_gen,how='inner', on=['qmin','xmin','tmin','pmin'])

        print(df_merged_total)
        df_merged_total.to_pickle(datafile_base_dir+binned_data_dir + merged_data_name+".pkl")


        # Calc x-section:

        base_dir = datafile_base_dir+binned_data_dir
        df = df_merged_total


        df.loc[:,"gamma_exp"] = get_gamma(df["xave_exp"],df["qave_exp"],10.604)[0] #get_gamma((dfout["xmin"]+dfout["xmax"])/2,(dfout["qmin"]+dfout["qmax"])/2)[0]
        df.loc[:,"epsi_exp"] =  get_gamma(df["xave_exp"],df["qave_exp"],10.604)[1] #get_gamma((dfout["xmin"]+dfout["xmax"])/2,(dfout["qmin"]+dfout["qmax"])/2)[1]

        df.loc[:,"binvol"] = (df["qmax"]-df["qmin"])*(df["xmax"]-df["xmin"])*(df["tmax"]-df["tmin"])*(df["pmax"]-df["pmin"])*3.14159/180

        df.loc[:,"acc_corr"] = df["counts_rec"]/df["counts_gen"]

        df.loc[:,"xsec"] = df["counts_exp"]/Clas12_exp_luminosity/df["binvol"]
        df.loc[:,"xsec_corr"] = df["xsec"]/df["acc_corr"]
        df.loc[:,"xsec_corr_red"] = df["xsec_corr"]/df["gamma_exp"]
        df.loc[:,"xsec_corr_red_nb"] = df["xsec_corr_red"]*1E33


        df.loc[:,"uncert_counts_exp"] = np.sqrt(df["counts_exp"])
        df.loc[:,"uncert_counts_rec"] = np.sqrt(df["counts_rec"])
        df.loc[:,"uncert_counts_gen"] = np.sqrt(df["counts_gen"])

        df.loc[:,"uncert_xsec"] = df["uncert_counts_exp"]/df["counts_exp"]*df["xsec"]
        df.loc[:,"uncert_acc_corr"] = np.sqrt(  np.square(df["uncert_counts_rec"]/df["counts_rec"]) + np.square(df["uncert_counts_gen"]/df["counts_gen"]))*df["acc_corr"]
        df.loc[:,"uncert_xsec_corr_red_nb"] = np.sqrt(  np.square(df["uncert_xsec"]/df["xsec"]) + np.square(df["uncert_acc_corr"]/df["acc_corr"]))*df["xsec_corr_red_nb"]




        df.to_pickle(datafile_base_dir + final_xsec_dir+final_output_name+".pkl")
        try:
            df.to_csv(datafile_base_dir + final_xsec_dir+final_output_name+".csv")
        except:
            print("Error saving CSV, continuing")

        print("Output pickle file save to {}".format(datafile_base_dir + final_xsec_dir+final_output_name+".pkl"))

    if plot_reduced_xsec_and_fit_c12_only:
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

        xmax = 360
        xspace = np.linspace(0, xmax, 1000)

        df = pd.read_pickle(datafile_base_dir + final_xsec_dir+final_output_name+".pkl")
                                            


        df.loc[:,"xsec_corr_nb_gamma"] = df["xsec_corr"]*1E33/df["gamma_exp"]



        df.loc[:,"uncert_xsec_corr_nb_gamma"] = np.sqrt(  np.square(df["uncert_xsec"]/df["xsec"]) + np.square(df["uncert_acc_corr"]/df["acc_corr"]))*df["xsec_corr_nb_gamma"]
        df.loc[:,"c_12_uncert_ratio"] = df['uncert_xsec_corr_nb_gamma']/df['xsec_corr_nb_gamma']

        df.loc[(df.acc_corr < 0.01),'xsec_corr_nb_gamma']=np.nan

        q2bins,xBbins, tbins, phibins = fs.q2bins, fs.xBbins, fs.tbins, fs.phibins

        qrange = [q2bins[0], q2bins[-1]]
        xBrange = [xBbins[0], xBbins[-1]]
        trange = [tbins[0], tbins[-1]]


        sf_data_vals = []

        reduced_plot_dir = datafile_base_dir+reduced_xsection_plots_dir+run_identifiyer+"/"
        if not os.path.exists(reduced_plot_dir):
            os.makedirs(reduced_plot_dir)

        for qmin,qmax in zip(q2bins[0:-1],q2bins[1:]):
            print(" \n Q2 bin: {} to {}".format(qmin,qmax))
            for xmin,xmax in zip(xBbins[0:-1],xBbins[1:]):
                for tmin,tmax in zip(tbins[0:-1],tbins[1:]):

                    query = "qmin == {} and xmin == {} and tmin == {}".format(qmin,xmin,tmin)

                    #ic(query)

                    df_small = df.query(query)

                    #ic(df_small)
                    
                    df_check = df_small[df_small["xsec_corr_nb_gamma"].notnull()]

                    if  df_check.empty or df_small[df_small["xsec_corr_nb_gamma"].notnull()].shape[0]<3:
                        pass
                    else:

                        epsi_mean_c12 = df_small["epsi_exp"].mean()
                        mean_xsec_uncer_ratio_c12 = df_small['c_12_uncert_ratio'].mean()

                       

                        binscenters_c12 = df_small["pave_exp"]
                        data_entries_c12 = df_small["xsec_corr_nb_gamma"]
                        sigma_c12 = df_small["uncert_xsec_corr_nb_gamma"]


                        def resid_weighted_c12(pars):
                            return (((y-fit_function(x,pars))**2)/sigma_c12).sum()

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

                        ###A +    Bcos(2x) + Ccos(x)
                        ###TEL +   ep*TT   + sqr*LT
                        

                        a_c12,b_c12,c_c12 = a,b,c 

                        tel_c12 = a_c12*6.28
                        tt_c12 = b_c12/epsi_mean_c12*6.28
                        lt_c12 = c_c12/np.sqrt(2*epsi_mean_c12*(1+epsi_mean_c12))*6.28

                        tel_c12_err = tel_c12*a_err/a
                        tt_c12_err = tt_c12*b_err/b
                        lt_c12_err = lt_c12*c_err/c

                        fit_y_data_weighted_new_c12 = fit_function(xspace, a_c12,b_c12,c_c12)


                        q_mean_c12 = df_small['qave_exp'].mean()
                        x_mean_c12 = df_small['xave_exp'].mean()
                        t_mean_c12 = df_small['tave_exp'].mean()

                        plt.rcParams["font.size"] = "20"

                        fig, ax = plt.subplots(figsize =(14, 10)) 

                        #plt.errorbar(binscenters, data_entries, yerr=sigma, color="red",fmt="o",label='CLAS6 Data')#. Bin Averages: Q2: {:.2f} xB: {:.2f} t: {:.2f}'.format(df_sf_binned_small['q'].values[0],df_sf_binned_small['x'].values[0],df_sf_binned_small['t'].values[0]))

                        plt.errorbar(binscenters_c12, data_entries_c12, yerr=sigma_c12, color="blue",fmt="x",label='CLAS12 Data')#. Bin Averages: Q2: {:.2f} xB: {:.2f} t: {:.2f}'.format(df_small["qave_exp"].mean(),df_small["xave_exp"].mean(),df_small["tave_exp"].mean()))

                        plt.rcParams["font.size"] = "20"

                        #fit2, = ax.plot(xspace, fit_y_data_weighted, color='red', linewidth=2.5, label='CLAS6 Fit:         t+l:{:.0f} tt:{:.0f} lt:{:.0f}'.format(pub_tel,pub_tt,pub_lt))
                        #fit2, = ax.plot(xspace, fit_y_data_weighted, color='red', linewidth=2.5)#, label='CLAS6 Fit')        
                        
                        #fit4, = ax.plot(xspace, fit_y_data_weighted_new_c12, color='blue', linewidth=2.5, label='CLAS12 Fit: t+l:{:.0f} tt:{:.0f} lt:{:.0f}'.format(tel_c12,tt_c12,lt_c12))
                        fit4, = ax.plot(xspace, fit_y_data_weighted_new_c12, color='blue', linewidth=2.5)#, label='CLAS12 Fit')     
                        
                        ax.legend(loc="best")
                        ax.set_xlabel("Phi")  
                        ax.set_ylabel('Reduced Cross Section (nb/GeV$^2$)')  
                        title = "Reduced Cross Section Fit Over Phi, Q$^2$ = {:.2f}, x$_B$ = {:.2f}, t = {:.1f}".format(q_mean_c12,x_mean_c12,t_mean_c12)
                        plt.title(title)

                        plt.savefig(reduced_plot_dir+title.replace("$","").replace(".","").replace("^","").replace(" ","").replace("=","").replace(",","_")+".png")

                        plt.close()


    if comp_2_config:

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

        xmax = 360
        xspace = np.linspace(0, xmax, 1000)



        df = pd.read_pickle(df_final_config_1)
        df2 = pd.read_pickle(df_final_config_2)

                                            


        df.loc[:,"xsec_corr_nb_gamma"] = df["xsec_corr"]*1E33/df["gamma_exp"]
        df.loc[:,"uncert_xsec_corr_nb_gamma"] = np.sqrt(  np.square(df["uncert_xsec"]/df["xsec"]) + np.square(df["uncert_acc_corr"]/df["acc_corr"]))*df["xsec_corr_nb_gamma"]
        df.loc[:,"c_12_uncert_ratio"] = df['uncert_xsec_corr_nb_gamma']/df['xsec_corr_nb_gamma']
        df.loc[(df.acc_corr < 0.01),'xsec_corr_nb_gamma']=np.nan
        df.loc[(df.xsec_corr_nb_gamma > 1E9),'xsec_corr_nb_gamma']=np.nan



        df2.loc[:,"xsec_corr_nb_gamma"] = df2["xsec_corr"]*1E33/df2["gamma_exp"]
        df2.loc[:,"uncert_xsec_corr_nb_gamma"] = np.sqrt(  np.square(df2["uncert_xsec"]/df2["xsec"]) + np.square(df2["uncert_acc_corr"]/df2["acc_corr"]))*df2["xsec_corr_nb_gamma"]
        df2.loc[:,"c_12_uncert_ratio"] = df2['uncert_xsec_corr_nb_gamma']/df2['xsec_corr_nb_gamma']
        df2.loc[(df2.acc_corr < 0.01),'xsec_corr_nb_gamma']=np.nan
        df2.loc[(df2.xsec_corr_nb_gamma > 1E9),'xsec_corr_nb_gamma']=np.nan




        q2bins,xBbins, tbins, phibins = fs.q2bins, fs.xBbins, fs.tbins, fs.phibins

        qrange = [q2bins[0], q2bins[-1]]
        xBrange = [xBbins[0], xBbins[-1]]
        trange = [tbins[0], tbins[-1]]


        sf_data_vals = []

        reduced_plot_dir = datafile_base_dir+reduced_xsection_plots_dir+run_identifiyer+"comparison/"
        if not os.path.exists(reduced_plot_dir):
            os.makedirs(reduced_plot_dir)

        for qmin,qmax in zip(q2bins[0:-1],q2bins[1:]):
            print(" \n Q2 bin: {} to {}".format(qmin,qmax))
            for xmin,xmax in zip(xBbins[0:-1],xBbins[1:]):
                for tmin,tmax in zip(tbins[0:-1],tbins[1:]):

                    query = "qmin == {} and xmin == {} and tmin == {}".format(qmin,xmin,tmin)

                    df_small = df.query(query)
                    df2_small = df2.query(query)


                    df_check = df_small[df_small["xsec_corr_nb_gamma"].notnull()]
                    df2_check = df2_small[df2_small["xsec_corr_nb_gamma"].notnull()]


                    if  df_check.empty or df2_check.empty or df_small[df_small["xsec_corr_nb_gamma"].notnull()].shape[0]<3 or df2_small[df2_small["xsec_corr_nb_gamma"].notnull()].shape[0]<3:
                        pass
                    else:

                        epsi_mean_c12 = df_small["epsi_exp"].mean()
                        mean_xsec_uncer_ratio_c12 = df_small['c_12_uncert_ratio'].mean()
                        
                        epsi_mean_c12_2 = df2_small["epsi_exp"].mean()
                        mean_xsec_uncer_ratio_c12_2 = df2_small['c_12_uncert_ratio'].mean()
                       

                        binscenters_c12 = df_small["pave_exp"]
                        data_entries_c12 = df_small["xsec_corr_nb_gamma"]
                        sigma_c12 = df_small["uncert_xsec_corr_nb_gamma"]

                        binscenters_c12_2 = df2_small["pave_exp"]
                        data_entries_c12_2 = df2_small["xsec_corr_nb_gamma"]
                        sigma_c12_2 = df2_small["uncert_xsec_corr_nb_gamma"]


                        def resid_weighted_c12(pars):
                            return (((y-fit_function(x,pars))**2)/sigma_c12).sum()

                        def resid_weighted_c12_2(pars):
                            return (((y-fit_function(x,pars))**2)/sigma_c12_2).sum()

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

                        x_2 = binscenters_c12_2
                        y_2 = data_entries_c12_2
                        valid_2 = ~(np.isnan(x_2) | np.isnan(y_2))

                        print(x[valid])
                        print(y[valid])
                        print(x_2[valid_2])
                        print(y_2[valid_2])

                        popt_0, pcov = curve_fit(fit_function, xdata=x[valid], ydata=y[valid], p0=[100,-60,-11],
                            sigma=sigma_c12[valid], absolute_sigma=True)

                        popt, pcov = curve_fit(fit_function, xdata=x[valid], ydata=y[valid], p0=[popt_0[0],popt_0[1],popt_0[2]],
                                    sigma=sigma_c12[valid], absolute_sigma=True)

                        popt_0_2, pcov_2 = curve_fit(fit_function, xdata=x_2[valid_2], ydata=y_2[valid_2], p0=[100,-60,-11],
                            sigma=sigma_c12_2[valid_2], absolute_sigma=True)

                        popt_2, pcov_2 = curve_fit(fit_function, xdata=x_2[valid_2], ydata=y_2[valid_2], p0=[popt_0_2[0],popt_0_2[1],popt_0_2[2]],
                                    sigma=sigma_c12_2[valid_2], absolute_sigma=True)


                        a,b,c = popt[0],popt[1],popt[2]
                        
                        a_err = np.sqrt(pcov[0][0])#*qmod
                        b_err = np.sqrt(pcov[1][1])#*qmod
                        c_err = np.sqrt(pcov[2][2])#*qmod

                        a2,b2,c2 = popt_2[0],popt_2[1],popt_2[2]
                        
                        a2_err = np.sqrt(pcov_2[0][0])#*qmod
                        b2_err = np.sqrt(pcov_2[1][1])#*qmod
                        c2_err = np.sqrt(pcov_2[2][2])#*qmod

                        ###A +    Bcos(2x) + Ccos(x)
                        ###TEL +   ep*TT   + sqr*LT
                        

                        a_c12,b_c12,c_c12 = a,b,c 
                        a2_c12,b2_c12,c2_c12 = a2,b2,c2 


                        tel_c12 = a_c12*6.28
                        tt_c12 = b_c12/epsi_mean_c12*6.28
                        lt_c12 = c_c12/np.sqrt(2*epsi_mean_c12*(1+epsi_mean_c12))*6.28

                        tel_c12_err = tel_c12*a_err/a
                        tt_c12_err = tt_c12*b_err/b
                        lt_c12_err = lt_c12*c_err/c

                        fit_y_data_weighted_new_c12 = fit_function(xspace, a_c12,b_c12,c_c12)

                        fit_y_data_weighted_new_c12_2 = fit_function(xspace, a2_c12,b2_c12,c2_c12)

                        q_mean_c12 = df_small['qave_exp'].mean()
                        x_mean_c12 = df_small['xave_exp'].mean()
                        t_mean_c12 = df_small['tave_exp'].mean()

                        q_mean_c12_2 = df2_small['qave_exp'].mean()
                        x_mean_c12_2 = df2_small['xave_exp'].mean()
                        t_mean_c12_2 = df2_small['tave_exp'].mean()

                        plt.rcParams["font.size"] = "20"

                        fig, ax = plt.subplots(figsize =(14, 10)) 

                        #plt.errorbar(binscenters, data_entries, yerr=sigma, color="red",fmt="o",label='CLAS6 Data')#. Bin Averages: Q2: {:.2f} xB: {:.2f} t: {:.2f}'.format(df_sf_binned_small['q'].values[0],df_sf_binned_small['x'].values[0],df_sf_binned_small['t'].values[0]))

                        plt.errorbar(binscenters_c12, data_entries_c12, yerr=sigma_c12, color="blue",fmt="x",label='CLAS12 Outb.')#. Bin Averages: Q2: {:.2f} xB: {:.2f} t: {:.2f}'.format(df_small["qave_exp"].mean(),df_small["xave_exp"].mean(),df_small["tave_exp"].mean()))
                        plt.errorbar(binscenters_c12_2, data_entries_c12_2, yerr=sigma_c12_2, color="red",fmt="o",label='CLAS12 Inb.')#. Bin Averages: Q2: {:.2f} xB: {:.2f} t: {:.2f}'.format(df_small["qave_exp"].mean(),df_small["xave_exp"].mean(),df_small["tave_exp"].mean()))

                        plt.rcParams["font.size"] = "20"

                        #fit2, = ax.plot(xspace, fit_y_data_weighted, color='red', linewidth=2.5, label='CLAS6 Fit:         t+l:{:.0f} tt:{:.0f} lt:{:.0f}'.format(pub_tel,pub_tt,pub_lt))
                        #fit2, = ax.plot(xspace, fit_y_data_weighted, color='red', linewidth=2.5)#, label='CLAS6 Fit')        
                        
                        #fit4, = ax.plot(xspace, fit_y_data_weighted_new_c12, color='blue', linewidth=2.5, label='CLAS12 Fit: t+l:{:.0f} tt:{:.0f} lt:{:.0f}'.format(tel_c12,tt_c12,lt_c12))
                        fit4, = ax.plot(xspace, fit_y_data_weighted_new_c12, color='blue', linewidth=2.5)#, label='CLAS12 Fit')     
                        fit5, = ax.plot(xspace, fit_y_data_weighted_new_c12_2, color='red', linewidth=2.5)#, label='CLAS12 Fit')     
                        
                        ax.legend(loc="best")
                        ax.set_xlabel("Phi")  
                        ax.set_ylabel('Reduced Cross Section (nb/GeV$^2$)')  
                        title = "Reduced Cross Section Fit Over Phi, Q$^2$ = {:.2f} ({:.2f}), x$_B$ = {:.2f} ({:.2f}), t = {:.1f} ({:.2f})".format(q_mean_c12,q_mean_c12_2,x_mean_c12,x_mean_c12_2,t_mean_c12,t_mean_c12_2)
                        plt.title(title)

                        plt.savefig(reduced_plot_dir+title.replace("$","").replace(".","").replace("^","").replace(" ","").replace("=","").replace(",","_")+".png")

                        plt.close()


# Directory definitions
# # unique_identifyer = ""
# # det_proton_loc="CD"
# # det_photon1_loc="All"
# # det_photon2_loc="All"

# # #Analysis topology defintion
# # #generator_type = "norad"
# # #mag_config = "inbending"
# # #mag_config = "outbending"

# # # mag_configs = ["inbending","outbending"]
# # # generator_type = "rad"
# # # proton_locs = ["CD","FD"]
# # # photon1_locs = ["FD","FT","All"]
# # # photon2_locs = ["FD","FT","All"]

#mag_configs = ["inbending","outbending"]
#mag_configs = ["outbending",]#"outbending"]

if __name__ == "__main__":
    #run_name = "new_f18_in_processing_simple_cuts"
    run_name = "testing_new_binning_mechanism_small_bins"



    if 1==1:
        mag_configs = ["inbending"]#,"outbending"]
        generator_type = "rad"
        proton_locs = ["All",]
        photon1_locs = ["All",]
        photon2_locs = ["All",]
        sigma_multis = [3,]

        for mc in mag_configs:
            for sigma_multiplier in sigma_multis:
                for pl in proton_locs:
                    for p1l in photon1_locs:
                        for p2l in photon2_locs:
                            print("ON SIGMA, MAG CONFIG: {},{}".format(sigma_multiplier,mc))
                            run_analysis(mc,generator_type,unique_identifyer=run_name,#"for_aps_gen_plots_norad_bigplots",
                                        det_proton_loc=pl,det_photon1_loc=p1l,det_photon2_loc=p2l,
                                        convert_roots = 0,
                                        make_exclusive_cuts = 1,
                                        plot_initial_distros = 0,
                                        plot_final_distros = 0,
                                        bin_all_events = 1,
                                        bin_gen = 1,
                                        calc_xsection = 0,
                                        plot_reduced_xsec_and_fit = 0,
                                        calc_xsection_c12_only = 1,
                                        plot_reduced_xsec_and_fit_c12_only = 1,
                                        plot_1_D_hists = 0,
                                        simple_exclusivity_cuts=False,
                                        emergency_stop = 0,
                                        comp_2_config=False,
                                        gen_ex_cut_table=False,
                                        sigma_multiplier=sigma_multiplier)




    if 0==1:
        mag_configs = ["outbending"]
        generator_type = "rad"
        proton_locs = ["All",]
        photon1_locs = ["All",]
        photon2_locs = ["All",]

        for mc in mag_configs:
            for pl in proton_locs:
                for p1l in photon1_locs:
                    for p2l in photon2_locs:
                        run_analysis(mc,generator_type,unique_identifyer=run_name,#"for_aps_gen_plots_norad_bigplots",
                                    det_proton_loc=pl,det_photon1_loc=p1l,det_photon2_loc=p2l,
                                    convert_roots = 0,
                                    make_exclusive_cuts = 0,
                                    plot_initial_distros = 0,
                                    plot_final_distros = 0,
                                    bin_all_events = 0,
                                    bin_gen = 0,
                                    calc_xsection = 0,
                                    calc_xsection_c12_only = 0,
                                    plot_reduced_xsec_and_fit = 0,
                                    plot_1_D_hists = 0,
                                    simple_exclusivity_cuts=False,
                                    emergency_stop = 0,
                                    plot_reduced_xsec_and_fit_c12_only = 0,
                                    comp_2_config=True)




    sys.exit()

    q2_ranges = [[1.5,2.0],[2.0,2.5],[4.0,4.5]]
    xB_ranges = [[0.3,0.35],[0.3,0.35],[0.5,0.55]]
    t_ranges = [[0.4,0.6],[0.4,0.6],[0.6,1]]

    for q,x,t in zip(q2_ranges,xB_ranges,t_ranges):
        unique_identifyer = "q2exam_{}_{}_{}".format(round((q[0]+q[1])/2,2),round((x[0]+x[1])/2,2),round((t[0]+t[1])/2,2))
        q_min = q[0]
        q_max = q[1]
        x_min = x[0]
        x_max = x[1]
        t_min = t[0]
        t_max = t[1]

        #mag_configs = ["inbending","outbending"]
        mag_configs = ["outbending"]
        generator_type = "norad"
        pl = "All"
        p1l = "All"
        p2l = "All"
        for mc in mag_configs:
            run_analysis(mc,generator_type,unique_identifyer=unique_identifyer,
                                        det_proton_loc=pl,det_photon1_loc=p1l,det_photon2_loc=p2l,
                                        convert_roots = 0,
                                        make_exclusive_cuts = 1,
                                        plot_initial_distros = 0,
                                        plot_final_distros = 1,
                                        bin_all_events = 1,
                                        bin_gen = 0,
                                        calc_xsection = 1,
                                        plot_reduced_xsec_and_fit = 1,
                                        plot_1_D_hists = 0,
                                        emergency_stop = 1,
                                        qxt_cuts = [q,x,t])

