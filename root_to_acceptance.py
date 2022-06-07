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

def get_files(dirname):
    """
    Get all files in base_dir
    """

    print("looking for files in {}".format(dirname))
    jobs_list = []
    print(os.listdir(dirname))
    for f in os.listdir(dirname):
        f_path = dirname + f
        if os.path.isfile(f_path):
            jobs_list.append(f)
    jobs_list = sorted(jobs_list)
    print("Found {} files in jobs directory".format(len(jobs_list)))

    for fname in jobs_list:
        if "recon" in fname:
            recon_file = fname
        if "gen" in fname:
            gen_file = fname

    print("Generator file: {}".format(gen_file))
    print("Reconstructed file: {}".format(recon_file))
            
    return gen_file, recon_file

fs = filestruct.fs()


allz = False
QuickTesting = False
#All
if allz:
    DoWipe = True
    DoGen = True
    DoRecon = True
    DoInspect = True
    DoPlot = False
    DoBin = True
    DoCombine = True
    DoMetaCombine = True
#Just ana
else:
    DoWipe = False
    DoGen = False
    DoRecon = False
    DoInspect = False
    DoBin = True
    DoCombine = True
    DoMetaCombine = True



base_dir = "/mnt/d/GLOBUS/CLAS12/simulations/production/F2018_In_Norad"
# To add data to this base dir, make a subdir /runs/<run_number>/roots and put the files there
# Note that dataset 101 is just for testing
if base_dir[-1] != '/':
    base_dir += '/'

dirname = base_dir+"runs/"
#print(os.listdir(dirname))
runs_list = [run for run in os.listdir(dirname) if os.path.isdir(dirname+"/"+run)]


dir_ending_list = ["/event_pickles/", "/plots/", "/binned_pickles/"]
for run in runs_list:
    for dir_ending in dir_ending_list:
        if DoWipe:
            shutil.rmtree(dirname+run+dir_ending) if os.path.isdir(dirname+run+dir_ending) else None
        if not os.path.isdir(dirname+run+dir_ending):
            os.makedirs(dirname+run+dir_ending)

generator_type = "norad"

converter_gen = convert_GEN_NORAD_root_to_pkl
converter_recon = convert_RECON_NORAD_root_to_pkl

if generator_type == "rad":
    converter_gen = convert_GEN_RAD_root_to_pkl
    converter_recon = convert_RECON_RAD_root_to_pkl

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get args",formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-f","--fname", help="a single root file to convert into pickles", default="/Users/sangbaek/Dropbox (MIT)/data/project/merged_9628_files.root")
    parser.add_argument("-o","--out", help="a single pickle file name as an output", default="goodbyeRoot.pkl")
    parser.add_argument("-s","--entry_stop", help="entry_stop to stop reading the root file", default = None)
    parser.add_argument("-p","--polarity", help="polarity", default = "inbending")
    
    args = parser.parse_args()


    if QuickTesting:
        runs_list = [runs_list[0]]
        print("Using first run directory only as test: {}".format(runs_list))

    print("starting analysis")
    for run in runs_list:

        root_file_list = os.listdir(dirname+run+"/roots/")
        gen_file, recon_file = get_files(dirname+run+"/roots/")

        rec_out_name = recon_file.split(".")[0]
        gen_out_name = gen_file.split(".")[0]

        output_loc_event_pkl_after_cuts =     dirname+run+"/event_pickles/"+rec_out_name+"_reconstructed_events_after_cuts.pkl"
        output_loc_event_pkl_all_gen_events = dirname+run+"/event_pickles/"+gen_out_name+"_all_generated_events.pkl"

        if DoGen:
            outname = gen_out_name+"_all_generated_events"
            output_loc_event_pkl = dirname+run+"/event_pickles/"+outname+"_all_generated_events.pkl"
            output_loc_plots = dirname+run+"/plots/"+outname+"/"

            tree = converter_gen.readFile(dirname+run+"/roots/" + gen_file)
            df_gen_all = converter_gen.readEPGG(tree)
            #ic("saving file to: {}".format(output_loc_event_pkl))
            df_gen_all.to_pickle(output_loc_event_pkl_all_gen_events)
        
            if DoPlot:
                histo_plotting.make_all_histos(df_gen_all,datatype="Gen",
                    hists_2d=True,hists_1d=True,hists_overlap=False,
                    saveplots=True,output_dir=output_loc_plots)

            print(df_gen_all.columns)
            print(df_gen_all.head(5))
            print("Number of events: {}".format(df_gen_all.shape[0]))

        if DoRecon:
            outname = recon_file.split(".")[0]
            args.fname = dirname+run+"/roots/" + recon_file
            output_loc_plots_after_cuts = dirname+run+"/plots/"+outname+"_after_cuts/"
            output_loc_plots_truth_after_cuts = dirname+run+"/plots/"+outname+"_truth_after_cuts/"

            output_loc_event_pkl_before_cuts = dirname+run+"/event_pickles/"+outname+"_reconstructed_events_before_cuts.pkl"
            output_loc_plots_before_cuts = dirname+run+"/plots/"+outname+"_before_cuts/"
            output_loc_plots_truth_before_cuts = dirname+run+"/plots/"+outname+"_truth_before_cuts/"



            converter = root2pickle(args.fname, entry_stop = args.entry_stop, pol = args.polarity)
            df_after_cuts = converter.df_after_cuts
            df_after_cuts.to_pickle(output_loc_event_pkl_after_cuts)

            df_before_cuts = converter.df_before_cuts
            df_before_cuts.to_pickle(output_loc_event_pkl_before_cuts)

            if DoPlot:
                print("NOW PLOTTING RECON AFTER CUTS")
                histo_plotting.make_all_histos(df_after_cuts,datatype="Recon",
                                    hists_2d=True,hists_1d=True,hists_overlap=False,
                                    saveplots=True,output_dir=output_loc_plots_after_cuts)

                print("NOW PLOTTING RECON TRUTH AFTER CUTS")

                histo_plotting.make_all_histos(df_after_cuts,datatype="Truth",
                                    hists_2d=True,hists_1d=False,hists_overlap=False,
                                    saveplots=True,output_dir=output_loc_plots_truth_after_cuts)

                print("NOW PLOTTING RECON BEFORE CUTS")

                histo_plotting.make_all_histos(df_before_cuts,datatype="Recon",
                                    hists_2d=True,hists_1d=True,hists_overlap=False,
                                    saveplots=True,output_dir=output_loc_plots_before_cuts)

                print("NOW PLOTTING RECON TRUTH BEFORE CUTS")

                histo_plotting.make_all_histos(df_before_cuts,datatype="Truth",
                                    hists_2d=True,hists_1d=False,hists_overlap=False,
                                    saveplots=True,output_dir=output_loc_plots_truth_before_cuts)

            #print(df_after_cuts.columns)
            #print(df_after_cuts.head(5))
            print("Number of events: {}".format(df_after_cuts.shape[0]))

        if DoBin:
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

            
            df_recon = df[["Q2", "W", "xB", "t1", "phi1"]]
            df_gen = df_gen[["GenQ2", "GenW", "GenxB", "Gent1", "Genphi1"]]

            dfs = [df_recon, df_gen]

            for index,df0 in enumerate(dfs):
                print("Binning df: {}".format(df))
                prefix = "Gen" if index == 1 else ""


                args.test = False
                q2bins,xBbins, tbins, phibins = fs.q2bins, fs.xBbins, fs.tbins, fs.phibins
                if args.test:
                        q2bins,xBbins, tbins, phibins = fs.q2bins_test, fs.xBbins_test, fs.tbins_test, fs.phibins_test

                num_counts = []

                qrange = [q2bins[0], q2bins[-1]]
                xBrange = [xBbins[0], xBbins[-1]]
                trange = [tbins[0], tbins[-1]]

                if prefix=="Gen":                               
                    total_num = df0.query('GenQ2>{} and GenQ2<{} and GenxB>{} and GenxB<{} and Gent1>{} and Gent1<{}'.format(*qrange, *xBrange, *trange)).shape[0]
                else:
                    total_num = df0.query('Q2>{} and Q2<{} and xB>{} and xB<{} and t1>{} and t1<{}'.format(*qrange, *xBrange, *trange)).shape[0]

                for qmin,qmax in zip(q2bins[0:-1],q2bins[1:]):
                    if prefix=="Gen":
                        query = "GenQ2 > {} and GenQ2 < {}".format(qmin,qmax)
                    else:  
                        query = "Q2 > {} and Q2 <= {}".format(qmin,qmax)            
                    df_q = df0.query(query)
                    print("Q2 bin: {} to {}".format(qmin,qmax))                    
                    for xmin,xmax in zip(xBbins[0:-1],xBbins[1:]):
                       # print("xB bin: {} to {}".format(xmin,xmax))
                        if prefix=="Gen":
                            query = "GenQ2 > {} and GenQ2 < {} and GenxB > {} and GenxB".format(qmin,qmax,xmin,xmax)
                        else:  
                            query = "Q2 > {} and Q2 <= {} and xB > {} and xB <= {}".format(qmin,qmax,xmin,xmax)
                        df_qx = df_q.query(query)
                        for tmin,tmax in zip(tbins[0:-1],tbins[1:]):
                            if prefix=="Gen":
                                query = "GenQ2 > {} and GenQ2 < {} and GenxB > {} and GenxB < {} and Gent1 > {} and Gent1 < {}".format(qmin,qmax,xmin,xmax,tmin,tmax)
                            else:  
                                query = "Q2 > {} and Q2 <= {} and xB > {} and xB <= {} and t1 > {} and t1 <= {}".format(qmin,qmax,xmin,xmax,tmin,tmax)
                            df_qxt = df_qx.query(query)
                            for pmin,pmax in zip(phibins[0:-1],phibins[1:]):
                                if prefix=="Gen":
                                    query = "GenQ2 > {} and GenQ2 < {} and GenxB > {} and GenxB < {} and Gent1 > {} and Gent1 < {} and Genphi1 > {} and Genphi1 < {}".format(qmin,qmax,xmin,xmax,tmin,tmax,pmin,pmax)
                                else:  
                                    query = "Q2 > {} and Q2 <= {} and xB > {} and xB <= {} and t1 > {} and t1 <= {} and phi1>{} and phi1<={}".format(qmin,qmax,xmin,xmax,tmin,tmax,pmin,pmax)
                                
                                df_bin = df_qxt.query(query)
                                num_counts.append([qmin,xmin,tmin,pmin,len(df_bin.index)])

            

                df_minibin = pd.DataFrame(num_counts, columns = ['qmin','xmin','tmin','pmin',prefix+'counts'])
                print("Total number of binned events: {}".format(df_minibin[prefix+'counts'].sum()))
                print("Total number of original events: {}".format(total_num))
                df_minibin.to_pickle(dirname+run+"/binned_pickles/"+rec_out_name+prefix+"_reconstructed_events_binned.pkl")
                ################## END OF OLD WAY
                # sys.exit()

                # #bins = [q2bins,xBbins, tbins, phibins]
                # #qlabels, xBlabels, tlabels, philabels = [],[] ,[],[]

                # #labels = [qlabels, xBlabels, tlabels, philabels]

                # #for label_item,bin_item in zip(labels,bins):
                # ##    for count in range(1,len(bin_item)):
                #         label_item.append(str(bin_item[count-1])+"-"+str(bin_item[count]))
                
                # four_squared['qbin'] = pd.cut(four_squared[prefix+'Q2'], q2bins, labels=qlabels)
                # four_squared['tbin'] = pd.cut(four_squared[prefix+'t1'], tbins, labels=tlabels)
                # four_squared['xBbin'] = pd.cut(four_squared[prefix+'xB'], xBbins, labels=xBlabels)
                # four_squared['phibin'] = pd.cut(four_squared[prefix+'phi1'], phibins, labels=philabels)

                # rude_sum = 0

                # ##################### NEW WAY
                # num_counts = []

                # for qval in qlabels:
                #     for xval in xBlabels:
                #         for tval in tlabels:
                #             for phival in philabels:
                #                 df_min = four_squared.query("qbin==@qval and xBbin==@xval and tbin==@tval and phibin==@phival")
                #                 num_counts.append([qval,xval,tval,phival,len(df_min.index)])

                # df = pd.DataFrame(num_counts, columns = ['qbin','xBbin','tbin','phibin',prefix+'counts'])
                # print("Total number of binned events: {}".format(df[prefix+'counts'].sum()))
                # print("Total number of original events: {}".format(orginial_sum))
                # df.to_pickle(dirname+run+"/binned_pickles/"+rec_out_name+prefix+"_reconstructed_events_binned_NEW2.pkl")
                # ################## END OF OLD WAY

                # sys.exit()



                # ################ OLD WAY
                # num_counts = []

                # for qval in qlabels:
                #     ic(qval)
                #     df_min = four_squared.query("qbin==@qval")
                #     if len(df_min.index) == 0:
                #             print("no events found")
                #             #fix this in the future so don't have nested for lops:
                #             for iii in xBlabels:
                #                 for ii in tlabels:
                #                     for i in philabels:
                #                             num_counts.append(0)
                #             #num_counts.append([0]*len(xBlabels)*len(tlabels)*len(philabels))
                #             #print([0]*len(xBlabels)*len(tlabels)*len(philabels))
                #             #print("made a triple")
                #     else:
                #         for xval in xBlabels:
                #             ic(xval)
                #             df_min2 = df_min.query("xBbin==@xval")
                #             if len(df_min2.index) == 0:
                #                 print("no events found")
                #                 for ii in tlabels:
                #                     for i in philabels:
                #                             num_counts.append(0)
                #                 #num_counts.append([0]*len(tlabels)*len(philabels))
                #                 #print([0]*len(tlabels)*len(philabels))
                #                 #print("made a triple")
                #             else:
                #                 for tval in tlabels:
                #                     ic(tval)
                #                     df_min3 = df_min2.query("tbin==@tval")
                #                     if len(df_min3.index) == 0:
                #                         print("no events found")
                #                         for i in philabels:
                #                             num_counts.append(0)
                #                     else:
                #                         for phival in philabels:
                #                             df_min4 = df_min3.query("phibin==@phival")
                #                             print(len(df_min4.index))
                #                             rude_sum += len(df_min4.index)
                #                             num_counts.append(len(df_min4.index))

                # col_t = pd.DataFrame(tlabels,columns=["t1"])
                # col_phi = pd.DataFrame(philabels,columns=["phi"])
                # df_t_phi = pd.merge(col_t,col_phi,how='cross')
                # col_q2 = pd.DataFrame(qlabels,columns=["Q2"])
                # col_xb = pd.DataFrame(xBlabels,columns=["xB"])
                # df_q2_xb = pd.merge(col_q2,col_xb,how='cross')
                # df = pd.merge(df_t_phi,df_q2_xb,how='cross')
                # df[prefix+'counts'] = num_counts
                # print("Total number of binned events: {}".format(df[prefix+'counts'].sum()))
                # print("Total number of original events: {}".format(orginial_sum))
                # df.to_pickle(dirname+run+"/binned_pickles/"+rec_out_name+prefix+"_reconstructed_events_binned.pkl")
                # ################## END OF OLD WAY

    if DoCombine:
        #runs_list = [runs_list[0]]
        for run in runs_list:

            root_file_list = os.listdir(dirname+run+"/roots/")
            gen_file, recon_file = get_files(dirname+run+"/roots/")

            rec_out_name = recon_file.split(".")[0]
            gen_out_name = gen_file.split(".")[0]

            recon_df_loc = dirname+run+"/binned_pickles/"+rec_out_name+"_reconstructed_events_binned.pkl"
            gen_df_loc = dirname+run+"/binned_pickles/"+rec_out_name+"Gen_reconstructed_events_binned.pkl"

            if os.path.isfile(recon_df_loc) and os.path.isfile(gen_df_loc):
                rec_df = pd.read_pickle(recon_df_loc)
                gen_df = pd.read_pickle(gen_df_loc)

                ic(rec_df)
                ic(gen_df)
                df = pd.concat([rec_df, gen_df['Gencounts']], axis=1)
                ic(df)
                print("Saved to {}".format(dirname+run+"/rec_and_gen_binned_events.pkl"))
                df.to_pickle(dirname+run+"/rec_and_gen_binned_events.pkl")
             
    if DoMetaCombine:
        for index,run in enumerate(runs_list):
            binned_loc = dirname+run+"/rec_and_gen_binned_events.pkl"
            if os.path.isfile(binned_loc):
                df = pd.read_pickle(binned_loc)
                print(df['counts'].sum(),df['Gencounts'].sum())
                if index == 0:
                    df_all = df[["qmin","xmin","tmin","pmin"]]
                    df_all['rec_0'] = df['counts']
                    df_all['gen_0'] = df['Gencounts']
                else:
                    df_all['rec_{}'.format(index)] = df['counts']
                    df_all['gen_{}'.format(index)] = df['Gencounts']

                ic(df_all)
            
        rec_cols = [col for col in df_all if col.startswith('rec')]
        gen_cols = [col for col in df_all if col.startswith('gen')]

        df_all['rec_sum'] = df_all[rec_cols].sum(axis=1)
        df_all['gen_sum'] = df_all[gen_cols].sum(axis=1)

        df_all['acceptance'] = df_all['rec_sum']/df_all['gen_sum']
        df_all['acceptance_err'] = df_all['rec_sum']/df_all['gen_sum']*np.sqrt(1/df_all['rec_sum']+1/df_all['gen_sum'])

        df_all.to_pickle(dirname+"/rec_and_gen_binned_events_meta.pkl")

        ic(df_all)


            


sys.exit()



if DoWipe:
    if not os.path.exists(save_base_dir):
        os.makedirs(save_base_dir)
    else:
        shutil.rmtree(save_base_dir)
        os.makedirs(save_base_dir)

sys.exit()

base_dir = "/mnt/d/GLOBUS/CLAS12/simulations/production/new_100/raw_root/"
#base_dir = "/mnt/d/GLOBUS/CLAS12/simulations/production/Fall_2018_Inbending/Raw_Root_Files/Norad/testana/"










#############!!!!!!!!!!!!!!!!!!
    

if True:
    """-----------------------------------------"""
    
    if base_dir[-1] != '/':
        base_dir += '/'

    save_base_dir =  base_dir + "analyzed/"



    gen_file, recon_file = get_files(base_dir)
    


    # Process gen_all


    # Process recon
    # outname = recon_file.split(".")[0]

    # tree = converter_gen.readFile(base_dir + recon_file)

    # df_gen, df_rec = converter_recon.readEPGG(tree)
    # ic("saving file to: {}".format(save_base_dir+outname+"_reconstructed_events.pkl"))
    # df_rec.to_pickle(save_base_dir+outname+"_reconstructed_events.pkl")
    # df_gen.to_pickle(save_base_dir+outname+"_detected_gen_events.pkl")

    # df_rec2 = pickle_analysis.makeDVpi0vars(df_rec)
    # ic(df_rec2.columns)
    # print("RYIN RECON PLOT")                    
    # histo_plotting.make_all_histos(df_rec2,datatype="Recon",
    #                     hists_2d=True,hists_1d=True,hists_overlap=False,
    #                     saveplots=True,output_dir=save_base_dir+outname+"/")

    # print(df_rec2.columns)
    # print(df_rec2.head(5))
    # print("Number of events: {}".format(df_rec2.shape[0]))



    if DoInspect:
        pkl_name = base_dir + "analyzed/100_20211103_1524_merged_Fall_2018_Inbending_recon_reconstructed_events.pkl"
        #pkl_name = base_dir + "analyzed/5000_20210731_2317_norad_recon_reconstructed_events.pkl"

        df = pd.read_pickle(pkl_name)

        mins = df.min()
        ic(df.GenEtheta.min())#for minz,name in mins:
        ic(df.Etheta.min())#for minz,name in mins:

        

        pkl_name = base_dir + "analyzed/100_20211103_1524_merged_Fall_2018_Inbending_gen_all_generated_events_all_generated_events.pkl"
        #pkl_name = base_dir + "analyzed/5000_20210731_2317_norad_gen_all_generated_events_all_generated_events.pkl"

        df = pd.read_pickle(pkl_name)

        mins = df.min()
        ic(df.GenEtheta.min())#for minz,name in mins:
        df0=df
        #df0 = df.query("GenEtheta<10 and GenW>2")        #    ic(minz,name)

        ic(df0.columns)

        x_data = df0["GenW"]
        plot_title = "F 2018 Inbending, epgg, all exclusivity cuts"

        #plot_title = "F 2018 Inbending, epgg, no exclusivity cuts"

        vars = ["XB (GeV)"]
        make_histos.plot_1dhist(x_data,vars,ranges="none",second_x="none",logger=False,first_label="F18IN",second_label="norad",
                    saveplot=False,pics_dir="none",plot_title=plot_title,first_color="blue",sci_on=False)
            

        y_data = df0["GenQ2"]
        var_names = ["Etheta","Q2"]
        ranges = [[0,10,100],[0,2,100]]
        make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
                    saveplot=False,pics_dir="none",plot_title=plot_title,logger=False,first_label="rad",
                    filename="ExamplePlot",units=["GeV","GeV^2"])


        
        #ic(four_squared)

        #for index, row in four_squared.iterrows():
        #    for q_ind, q_val in enumerat(q2bins):
        #        if q_val>row["Q2"]:
                    
                
        #    print(row['Q2'], row['xB'])



