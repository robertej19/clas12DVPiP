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
import matplotlib as mpl

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

a = False
b = False
c = False
d = False
e = False
f = False
g = False
hh = True
jj = False

cmap = plt.cm.jet  # define the colormap
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# force the first color entry to be grey
cmaplist[0] = (0, 0, 0, 1.0)
cmaplist[1] = (0, 0, 0, 1.0)
cmaplist[2] = (0, 0, 0, 1.0)
cmaplist[3] = (0, 0, 0, 1.0)
#cmaplist[2] = (.5, 0 , 1, .5)
#cmaplist[1] = (0, .5, 1, 1.0)
#cmaplist[3] = (1, 1,0,  1.0)



# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)

def get_gamma(x,q2,BeamE):
    a8p = 1/137*(1/(8*3.14159))
    #print(a8p)
    energies = [BeamE]
    for e in energies:
        y = q2/(2*x*e*mp)
        num = 1-y-q2/(4*e*e)
        denom = 1- y + y*y/2 + q2/(4*e*e)
        #print(y,q2,e,num,denom)
        epsi = num/denom
        gamma = 1/(e*e)*(1/(1-epsi))*(1-x)/(x*x*x)*a8p*q2/(0.938*.938)

    return [gamma, epsi]



if hh:
  
    df_minibin = pd.read_pickle("EnergyDependenceRatio.pkl")
    df_minibin.loc[:,"xmax"] = df_minibin.loc[:,"xmin"]+0.05
    df_minibin.loc[:,"qmax"] = df_minibin.loc[:,"qmin"]+0.5
    df_minibin.loc[:,"lumi6"] =  11922445
    df_minibin.loc[:,"lumi12"] = 16047494


    df_minibin.loc[:,"gamma6"] = get_gamma((df_minibin["xmin"]+df_minibin["xmax"])/2,(df_minibin["qmin"]+df_minibin["qmax"])/2,5.776)[0]
    df_minibin.loc[:,"gamma12"] = get_gamma((df_minibin["xmin"]+df_minibin["xmax"])/2,(df_minibin["qmin"]+df_minibin["qmax"])/2,10.604)[0]
    #dfout.loc[:,"epsi"] = get_gamma((dfout["xmin"]+dfout["xmax"])/2,(dfout["qmin"]+dfout["qmax"])/2)[1]
    df_minibin.loc[:,"counts_low_err"] = np.sqrt(df_minibin.loc[:,"counts_low"])
    df_minibin.loc[:,"counts_high_err"] = np.sqrt(df_minibin.loc[:,"counts_high"])
    
    df_minibin.loc[:,"xsec12"] = df_minibin.loc[:,"counts_high"]/df_minibin.loc[:,"lumi12"]/df_minibin.loc[:,"gamma12"]
    df_minibin.loc[:,"xsec6"] = df_minibin.loc[:,"counts_low"]/df_minibin.loc[:,"lumi6"]/df_minibin.loc[:,"gamma6"]
    df_minibin.loc[:,"xsec_ratio"] = df_minibin.loc[:,"xsec12"]/df_minibin.loc[:,"xsec6"]
    df_minibin.loc[:,"xsec_ratio_err"] = df_minibin.loc[:,"xsec_ratio"]*np.sqrt((df_minibin.loc[:,"counts_high_err"]/df_minibin.loc[:,"counts_high"])**2 + (df_minibin.loc[:,"counts_low_err"]/df_minibin.loc[:,"counts_low"])**2)
    
    print(df_minibin.head(41))
    #sys.exit()

    #q2bins,xBbins, tbins, phibins = [fs.q2bins_test, fs.xBbins_test, fs.tbins_test, fs.phibins_test]
    q2bins,xBbins, tbins, phibins = [fs.q2bins, fs.xBbins, fs.tbins, fs.phibins]


    #q2bins = [2.0,2.5]
    #xBbins = [0.3,0.25]
    for qmin in q2bins[:-1]:
        for xmin in xBbins[:-1]:
            #print(" ON q-{} x-{}".format(qmin, xmin))
            # qmin = 1.5
            # xmin = 0.25
            df = df_minibin.query("qmin==@qmin and xmin==@xmin")
            #for tmin,tmax in zip(tbins[0:-1],tbins[1:]):

            #pave_arr = []

            #for pmin,pmax in zip(phibins[0:-1],phibins[1:]):
            #    pave_arr.append((pmin+pmax)/2)


            for tmin in tbins[0:-1]:
                for pmin,pmax in zip(phibins[0:-1],phibins[1:]):
                    #pave = (pmin+pmax)/2
                    df_t = df.query("tmin==@tmin and pmin==@pmin")
                    #print(df_t)
                    if df_t.shape[0] == 0:
                        #print("APPENDING ZEROS")
                        #dict0 = {'qmin':[qmin],'xmin':[xmin],'tmin':[tmin],'pmin':[pmin],'ratio':['nan'],'ratio_uncert':['nan']}
                        #dict0 = {'qmin':[qmin],'xmin':[xmin],'tmin':[tmin],'pmin':[pmin],'ratio':[0],'ratio_uncert':[0]}
                        dict0 = {'qmin':[qmin],'xmin':[xmin],'tmin':[tmin],'pmin':[pmin],'ratio':[0],'ratio_uncert':[0]}
                        
                        df2 = pd.DataFrame(dict0)
            #            df = pd.concat([df,df2],ignore_index=True)
                        df = df.append(df2)#,ignore_index=True)

            t = []
            p = []
            r = []
            run = []

            # for tmin,tmax in zip(tbins[0:-1],tbins[1:]):
            #     tave = (tmin+tmax)/2
            #     #print(tave)
            #     for pave in pave_arr:
            #         df_t = df.query("tmin==@tmin and pave==@pave")
            #         t.append(tave)
            #         p.append(pave)
            #         rval = df_t.ratio.values[0]
            #         print(tave,pave,rval)
            #         r.append(rval)


            for tind,tmin in enumerate(tbins):
                for pind,pmin in enumerate(phibins):
                    #print(tmin,pmin)
                    if (tind<len(tbins)-1) and (pind<len(phibins)-1):
                        df_t = df.query("tmin==@tmin and pmin==@pmin")
                        #rval = df_t.counts_low.values[0]
                        rval = df_t.xsec_ratio.values[0]
                        runcertval = df_t.xsec_ratio_err.values[0]

                    else:
                        rval = 0
                        runcertval = 0

                    t.append(tmin)
                    p.append(pmin)
                    r.append(rval)
                    run.append(runcertval)


            # for tmin in tbins[0:-1]:
            #     for pmin in phibins[0:-1]:
            #         #print(tmin,pmin)
            #         df_t = df.query("tmin==@tmin and pmin==@pmin")
            #         print(df_t)
            #         #t.append(tmin)
            #         #p.append(pmin)
            #         rval = df_t.ratio.values[0]
            #         #print(tave,pave,rval)
            #         r.append(rval)
            

            #print(t)
            #print(p)
            #print(len(t))
            #print(len(p))
            #print(len(r))



            x = np.reshape(p, (len(tbins), len(phibins)))
            y = np.reshape(t, (len(tbins), len(phibins)))
            z = np.reshape(r, (len(tbins), len(phibins)))
            z = np.ma.masked_where(z==0, z)
            zuncert = np.reshape(run, (len(tbins), len(phibins)))
            print(zuncert)

            cmap = mpl.cm.get_cmap("OrRd").copy()

            cmap.set_bad(color='black')

            #print(x)
            #print(y)
            #print(z)
            fig, ax = plt.subplots(figsize =(36, 17)) 

            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["font.size"] = "20"
            plt.pcolormesh(x,y,z,cmap=cmap)#norm=mpl.colors.LogNorm())


            plt.title("Ratio of Events from 10.6 to 5.7 GeV Electron Beam, Q2 = {}, xB = {}".format(qmin,xmin))
            ax.set_xlabel('Lepton-Hadron Angle')
            ax.set_ylabel('-t (GeV$^2)$')

            plt.colorbar()

            # data = np.random.random((4, 4))

            # fig, ax = plt.subplots()
            # # Using matshow here just because it sets the ticks up nicely. imshow is faster.
            # ax.matshow(data, cmap='seismic')
            # plt.show()
            
            #print(
            #    "RPINTING Z"
            #)
            #print(z.shape)
            #z = z[:-1,:-1]
            #print(z.shape)

            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["font.size"] = "7"

            for (i, j), zz in np.ndenumerate(z[:,:-1]):
                uncert = zuncert[:,:-1][i,j]

                #print(i,j)
                ii = x[i][j]+9
                jj = y[i][j]*1.2
                ax.text(ii, jj, '{:0.2f} $\pm$ {:0.1f}'.format(zz,uncert), ha='center', va='center',
                        bbox=dict(facecolor='white', edgecolor='0.3'))


            plt.ylim([0,2])
            #plt.show()
            #sys.exit()
            plt.savefig("EnergyRats_Jan2022/ratio_q2_{}_xB_{}.png".format(qmin,xmin))
            plt.close()

            #     sys.exit()

            #     plt.contourf(x,y,z, 20, plt.cm.nipy_spectral)
            #     plt.colorbar()
            #     # plt.show()


            #     # ax = fig.add_subplot(111, projection='3d')

            #     # ax.plot_surface(x, y, z)
                
            #     #ax.set_zlabel('CLAS12:CLAS6')
                        

                

            #     # print(df)
            #     # Y = df.tmin
            #     # X = df.pave
            #     # Z = df.ratio
            #     # plt.contourf(X, Y, Z, 20, cmap='RdGy')
            #     # plt.colorbar()
            #     # plt.show()








            





            # #     pd.concat(
            # #     objs,
            # #     axis=0,
            # #     join="outer",
            # #     ignore_index=False,
            # #     keys=None,
            # #     levels=None,
            # #     names=None,
            # #     verify_integrity=False,
            # #     copy=True,
            # # # )
            # #     print(dfout)
            # # sys.exit()


            # # #outname = recon_file.split(".")[0]
            # # #output_loc_event_pkl_after_cuts = dirname+run+"/binned_pickles/"+outname+"_reconstructed_events_after_cuts.pkl"
            # # df = pd.read_pickle(output_loc_event_pkl_after_cuts)
            # # #df = df.query("Q2 > 2 and Q2 < 2.5 and xB < 0.38 and xB>0.3 and t>0.2 and t<0.3")

            # # # print(df.shape)

            # # # x_data = df["phi1"]
            # # # plot_title = "F 2018 Inbending, epgg, all exclusivity cuts"

            # # # #plot_title = "F 2018 Inbending, epgg, no exclusivity cuts"

            # # # vars = ["XB (GeV)"]
            # # # make_histos.plot_1dhist(x_data,vars,ranges="none",second_x="none",logger=False,first_label="F18IN",second_label="norad",
            # # #             saveplot=False,pics_dir="none",plot_title=plot_title,first_color="blue",sci_on=False)

            # # # sys.exit()

            # # df_gen = pd.read_pickle(output_loc_event_pkl_all_gen_events)
            # # #df = pd.read_pickle(save_base_dir+"100_20211103_1524_merged_Fall_2018_Inbending_gen_all_generated_events_all_generated_events.pkl")
            # # for col in df.columns:
            # #     print(col)

            # # df['t1'] = df['t']
            # # orginial_sum = df.shape[0]


if g:
  
    #df_minibin = pd.read_pickle("data_clas6_clas12_ratioed.pkl")
    #df_minibin = pd.read_pickle("data_clas6_clas12_ratioed_Jan2022.pkl")
    df_minibin = pd.read_pickle("data_clas6_clas12_ratioed_Jan2022_nocorrs.pkl")


    print(df_minibin)
    #sys.exit()
    #q2bins,xBbins, tbins, phibins = [fs.q2bins_test, fs.xBbins_test, fs.tbins_test, fs.phibins_test]
    q2bins,xBbins, tbins, phibins = [fs.q2bins, fs.xBbins, fs.tbins, fs.phibins]

    #q2bins = [2.0,2.5]
    #xBbins = [0.3,0.25]
    for qmin in q2bins[:-1]:
        for xmin in xBbins[:-1]:
            print(" ON q-{} x-{}".format(qmin, xmin))
            # qmin = 1.5
            # xmin = 0.25
            df = df_minibin.query("qmin==@qmin and xmin==@xmin")
            #for tmin,tmax in zip(tbins[0:-1],tbins[1:]):

            #pave_arr = []

            #for pmin,pmax in zip(phibins[0:-1],phibins[1:]):
            #    pave_arr.append((pmin+pmax)/2)


            for tmin in tbins[0:-1]:
                for pmin,pmax in zip(phibins[0:-1],phibins[1:]):
                    #pave = (pmin+pmax)/2
                    df_t = df.query("tmin==@tmin and pmin==@pmin")
                    #print(df_t)
                    if df_t.shape[0] == 0:
                        #print("APPENDING ZEROS")
                        #dict0 = {'qmin':[qmin],'xmin':[xmin],'tmin':[tmin],'pmin':[pmin],'ratio':['nan'],'ratio_uncert':['nan']}
                        dict0 = {'qmin':[qmin],'xmin':[xmin],'tmin':[tmin],'pmin':[pmin],'ratio':[0],'ratio_uncert':[0]}
                        df2 = pd.DataFrame(dict0)
            #            df = pd.concat([df,df2],ignore_index=True)
                        df = df.append(df2)#,ignore_index=True)

            t = []
            p = []
            r = []
            run = []


            # for tmin,tmax in zip(tbins[0:-1],tbins[1:]):
            #     tave = (tmin+tmax)/2
            #     #print(tave)
            #     for pave in pave_arr:
            #         df_t = df.query("tmin==@tmin and pave==@pave")
            #         t.append(tave)
            #         p.append(pave)
            #         rval = df_t.ratio.values[0]
            #         print(tave,pave,rval)
            #         r.append(rval)

            for tind,tmin in enumerate(tbins):
                for pind,pmin in enumerate(phibins):
                    #print(tmin,pmin)
                    if (tind<len(tbins)-1) and (pind<len(phibins)-1):
                        df_t = df.query("tmin==@tmin and pmin==@pmin")
                        rval = df_t.ratio.values[0]
                        runcertval = df_t.ratio_uncert.values[0]
                    else:
                        rval = 0
                        runcertval = 0
                    t.append(tmin)
                    p.append(pmin)
                    r.append(rval)
                    run.append(runcertval)

            # for tmin in tbins[0:-1]:
            #     for pmin in phibins[0:-1]:
            #         #print(tmin,pmin)
            #         df_t = df.query("tmin==@tmin and pmin==@pmin")
            #         print(df_t)
            #         #t.append(tmin)
            #         #p.append(pmin)
            #         rval = df_t.ratio.values[0]
            #         #print(tave,pave,rval)
            #         r.append(rval)
            

            
            #zmins = r-run
            #zmaxs = r+run

            colors = []
            delta = 0.4
            for rval,runcert in zip(r,run):
                zmin = rval-runcert
                zmax = rval+runcert
                if (zmin<1+delta and zmax>1-delta):
                   colors.append(1)
                   if rval>5:

                    print("GOOD COLORS")
                    print(rval,runcert,zmin,zmax)

                elif (zmin>1+delta):
                    #print("hi COLORS")
                    #print(rval,zmin,zmax)
    
                    colors.append(3)
                elif (zmax<0.001):
                    #print("low")
                    #print(rval,zmin,zmax)
                    
                    colors.append(0)
                elif (zmax<1-delta):
                    #print("low")
                    #print(rval,zmin,zmax)
                    
                    colors.append(2)
                else:
                    colors.append(float("NaN"))

            #print(t)
            #print(p)
            #print(len(t))
            ##print(len(p))
            #print(len(r))
            #print(len(colors))
            #sys.exit()


            x = np.reshape(p, (len(tbins), len(phibins)))
            y = np.reshape(t, (len(tbins), len(phibins)))
            z = np.reshape(r, (len(tbins), len(phibins)))
            zuncert = np.reshape(run, (len(tbins), len(phibins)))
            colorsx = np.reshape(colors, (len(tbins), len(phibins)))

            



            z = np.ma.masked_where(z==0, z)
            #cmap = mpl.cm.get_cmap("OrRd").copy()
            

            cmap.set_bad(color='black')

            #print(x)
            ##print(y)
            #print(z)
            fig, ax = plt.subplots(figsize =(36, 17)) 

            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["font.size"] = "20"
            plt.pcolormesh(x,y,colorsx,cmap=cmap)#norm=mpl.colors.LogNorm())
            #plt.clim(0,3)

            plt.title("Ratio of CLAS12 to CLAS6 Reduced Cross Sections, Q2 = {}, xB = {}".format(qmin,xmin))
            ax.set_xlabel('Lepton-Hadron Angle')
            ax.set_ylabel('-t (GeV$^2)$')

            #plt.colorbar()

        # data = np.random.random((4, 4))

        # fig, ax = plt.subplots()
        # # Using matshow here just because it sets the ticks up nicely. imshow is faster.
        # ax.matshow(data, cmap='seismic')
        # plt.show()
            
            #print(
            #    "RPINTING Z"
            #)
            #print(z.shape)
            #z = z[:-1,:-1]
            #print(z.shape)

            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["font.size"] = "10"

            for (i, j), zz in np.ndenumerate(z[:,:-1]):
                uncert = zuncert[:,:-1][i,j]

                #print(i,j)
                ii = x[i][j]+9
                jj = y[i][j]*1.2
                ax.text(ii, jj, '{:0.2f} $\pm$ {:0.1f}'.format(zz,uncert), ha='center', va='center',
                        bbox=dict(facecolor='white', edgecolor='0.3'))


            plt.ylim([0,2])
            #plt.show()
            plt.savefig("ratios_over_t_Jan2022_nocorrs/ratio_q2_{}_xB_{}.png".format(qmin,xmin))
            #sys.exit()
            
            plt.close()

        #     sys.exit()

        #     plt.contourf(x,y,z, 20, plt.cm.nipy_spectral)
        #     plt.colorbar()
        #     # plt.show()


        #     # ax = fig.add_subplot(111, projection='3d')

        #     # ax.plot_surface(x, y, z)
            
        #     #ax.set_zlabel('CLAS12:CLAS6')
                    

            

        #     # print(df)
        #     # Y = df.tmin
        #     # X = df.pave
        #     # Z = df.ratio
        #     # plt.contourf(X, Y, Z, 20, cmap='RdGy')
        #     # plt.colorbar()
        #     # plt.show()








            





        # #     pd.concat(
        # #     objs,
        # #     axis=0,
        # #     join="outer",
        # #     ignore_index=False,
        # #     keys=None,
        # #     levels=None,
        # #     names=None,
        # #     verify_integrity=False,
        # #     copy=True,
        # # # )
        # #     print(dfout)
        # # sys.exit()


        # # #outname = recon_file.split(".")[0]
        # # #output_loc_event_pkl_after_cuts = dirname+run+"/binned_pickles/"+outname+"_reconstructed_events_after_cuts.pkl"
        # # df = pd.read_pickle(output_loc_event_pkl_after_cuts)
        # # #df = df.query("Q2 > 2 and Q2 < 2.5 and xB < 0.38 and xB>0.3 and t>0.2 and t<0.3")

        # # # print(df.shape)

        # # # x_data = df["phi1"]
        # # # plot_title = "F 2018 Inbending, epgg, all exclusivity cuts"

        # # # #plot_title = "F 2018 Inbending, epgg, no exclusivity cuts"

        # # # vars = ["XB (GeV)"]
        # # # make_histos.plot_1dhist(x_data,vars,ranges="none",second_x="none",logger=False,first_label="F18IN",second_label="norad",
        # # #             saveplot=False,pics_dir="none",plot_title=plot_title,first_color="blue",sci_on=False)

        # # # sys.exit()

        # # df_gen = pd.read_pickle(output_loc_event_pkl_all_gen_events)
        # # #df = pd.read_pickle(save_base_dir+"100_20211103_1524_merged_Fall_2018_Inbending_gen_all_generated_events_all_generated_events.pkl")
        # # for col in df.columns:
        # #     print(col)

        # # df['t1'] = df['t']
        # # orginial_sum = df.shape[0]


if jj:
  

    df_minibin = pd.read_pickle("EnergyDependenceRatio.pkl")
    df_minibin.loc[:,"xmax"] = df_minibin.loc[:,"xmin"]+0.05
    df_minibin.loc[:,"qmax"] = df_minibin.loc[:,"qmin"]+0.5
    df_minibin.loc[:,"lumi6"] =  11922445
    df_minibin.loc[:,"lumi12"] = 16047494


    df_minibin.loc[:,"gamma6"] = get_gamma((df_minibin["xmin"]+df_minibin["xmax"])/2,(df_minibin["qmin"]+df_minibin["qmax"])/2,5.776)[0]
    df_minibin.loc[:,"gamma12"] = get_gamma((df_minibin["xmin"]+df_minibin["xmax"])/2,(df_minibin["qmin"]+df_minibin["qmax"])/2,10.604)[0]
    #dfout.loc[:,"epsi"] = get_gamma((dfout["xmin"]+dfout["xmax"])/2,(dfout["qmin"]+dfout["qmax"])/2)[1]
    df_minibin.loc[:,"counts_low_err"] = np.sqrt(df_minibin.loc[:,"counts_low"])
    df_minibin.loc[:,"counts_high_err"] = np.sqrt(df_minibin.loc[:,"counts_high"])
    
    df_minibin.loc[:,"xsec12"] = df_minibin.loc[:,"counts_high"]/df_minibin.loc[:,"lumi12"]/df_minibin.loc[:,"gamma12"]
    df_minibin.loc[:,"xsec6"] = df_minibin.loc[:,"counts_low"]/df_minibin.loc[:,"lumi6"]/df_minibin.loc[:,"gamma6"]
    df_minibin.loc[:,"xsec_ratio"] = df_minibin.loc[:,"xsec12"]/df_minibin.loc[:,"xsec6"]
    df_minibin.loc[:,"xsec_ratio_err"] = df_minibin.loc[:,"xsec_ratio"]*np.sqrt((df_minibin.loc[:,"counts_high_err"]/df_minibin.loc[:,"counts_high"])**2 + (df_minibin.loc[:,"counts_low_err"]/df_minibin.loc[:,"counts_low"])**2)
    
    print(df_minibin.head(41))
    #sys.exit()

    #q2bins,xBbins, tbins, phibins = [fs.q2bins_test, fs.xBbins_test, fs.tbins_test, fs.phibins_test]
    q2bins,xBbins, tbins, phibins = [fs.q2bins, fs.xBbins, fs.tbins, fs.phibins]


    #q2bins = [2.0,2.5]
    #xBbins = [0.3,0.25]
    for qmin in q2bins[:-1]:
        for xmin in xBbins[:-1]:
            #print(" ON q-{} x-{}".format(qmin, xmin))
            # qmin = 1.5
            # xmin = 0.25
            df = df_minibin.query("qmin==@qmin and xmin==@xmin")
            #for tmin,tmax in zip(tbins[0:-1],tbins[1:]):

            #pave_arr = []

            #for pmin,pmax in zip(phibins[0:-1],phibins[1:]):
            #    pave_arr.append((pmin+pmax)/2)


            for tmin in tbins[0:-1]:
                for pmin,pmax in zip(phibins[0:-1],phibins[1:]):
                    #pave = (pmin+pmax)/2
                    df_t = df.query("tmin==@tmin and pmin==@pmin")
                    #print(df_t)
                    if df_t.shape[0] == 0:
                        #print("APPENDING ZEROS")
                        #dict0 = {'qmin':[qmin],'xmin':[xmin],'tmin':[tmin],'pmin':[pmin],'ratio':['nan'],'ratio_uncert':['nan']}
                        #dict0 = {'qmin':[qmin],'xmin':[xmin],'tmin':[tmin],'pmin':[pmin],'ratio':[0],'ratio_uncert':[0]}
                        dict0 = {'qmin':[qmin],'xmin':[xmin],'tmin':[tmin],'pmin':[pmin],'ratio':[0],'ratio_uncert':[0]}
                        
                        df2 = pd.DataFrame(dict0)
            #            df = pd.concat([df,df2],ignore_index=True)
                        df = df.append(df2)#,ignore_index=True)

            t = []
            p = []
            r = []
            run = []


            for tind,tmin in enumerate(tbins):
                for pind,pmin in enumerate(phibins):
                    #print(tmin,pmin)
                    if (tind<len(tbins)-1) and (pind<len(phibins)-1):
                        df_t = df.query("tmin==@tmin and pmin==@pmin")
                        #rval = df_t.counts_low.values[0]
                        rval = df_t.xsec_ratio.values[0]
                        runcertval = df_t.xsec_ratio_err.values[0]

                    else:
                        rval = 0
                        runcertval = 0

                    t.append(tmin)
                    p.append(pmin)
                    r.append(rval)
                    run.append(runcertval)


            # x = np.reshape(p, (len(tbins), len(phibins)))
            # y = np.reshape(t, (len(tbins), len(phibins)))
            # z = np.reshape(r, (len(tbins), len(phibins)))
            # z = np.ma.masked_where(z==0, z)
            # zuncert = np.reshape(run, (len(tbins), len(phibins)))
            # print(zuncert)




            colors = []
            delta = 0.4
            for rval,runcert in zip(r,run):
                zmin = rval-runcert
                zmax = rval+runcert
                if (zmin<1+delta and zmax>1-delta):
                   colors.append(1)
                   if rval>5:

                    print("GOOD COLORS")
                    print(rval,runcert,zmin,zmax)

                elif (zmin>1+delta):
                    #print("hi COLORS")
                    #print(rval,zmin,zmax)
    
                    colors.append(3)
                elif (zmax<0.001):
                    #print("low")
                    #print(rval,zmin,zmax)
                    
                    colors.append(0)
                elif (zmax<1-delta):
                    #print("low")
                    #print(rval,zmin,zmax)
                    
                    colors.append(2)
                else:
                    colors.append(float("NaN"))

            #print(t)
            #print(p)
            #print(len(t))
            ##print(len(p))
            #print(len(r))
            #print(len(colors))
            #sys.exit()


            x = np.reshape(p, (len(tbins), len(phibins)))
            y = np.reshape(t, (len(tbins), len(phibins)))
            z = np.reshape(r, (len(tbins), len(phibins)))
            zuncert = np.reshape(run, (len(tbins), len(phibins)))
            colorsx = np.reshape(colors, (len(tbins), len(phibins)))

            



            z = np.ma.masked_where(z==0, z)
            #cmap = mpl.cm.get_cmap("OrRd").copy()
            

            cmap.set_bad(color='black')

            #print(x)
            ##print(y)
            #print(z)
            fig, ax = plt.subplots(figsize =(36, 17)) 

            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["font.size"] = "20"
            plt.pcolormesh(x,y,colorsx,cmap=cmap)#norm=mpl.colors.LogNorm())
            #plt.clim(0,3)

            #plt.title("Ratio of CLAS12 to CLAS6 Reduced Cross Sections, Q2 = {}, xB = {}".format(qmin,xmin))
            plt.title("Ratio of Events from 10.6 to 5.7 GeV Electron Beam, Q2 = {}, xB = {}".format(qmin,xmin))

            ax.set_xlabel('Lepton-Hadron Angle')
            ax.set_ylabel('-t (GeV$^2)$')

            #plt.colorbar()

        # data = np.random.random((4, 4))

        # fig, ax = plt.subplots()
        # # Using matshow here just because it sets the ticks up nicely. imshow is faster.
        # ax.matshow(data, cmap='seismic')
        # plt.show()
            
            #print(
            #    "RPINTING Z"
            #)
            #print(z.shape)
            #z = z[:-1,:-1]
            #print(z.shape)

            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["font.size"] = "10"

            for (i, j), zz in np.ndenumerate(z[:,:-1]):
                uncert = zuncert[:,:-1][i,j]

                #print(i,j)
                ii = x[i][j]+9
                jj = y[i][j]*1.2
                ax.text(ii, jj, '{:0.2f} $\pm$ {:0.1f}'.format(zz,uncert), ha='center', va='center',
                        bbox=dict(facecolor='white', edgecolor='0.3'))


            plt.ylim([0,2])
            #plt.show()
            plt.savefig("EnergyRats_Jan2022/ratio_q2_{}_xB_{}.png".format(qmin,xmin))
            
            #sys.exit()
            
            plt.close()

            # cmap = mpl.cm.get_cmap("OrRd").copy()

            # cmap.set_bad(color='black')

            #print(x)
            #print(y)
            #print(z)
            # fig, ax = plt.subplots(figsize =(36, 17)) 

            # plt.rcParams["font.family"] = "Times New Roman"
            # plt.rcParams["font.size"] = "20"
            # plt.pcolormesh(x,y,z,cmap=cmap)#norm=mpl.colors.LogNorm())


            # ax.set_xlabel('Lepton-Hadron Angle')
            # ax.set_ylabel('-t (GeV$^2)$')

            # plt.colorbar()

            # # data = np.random.random((4, 4))

            # # fig, ax = plt.subplots()
            # # # Using matshow here just because it sets the ticks up nicely. imshow is faster.
            # # ax.matshow(data, cmap='seismic')
            # # plt.show()
            
            # #print(
            # #    "RPINTING Z"
            # #)
            # #print(z.shape)
            # #z = z[:-1,:-1]
            # #print(z.shape)

            # plt.rcParams["font.family"] = "Times New Roman"
            # plt.rcParams["font.size"] = "7"

            # for (i, j), zz in np.ndenumerate(z[:,:-1]):#[0:-5,:-1]):
            #     #print(i,j)
            #     uncert = zuncert[:,:-1][i,j]

            #     ii = x[i][j]+9
            #     jj = y[i][j]*1.2
            #     #ax.text(ii, jj, '{:0.3f}'.format(zz), ha='center', va='center',
            #     #        bbox=dict(facecolor='white', edgecolor='0.3'))

            #     ax.text(ii, jj, '{:0.2f} $\pm$ {:0.2f}'.format(zz,uncert), ha='center', va='center',
            #             bbox=dict(facecolor='white', edgecolor='0.3'))            
            # plt.clim(0.5,1.5)

            # plt.ylim([0,2])
            # #plt.show()
            # #sys.exit()
            # plt.close()

         




    # print(df_minibin)
    # #sys.exit()
    # #q2bins,xBbins, tbins, phibins = [fs.q2bins_test, fs.xBbins_test, fs.tbins_test, fs.phibins_test]
    # q2bins,xBbins, tbins, phibins = [fs.q2bins, fs.xBbins, fs.tbins, fs.phibins]

    # #q2bins = [2.0,2.5]
    # #xBbins = [0.3,0.25]
    # for qmin in q2bins[:-1]:
    #     for xmin in xBbins[:-1]:
    #         print(" ON q-{} x-{}".format(qmin, xmin))
    #         # qmin = 1.5
    #         # xmin = 0.25
    #         df = df_minibin.query("qmin==@qmin and xmin==@xmin")
    #         #for tmin,tmax in zip(tbins[0:-1],tbins[1:]):

    #         #pave_arr = []

    #         #for pmin,pmax in zip(phibins[0:-1],phibins[1:]):
    #         #    pave_arr.append((pmin+pmax)/2)


    #         for tmin in tbins[0:-1]:
    #             for pmin,pmax in zip(phibins[0:-1],phibins[1:]):
    #                 #pave = (pmin+pmax)/2
    #                 df_t = df.query("tmin==@tmin and pmin==@pmin")
    #                 #print(df_t)
    #                 if df_t.shape[0] == 0:
    #                     #print("APPENDING ZEROS")
    #                     #dict0 = {'qmin':[qmin],'xmin':[xmin],'tmin':[tmin],'pmin':[pmin],'ratio':['nan'],'ratio_uncert':['nan']}
    #                     dict0 = {'qmin':[qmin],'xmin':[xmin],'tmin':[tmin],'pmin':[pmin],'ratio':[0],'ratio_uncert':[0]}
    #                     df2 = pd.DataFrame(dict0)
    #         #            df = pd.concat([df,df2],ignore_index=True)
    #                     df = df.append(df2)#,ignore_index=True)

    #         t = []
    #         p = []
    #         r = []
    #         run = []


    #         # for tmin,tmax in zip(tbins[0:-1],tbins[1:]):
    #         #     tave = (tmin+tmax)/2
    #         #     #print(tave)
    #         #     for pave in pave_arr:
    #         #         df_t = df.query("tmin==@tmin and pave==@pave")
    #         #         t.append(tave)
    #         #         p.append(pave)
    #         #         rval = df_t.ratio.values[0]
    #         #         print(tave,pave,rval)
    #         #         r.append(rval)

    #         for tind,tmin in enumerate(tbins):
    #             for pind,pmin in enumerate(phibins):
    #                 #print(tmin,pmin)
    #                 if (tind<len(tbins)-1) and (pind<len(phibins)-1):
    #                     df_t = df.query("tmin==@tmin and pmin==@pmin")
    #                     rval = df_t.ratio.values[0]
    #                     runcertval = df_t.ratio_uncert.values[0]
    #                 else:
    #                     rval = 0
    #                     runcertval = 0
    #                 t.append(tmin)
    #                 p.append(pmin)
    #                 r.append(rval)
    #                 run.append(runcertval)

    #         # for tmin in tbins[0:-1]:
    #         #     for pmin in phibins[0:-1]:
    #         #         #print(tmin,pmin)
    #         #         df_t = df.query("tmin==@tmin and pmin==@pmin")
    #         #         print(df_t)
    #         #         #t.append(tmin)
    #         #         #p.append(pmin)
    #         #         rval = df_t.ratio.values[0]
    #         #         #print(tave,pave,rval)
    #         #         r.append(rval)
            

            
    #         #zmins = r-run
    #         #zmaxs = r+run



        #     sys.exit()

        #     plt.contourf(x,y,z, 20, plt.cm.nipy_spectral)
        #     plt.colorbar()
        #     # plt.show()


        #     # ax = fig.add_subplot(111, projection='3d')

        #     # ax.plot_surface(x, y, z)
            
        #     #ax.set_zlabel('CLAS12:CLAS6')
                    

            

        #     # print(df)
        #     # Y = df.tmin
        #     # X = df.pave
        #     # Z = df.ratio
        #     # plt.contourf(X, Y, Z, 20, cmap='RdGy')
        #     # plt.colorbar()
        #     # plt.show()








            





        # #     pd.concat(
        # #     objs,
        # #     axis=0,
        # #     join="outer",
        # #     ignore_index=False,
        # #     keys=None,
        # #     levels=None,
        # #     names=None,
        # #     verify_integrity=False,
        # #     copy=True,
        # # # )
        # #     print(dfout)
        # # sys.exit()


        # # #outname = recon_file.split(".")[0]
        # # #output_loc_event_pkl_after_cuts = dirname+run+"/binned_pickles/"+outname+"_reconstructed_events_after_cuts.pkl"
        # # df = pd.read_pickle(output_loc_event_pkl_after_cuts)
        # # #df = df.query("Q2 > 2 and Q2 < 2.5 and xB < 0.38 and xB>0.3 and t>0.2 and t<0.3")

        # # # print(df.shape)

        # # # x_data = df["phi1"]
        # # # plot_title = "F 2018 Inbending, epgg, all exclusivity cuts"

        # # # #plot_title = "F 2018 Inbending, epgg, no exclusivity cuts"

        # # # vars = ["XB (GeV)"]
        # # # make_histos.plot_1dhist(x_data,vars,ranges="none",second_x="none",logger=False,first_label="F18IN",second_label="norad",
        # # #             saveplot=False,pics_dir="none",plot_title=plot_title,first_color="blue",sci_on=False)

        # # # sys.exit()

        # # df_gen = pd.read_pickle(output_loc_event_pkl_all_gen_events)
        # # #df = pd.read_pickle(save_base_dir+"100_20211103_1524_merged_Fall_2018_Inbending_gen_all_generated_events_all_generated_events.pkl")
        # # for col in df.columns:
        # #     print(col)

        # # df['t1'] = df['t']
        # # orginial_sum = df.shape[0]
