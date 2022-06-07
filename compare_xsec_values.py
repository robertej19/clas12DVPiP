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
fs = filestruct.fs()
E = 10.604
cmap = plt.cm.jet  # define the colormap


df = pd.read_pickle("interactive/dataArrays/full_xsection_outbending_rad_All_All_All_compare_c12_c6_bin_averages.pkl")

df.loc[:,"ratio"]= df['xsec_corr_red_nb']/df['dsdtdp']
df.replace([np.inf, -np.inf], np.nan, inplace=True)

#q2bins,xBbins, tbins, phibins = fs.q2bins[0:8], fs.xBbins[0:12], np.array(fs.tbins[0:11]), fs.phibins
q2bins,xBbins, tbins, phibins = np.array(fs.tbins[0:11]), fs.xBbins[0:12], fs.q2bins[0:8], fs.phibins

#q2bins,xBbins, tbins, phibins = np.array(fs.tbins[0:9]), np.array(fs.xBbins[0:12]) ,np.array(fs.q2bins[0:8]), fs.phibins

compare_all_bins = False
int_across_phi = True

if compare_all_bins:


    reduced_plot_dir = "Comparison_plots/"

    if not os.path.exists(reduced_plot_dir):
        os.makedirs(reduced_plot_dir)

    #q2bins = [2,2.5]
    #xBbins = np.array([0.2,0.25])
    #tbins = np.array([0.09,0.15,0.2])
    phibins = np.array(phibins)

    for qmin,qmax in zip(q2bins[0:-1],q2bins[1:]):
        print(" \n Q2 bin: {} to {}".format(qmin,qmax))
        for xmin,xmax in zip(xBbins[0:-1],xBbins[1:]):
            zs = []
            for tmin,tmax in zip(tbins[0:-1],tbins[1:]):

                query = "qmin == {} and xmin == {} and tmin == {}".format(qmin,xmin,tmin)
                #query = "qmin == {} and xmin == {} and tmin == {}".format(tmin,xmin,qmin)

                print(query)
                df_small = df.query(query)
                #print(df_small['xsec_corr_red_nb'])
                #print(df_small['dsdtdp'])


                cmap.set_bad(color='black')

                zs.append(df_small['ratio'].values)
                #zs.append(df_small['xsec_corr_red_nb'].values)

                # z = df_small['ratio'].values
                # z = np.expand_dims(z, axis=0)  # or axis=1
                # print(z)

                # x = phibins
                # y = tbins
                # fig, ax = plt.subplots(figsize =(36, 17)) 

                # print(x.size)
                # print(y.size)
                # print(z.size)

                # plt.rcParams["font.family"] = "Times New Roman"
                # plt.rcParams["font.size"] = "20"


                # plt.pcolormesh(x,y,z)#,cmap=cmap)#norm=mpl.colors.LogNorm())
                # #plt.imshow(z,interpolation='none')

                # plt.title("Ratio of CLAS12 to CLAS6 Reduced Cross Sections, Q2 = {}, xB = {}".format(qmin,xmin))
                # ax.set_xlabel('Lepton-Hadron Angle')
                # ax.set_ylabel('-t (GeV$^2)$')

                # plt.colorbar()

                # plt.show()

            #z = np.expand_dims(zs, axis=0)  # or axis=1
            z = zs
            print(z)

            x = phibins
            y = tbins
            fig, ax = plt.subplots(figsize =(36, 17)) 

            print(x.size)
            print(y.size)

            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["font.size"] = "20"

            vmin,vmax = 0.5,1.5
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

            plt.pcolormesh(x,y,z,norm=norm)#,cmap=cmap)#norm=mpl.colors.LogNorm())
            #plt.imshow(z,interpolation='none')

            plt.title("Ratio of CLAS12 to CLAS6 Reduced Cross Sections, Q2 = {}, xB = {}".format(qmin,xmin))
            ax.set_xlabel('Lepton-Hadron Angle')
            ax.set_ylabel('-t (GeV$^2)$')

            plt.colorbar()

            plt.savefig(reduced_plot_dir+"ratio_q2_{}_xB_{}.png".format(qmin,xmin))
            plt.close()



if int_across_phi:

    reduced_plot_dir = "Comparison_plots_phi_int/"

    if not os.path.exists(reduced_plot_dir):
        os.makedirs(reduced_plot_dir)

    #q2bins = [2,2.5]
    #xBbins = np.array([0.2,0.25])
    #tbins = np.array([0.09,0.15,0.2])
    phibins = np.array(phibins)

    base_x_q = []
    base_x_q_6 = []

    for qmin,qmax in zip(q2bins[0:-1],q2bins[1:]):
        print(" \n Q2 bin: {} to {}".format(qmin,qmax))
        base_x=[]
        base_x_6=[]

        for xmin,xmax in zip(xBbins[0:-1],xBbins[1:]):
            means_on_t_base = []
            means_on_t_base_6 = []

            for tmin,tmax in zip(tbins[0:-1],tbins[1:]):

                #query = "qmin == {} and xmin == {} and tmin == {}".format(qmin,xmin,tmin)
                query = "qmin == {} and xmin == {} and tmin == {}".format(tmin,xmin,qmin)

                #print(query)
                df_small = df.query(query)
                #print(df_small['ratio'])
                #print(df_small['ratio'].mean())
                #means_on_t_base.append(df_small['ratio'].mean())
                means_on_t_base.append(df_small['xsec_corr_red_nb'].mean())
                means_on_t_base_6.append(df_small['dsdtdp'].mean())

            base_x.append(means_on_t_base)
            base_x_6.append(means_on_t_base_6)
        base_x_q.append(base_x)
        base_x_q_6.append(base_x_6)


    q_colors = ['red','orange','yellow','green','blue','purple','black','cyan','magenta','brown','pink','gray','olive','salmon','gold','teal','navy','indigo','maroon','lime','tan','aqua','darkgreen','darkblue','darkcyan','darkmagenta','darkred','darkorange','darkyellow','darkgreen','darkblue','darkpurple','darkcyan','darkmagenta','darkbrown','darkpink','darkgray','darkolive','darksalmon','darkgold','darkteal','darknavy','darkindigo','darkmaroon','darklime','darktan','darkaqua']
    #q_colors = ['black','purple','blue','green','yellow','orange','red']

    # # q_labels = q2bins[0:-1]
    # # for q_count, bigarr in enumerate(base_x_q):
    # #     color = q_colors[q_count]
    # #     label = q_labels[q_count]
    # #     legend_counter = 0
    # #     for arr in bigarr:
    # #         print("here")
    # #         print(arr)
    # #         if legend_counter == 0:
    # #             plt.plot(tbins[0:-1],arr,color=color,label="-t: {}".format(label))
    # #             legend_counter += 1
    # #         else:
    # #             plt.plot(tbins[0:-1],arr,color=color)

    #FOR integraton over xB
    q_labels = q2bins[0:-1]

    fig, ax = plt.subplots(figsize =(36, 17)) 

    for q_count, bigarr in enumerate(base_x_q):
        bigarr_6 = base_x_q_6[q_count]
        color = q_colors[q_count]
        label = q_labels[q_count]
        legend_counter = 0
        print('HEREE')
        print(bigarr)
        arr = np.nanmean(bigarr,axis=0)
        arr_6 = np.nanmean(bigarr_6,axis=0)

        print(arr)
        if legend_counter == 0:
            plt.plot(tbins[0:-1],arr,color=color,label="Q$^2$: {}".format(label))
            plt.plot(tbins[0:-1],arr_6,color=color,linestyle='--',)#,label="Q$^2_6$: {}".format(label))

            legend_counter += 1
        else:
            plt.plot(tbins[0:-1],arr,color=color)
            plt.plot(tbins[0:-1],arr_6,color=color,linestyle='--',)


    #FOR integraton over everything but 1
    # q_labels = q2bins[0:-1]

    # #for q_count, bigarr in enumerate(base_x_q):

    # arr1 = np.nanmean(base_x_q,axis=0)

    # arr = np.nanmean(arr1,axis=0)
    # print(arr)

    # q_count = 0
    # color = q_colors[q_count]
    # label = q_labels[q_count]

    # legend_counter = 1
    # print('HEREE')
   
    # print(arr)
    # if legend_counter == 0:
    #     plt.plot(tbins[0:-1],arr,color=color,label="-q: {}".format(label))
    #     legend_counter += 1
    # else:
    #     plt.plot(tbins[0:-1],arr,color=color)



    ax.set_yscale("log")
    plt.title("T dependence of Cross Section CLAS12 to CLAS6 Reduced Cross Sections")
    plt.xlabel('-t (GeV$^2)$')
    plt.ylabel('Ratio of CLAS12 to CLAS6 Reduced Cross Sections')
    plt.legend()
    plt.show()
                #print(df_small['dsdtdp'])


                #cmap.set_bad(color='black')

               # zs.append(df_small['ratio'].values)