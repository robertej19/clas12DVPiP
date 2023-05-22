import uproot
import pandas as pd
import numpy as np
import argparse
import itertools
import os, sys
from icecream import ic
import matplotlib
import statistics
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

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import argparse
import sys 
import pandas as pd
from matplotlib.patches import Rectangle

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


def fit_function(x,A,B):
    #A + B*np.cos(2*phi) +C*np.cos(phi)
    #rads = phi*np.pi/180
    #return (A * np.exp(-x/beta) + B * np.exp(-1.0 * (x - mu)**2 / (2 * sigma**2)))
    #A = T+L, B=TT, C=LT
    #A = black, B=blue, C=red
    return A*np.exp(x*B)


df = pd.read_pickle("interactive/dataArrays/full_xsection_outbending_rad_All_All_All_compare_c12_c6_bin_averages.pkl")
#df = pd.read_pickle("/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/struct_funcsoutbending_rad_All_All_All_for_t_q_deps.pkl")

for col in df.columns:
    print(col)


"""
qmin
xmin
tmin
pmin
qave_exp
yave_x
xave_exp
tave_exp
pave_exp
counts_exp
qave_rec
yave_y
xave_rec
tave_rec
pave_rec
counts_rec
qave_gen
yave
xave_gen
tave_gen
pave_gen
counts_gen
q
x
t
p
dsdtdp
stat
sys
qmax
xmax
tmax
pmax
gamma_exp
epsi_exp
binvol
acc_corr
xsec
xsec_corr
xsec_corr_red
xsec_corr_red_nb
xsec_ratio_exp
uncert_counts_exp
uncert_counts_rec
uncert_counts_gen
uncert_xsec
uncert_acc_corr
uncert_xsec_corr_red_nb
uncert_xsec_ratio_exp

"""

#df.loc[:,"ratio"]= df['xsec_corr_red_nb']/df['dsdtdp']
df.replace([np.inf, -np.inf], np.nan, inplace=True)

q2bins,xBbins, tbins, phibins = fs.q2bins[0:8], fs.xBbins[0:12], np.array(fs.tbins[1:]), fs.phibins
#q2bins,xBbins, tbins, phibins = np.array(fs.tbins[0:11]), fs.xBbins[0:12], fs.q2bins[0:8], fs.phibins

#q2bins,xBbins, tbins, phibins = np.array(fs.tbins[0:9]), np.array(fs.xBbins[0:12]) ,np.array(fs.q2bins[0:8]), fs.phibins

#print(df['xsec_corr_red_nb'].query("tmin>3"))
#print(df['xsec_corr_red_nb'].loc[df['tmin'] > 2])


compare_all_bins = False
int_across_phi = False
int_across_phi_with_clas6 = True

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

    xb_dep = []
    xb_6_array = []
    xb_12_array = []
    q2_6_array = []
    q2_12_array = []

    b_values_6 = []
    b_values_12 = []
    b_errs_6 = []
    b_errs_12 = []

    for qmin,qmax in zip(q2bins[0:-1],q2bins[1:]):
        #print(" \n Q2 bin: {} to {}".format(qmin,qmax))
        base_x=[]
        base_x_6=[]

        for xmin,xmax in zip(xBbins[0:-1],xBbins[1:]):
            means_on_t_base = []
            means_on_t_base_6 = []
            t_errs_12 = []
            t_errs_6 = []

            x_6 = []
            x_12 = []

            xb_6 = []
            xb_12 = []
            q2_6 = []
            q2_12 = []
            for tmin,tmax in zip(tbins[0:-1],tbins[1:]):

                query = "qmin == {} and xmin == {} and tmin == {}".format(qmin,xmin,tmin)
                #query = "qmin == {} and xmin == {} and tmin == {}".format(tmin,xmin,qmin)

                print(query)
                df_small = df.query(query)
                #print(df_small['ratio'])
                #print(df_small['ratio'].mean())
                #means_on_t_base.append(df_small['ratio'].mean())


                # print(df_small['xsec_corr_red_nb'].isna().sum())
                #if df_small['xsec_corr_red_nb'].isna().sum() > 3  and df_small['xsec_corr_red_nb'].isna().sum() < 10:
                if df_small['xsec_corr_red_nb'].isna().sum() < 10:


                    #print(df_small['counts_gen'])
                    #print(df_small['xsec_corr_red_nb'])
                    total = df_small['counts_gen'].sum()
                    empty = df_small['counts_gen'].loc[df_small['xsec_corr_red_nb'].isna()].sum()
                    remaining = total-empty
                    eta = remaining/total
                    print(eta)

                    total6 = df_small['dsdtdp'].sum()
                    empty6 = df_small['dsdtdp'].loc[df_small['dsdtdp'].isna()].sum()
                    remaining6 = total6-empty6
                    eta6 = remaining6/total6
                    print(eta6)

                    #eta=1

                    #sys.exit()
                    #print(df_small['xsec_corr_red_nb'])

                    # means_on_t_base.append(df_small['xsec_corr_red_nb'].mean())
                    # means_on_t_base_6.append(df_small['dsdtdp'].mean())
                    # t_errs.append(df_small['uncert_xsec_corr_red_nb'].mean()*1.2)
                    # t_errs_6.append(df_small['stat'].mean()*1.2)

                    ##print("ERRS")
                    ##print(df_small['uncert_xsec_corr_red_nb'].sum())
                    ##print(df_small['stat'].sum())
                    means_on_t_base.append(df_small['xsec_corr_red_nb'].sum()*3.14159/10/eta)
                    #means_on_t_base_6.append(df_small['dsdtdp'].sum()*3.14159/10)
                    t_errs_12.append(df_small['uncert_xsec_corr_red_nb'].sum()*3.14159/10)
                    #t_errs_6.append(df_small['stat'].sum()*3.14159/10)
                    
                    #_6.append(df_small['t'].mean())
                    x_12.append(df_small['tave_exp'].mean())

                    #xb_6.append(df_small['x'].mean())
                    xb_12.append(df_small['xave_exp'].mean())
                    #q2_6.append(df_small['q'].mean())
                    q2_12.append(df_small['qave_exp'].mean())
                # # # means_on_t_base.append(df_small['tel_c12'].mean())
                # # # means_on_t_base_6.append(df_small['telC6'].mean())
                # # # t_errs.append(df_small['mean_uncert_c12'].mean())
                # # # t_errs_6.append(df_small['tel-statC6'].mean())



            #x = tbins[1:]#.tolist()
            #x_6 = np.array(x_6)
            x_12 = np.array(x_12)
            y = np.array(means_on_t_base)#.tolist()
            #y_6 = np.array(means_on_t_base_6)#.tolist()
            #y_err_6 = np.array(t_errs_6)#.tolist()
            y_err_12 = np.array(t_errs_12)#.tolist()



            valid = ~(np.isnan(x_12) | np.isnan(means_on_t_base) | np.isnan(y_err_12))
            #valid2 = ~(np.isnan(x_6) | np.isnan(y_6) | np.isnan(y_err_6))

            #errs_12 = np.ones(len(y_err))*10
            #errs_6 = np.ones(len(y_err_6))*10



            #print(x[valid])
            fit1, fit2 = False,False
            try:
                #print(valid)
                popt, pcov = curve_fit(fit_function, xdata=x_12[valid], ydata=y[valid],
                    sigma=y_err_12[valid], absolute_sigma=True)
                fit1 = True
            except Exception as e:
                pass
                #print(e)
            
            
            
            spare_plot=True
            if fit1:
                a,b = popt[0],popt[1]
                
                #print("ERRORS:")
                #print(y_err_6)
                #print(y_err)

                #print(b)
                if b>-10:
                    spare_plot = False
                    a_err = np.sqrt(pcov[0][0])#*qmod
                    b_err = np.sqrt(pcov[1][1])#*qmod

                    #print(b_err,b_err2)
                    #print(a/a2,b/b2)
                    #print("\n Q2 bin: {} to {}".format(qmin,qmax))
                    #print("xB bin: {} to {}".format(xmin,xmax))
                    #print(b/b2,np.sqrt(b_err*b_err+b_err2*b_err2))
                    #print(b,b_err)
                    #print(b2,b_err2)
                    xb_dep.append((xmin+xmax)/2)
                    
                    xb_12_array.append(np.nanmean(xb_12,axis=0))
                    
                    q2_12_array.append(np.nanmean(q2_12,axis=0))

                    b_values_12.append(-1*b)
                    b_errs_12.append(b_err)


                    xspace = np.linspace(0, 2, 1000)

                    fit_y_data_weighted_12 = fit_function(xspace,a,b)



                    plt.rcParams["font.family"] = "Times New Roman"
                    plt.rcParams["font.size"] = "80"
                    fig, ax = plt.subplots(figsize =(36, 17)) 
                    plt.errorbar(x_12,means_on_t_base,yerr=y_err_12,linestyle="None",marker="v",ms=25,color="blue")#,label="CLAS12")
                    #fit3, = ax.plot(xspace, fit_y_data_weighted_new, color='black', linewidth=2.5, label='New CLAS6 Fit: t+l:{:.0f} tt:{:.0f} lt:{:.0f}'.format(tel,tt,lt))
                    fit4, = ax.plot(xspace, fit_y_data_weighted_12, color='blue', linewidth=6.5, label='CLAS12 B: {} +/- {}'.format(round(-1*b,2),round(b_err,2)))
                    plt.legend()
                    #plt.plot(tbins[1:],means_on_t_base,marker="+",ms=20)
                    #plt.plot(tbins[1:],means_on_t_base_6,marker="+",ms=20)
                    plt.title("T Dependence of Cross Section, Q2 = {}, xB = {}".format(qmin,xmin))
                    ax.set_yscale("log")
                    print(y_err_12)
                    plt.ylim(1e0,5e2)
                    #plt.show()
                    #sys.exit()
                    plt.savefig("tdepfigs/fig_{}_{}.png".format(qmin,xmin))
                    plt.close()
            
            if spare_plot:
                fig, ax = plt.subplots(figsize =(36, 17)) 
                plt.savefig("tdepfigs/fig_{}_{}.png".format(qmin,xmin))
                plt.close()

            
            
            base_x.append(means_on_t_base)
        base_x_q.append(base_x)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "20"
    fig, ax = plt.subplots(figsize =(36, 17)) 
    #plt.errorbar(xb_dep,b_values_6,yerr=b_errs_6,linestyle="None",marker="x",ms=12,color="blue",label="CLAS6")
    #plt.errorbar(xb_dep,b_values_12,yerr=b_errs_12,linestyle="None",marker="x",ms=12,color="red",label="CLAS12") 
    #print(q2_12_array)
    #print(q2_6_array)


    phi = np.linspace(0, 2*np.pi, 60)
    x = np.sin(phi)
    y = np.cos(phi)
    rgb_cycle = np.vstack((            # Three sinusoids
        .5*(1.+np.cos(phi          )), # scaled to [0,1]
        .5*(1.+np.cos(phi+2*np.pi/3)), # 120° phase shifted.
        .5*(1.+np.cos(phi-2*np.pi/3)))).T # Shape = (60,3)
    


    norm_max_12 = max(q2_12_array)
    norm_min_12 = min(q2_12_array)

    norm_max = max((norm_max_12,norm_max_12))
    norm_min = min((norm_min_12,norm_min_12))
    print("MAX MIN IS {} {}".format(norm_max,norm_min))

    colors_12 = []


    import colorsys

    #print(colorsys.rgb_to_hls(222, 42, 42))
    #print(colorsys.rgb_to_hls(42, 222, 99))
    #print(colorsys.rgb_to_hls(51, 42, 222))

    for q_val in q2_12_array:
        #print(int(q_val/norm_max))
        col_new = colorsys.hls_to_rgb(q_val/norm_max,132.0, -0.6870229007633588)
        #print(int(col_new[0]))
        #print(int(col_new[1]))
        #print(int(col_new[2]))
        
        #col_new = (int(col_new[0]),int(col_new[1]),int(col_new[2]*255))
        col_new = (col_new[0]/256,col_new[1]/256,col_new[2]/256)
        colors_12.append(col_new)

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    #cm = plt.cm.get_cmap('RdYlBu')
    cm = ListedColormap(colors_12)

    q2_12_array_normed = q2_12_array/norm_max

    #plt.scatter(xb_6_array,b_values_6,c="red", marker="v", vmin=norm_min, vmax=norm_max, s=195, cmap=cm,label="CLAS6")
    #plt.scatter(xb_12_array,b_values_12,c="blue", marker="o", vmin=norm_min, vmax=norm_max, s=195, cmap=cm,label="CLAS12")
    sc = plt.scatter(xb_12_array,b_values_12,c=q2_12_array, marker="o", vmin=norm_min, vmax=norm_max, s=1, cmap=cm)

    count_12 = 0
    for x,y,err,color in zip(xb_12_array,b_values_12,b_errs_12,colors_12):
        if count_12 == 0:
            ax.errorbar(x,y,yerr=err,linestyle="None",marker="o",ms=15,color=color,label="CLAS12")
        else:
            ax.errorbar(x,y,yerr=err,linestyle="None",marker="o",ms=15,color=color)
        count_12 +=1
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "20"

    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Q$^2$ (GeV$^2$)')#, rotation=270)

    
    #cbar.ax.get_yaxis().set_ticks([])
    #for j, lab in enumerate(['$0$','$1$','$2$','$>3$']):
    #    cbar.ax.text(.5, (2 * j + 1) / 8.0, lab, ha='center', va='center')
    #cbar.ax.get_yaxis().labelpad = 15

    #plt.errorbar(xb_6_array,b_values_6,yerr=b_errs_6,linestyle="None",marker="v",ms=12,c=rgb_cycle,label="CLAS6")
    #plt.errorbar(xb_12_array,b_values_12,yerr=b_errs_12,linestyle="None",marker="o",ms=12,c="blue",label="CLAS12") 
    #plt.plot(tbins[1:],means_on_t_base,marker="+",ms=20)
    #plt.plot(tbins[1:],means_on_t_base_6,marker="+",ms=20)
    plt.title("B Parameter of Exp. Fit to t Dependence of CLAS12 and CLAS6 Cross Sections".format(qmin,xmin))
    plt.xlabel("xB")
    plt.ylim(0.0,2.5)
    plt.ylabel("B fit parameter")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "20"
    plt.xlim
    #ax.set_yscale("log")
    plt.legend()
    plt.show()


    sys.exit()

    q_colors = ['red','orange','yellow','green','blue','purple','black','cyan','magenta','brown','pink','gray','olive','salmon','gold','teal','navy','indigo','maroon','lime','tan','aqua','darkgreen','darkblue','darkcyan','darkmagenta','darkred','darkorange','darkyellow','darkgreen','darkblue','darkpurple','darkcyan','darkmagenta','darkbrown','darkpink','darkgray','darkolive','darksalmon','darkgold','darkteal','darknavy','darkindigo','darkmaroon','darklime','darktan','darkaqua']
    #q_colors = ['black','purple','blue','green','yellow','orange','red']

    fig, ax = plt.subplots(figsize =(36, 17)) 

    q_labels = q2bins[0:-1]
    for q_count, bigarr in enumerate(base_x_q):
        color = q_colors[q_count]
        label = q_labels[q_count]
        legend_counter = 0
        for arr in bigarr:
            print("here")
            print(arr)
            if legend_counter == 0:
                plt.plot(tbins[0:-1],arr,color=color,label="-t: {}".format(label))
                legend_counter += 1
            else:
                plt.plot(tbins[0:-1],arr,color=color)



    ax.set_yscale("log")
    plt.title("T dependence of Cross Section CLAS12 to CLAS6 Reduced Cross Sections")
    plt.xlabel('-t (GeV$^2)$')
    plt.ylabel('Ratio of CLAS12 to CLAS6 Reduced Cross Sections')
    plt.legend()
    plt.show()



if int_across_phi_with_clas6:

    reduced_plot_dir = "Comparison_plots_phi_int/"

    if not os.path.exists(reduced_plot_dir):
        os.makedirs(reduced_plot_dir)

    #q2bins = [2,2.5]
    #xBbins = np.array([0.2,0.25])
    #tbins = np.array([0.09,0.15,0.2])
    phibins = np.array(phibins)

    base_x_q = []
    base_x_q_6 = []

    xb_dep = []
    xb_6_array = []
    xb_12_array = []
    q2_6_array = []
    q2_12_array = []

    b_values_6 = []
    b_values_12 = []
    b_errs_6 = []
    b_errs_12 = []

    for qmin,qmax in zip(q2bins[0:-1],q2bins[1:]):
        #print(" \n Q2 bin: {} to {}".format(qmin,qmax))
        base_x=[]
        base_x_6=[]

        for xmin,xmax in zip(xBbins[0:-1],xBbins[1:]):
            means_on_t_base = []
            means_on_t_base_6 = []
            t_errs_12 = []
            t_errs_6 = []

            x_6 = []
            x_12 = []

            xb_6 = []
            xb_12 = []
            q2_6 = []
            q2_12 = []
            for tmin,tmax in zip(tbins[0:-1],tbins[1:]):

                query = "qmin == {} and xmin == {} and tmin == {}".format(qmin,xmin,tmin)
                #query = "qmin == {} and xmin == {} and tmin == {}".format(tmin,xmin,qmin)

                #print(query)
                df_small = df.query(query)
                #print(df_small['ratio'])
                #print(df_small['ratio'].mean())
                #means_on_t_base.append(df_small['ratio'].mean())


                # print(df_small['xsec_corr_red_nb'].isna().sum())
                #if df_small['xsec_corr_red_nb'].isna().sum() > 3  and df_small['xsec_corr_red_nb'].isna().sum() < 10:
                if df_small['xsec_corr_red_nb'].isna().sum() < 3:

                    #print(df_small)
                    #for col in df_small.columns:
                    #    print(col)

                    #print(df_small['counts_gen'])
                    #print(df_small['xsec_corr_red_nb'])
                    total = df_small['counts_gen'].sum()
                    empty = df_small['counts_gen'].loc[df_small['xsec_corr_red_nb'].isna()].sum()
                    remaining = total-empty
                    eta = remaining/total
                    print(eta)
                    

                    total6 = df_small['counts_gen'].sum()
                    empty6 = df_small['counts_gen'].loc[df_small['dsdtdp'].isna()].sum()
                    remaining6 = total6-empty6
                    eta6 = remaining6/total6
                    print(eta6)
                    eta=1
                    eta6=1
                    #sys.exit()
                    #print(df_small['xsec_corr_red_nb'])

                    # means_on_t_base.append(df_small['xsec_corr_red_nb'].mean())
                    # means_on_t_base_6.append(df_small['dsdtdp'].mean())
                    # t_errs.append(df_small['uncert_xsec_corr_red_nb'].mean()*1.2)
                    # t_errs_6.append(df_small['stat'].mean()*1.2)

                    ##print("ERRS")
                    ##print(df_small['uncert_xsec_corr_red_nb'].sum())
                    ##print(df_small['stat'].sum())
                    means_on_t_base.append(df_small['xsec_corr_red_nb'].sum()*3.14159/10/eta)
                    means_on_t_base_6.append(df_small['dsdtdp'].sum()*3.14159/10/eta6)
                    t_errs_12.append(df_small['uncert_xsec_corr_red_nb'].sum()*3.14159/10)
                    t_errs_6.append(df_small['stat'].sum()*3.14159/10)
                    
                    x_6.append(df_small['t'].mean())
                    x_12.append(df_small['tave_exp'].mean())

                    xb_6.append(df_small['x'].mean())
                    xb_12.append(df_small['xave_exp'].mean())
                    q2_6.append(df_small['q'].mean())
                    q2_12.append(df_small['qave_exp'].mean())
                # # # means_on_t_base.append(df_small['tel_c12'].mean())
                # # # means_on_t_base_6.append(df_small['telC6'].mean())
                # # # t_errs.append(df_small['mean_uncert_c12'].mean())
                # # # t_errs_6.append(df_small['tel-statC6'].mean())



            #x = tbins[1:]#.tolist()
            x_6 = np.array(x_6)
            x_12 = np.array(x_12)
            y = np.array(means_on_t_base)#.tolist()
            y_6 = np.array(means_on_t_base_6)#.tolist()
            y_err_6 = np.array(t_errs_6)#.tolist()
            y_err_12 = np.array(t_errs_12)#.tolist()



            valid = ~(np.isnan(x_12) | np.isnan(means_on_t_base) | np.isnan(y_err_12))
            valid2 = ~(np.isnan(x_6) | np.isnan(y_6) | np.isnan(y_err_6))

            #errs_12 = np.ones(len(y_err))*10
            #errs_6 = np.ones(len(y_err_6))*10



            #print(x[valid])
            fit1, fit2 = False,False
            try:
                #print(valid)
                popt, pcov = curve_fit(fit_function, xdata=x_12[valid], ydata=y[valid],
                    sigma=y_err_12[valid], absolute_sigma=True)
                fit1 = True
            except Exception as e:
                pass
                #print(e)
            
            try:
                popt2, pcov2 = curve_fit(fit_function, xdata=x_6[valid2], ydata=y_6[valid2],
                    sigma=y_err_6[valid2], absolute_sigma=True)
                fit2 = True
            except:
                pass
            
            spare_plot=True
            if fit1 and fit2:
                a,b = popt[0],popt[1]
                a2,b2 = popt2[0],popt2[1]
                #print("ERRORS:")
                #print(y_err_6)
                #print(y_err)

                #print(b)
                if b>-10 and b2>-3:
                    spare_plot = False
                    a_err = np.sqrt(pcov[0][0])#*qmod
                    b_err = np.sqrt(pcov[1][1])#*qmod
                    a_err2 = np.sqrt(pcov2[0][0])#*qmod
                    b_err2 = np.sqrt(pcov2[1][1])#*qmod

                    #print(b_err,b_err2)
                    #print(a/a2,b/b2)
                    #print("\n Q2 bin: {} to {}".format(qmin,qmax))
                    #print("xB bin: {} to {}".format(xmin,xmax))
                    #print(b/b2,np.sqrt(b_err*b_err+b_err2*b_err2))
                    #print(b,b_err)
                    #print(b2,b_err2)
                    xb_dep.append((xmin+xmax)/2)
                    
                    xb_6_array.append(np.nanmean(xb_6,axis=0))
                    xb_12_array.append(np.nanmean(xb_12,axis=0))
                    
                    q2_6_array.append(np.nanmean(q2_6,axis=0))
                    q2_12_array.append(np.nanmean(q2_12,axis=0))

                    b_values_6.append(-1*b2)
                    b_values_12.append(-1*b)
                    b_errs_6.append(b_err2)
                    b_errs_12.append(b_err)


                    xspace = np.linspace(0, 2, 1000)

                    fit_y_data_weighted_12 = fit_function(xspace,a,b)
                    fit_y_data_weighted_6 = fit_function(xspace,a2,b2)



                    plt.rcParams["font.family"] = "Times New Roman"
                    plt.rcParams["font.size"] = "30"
                    fig, ax = plt.subplots(figsize =(36, 17)) 
                    plt.errorbar(x_12,means_on_t_base,yerr=y_err_12,linestyle="None",marker="v",ms=25,color="blue")#,label="CLAS12")
                    plt.errorbar(x_6,means_on_t_base_6,yerr=y_err_6,linestyle="None",marker="o",ms=25,color="red")#,label="CLAS6")  
                    fit2, = ax.plot(xspace, fit_y_data_weighted_6,  color='red', linewidth=6.5, label='CLAS6 B: {} +/- {}'.format(round(-1*b2,2),round(b_err2,2)))
                    #fit3, = ax.plot(xspace, fit_y_data_weighted_new, color='black', linewidth=2.5, label='New CLAS6 Fit: t+l:{:.0f} tt:{:.0f} lt:{:.0f}'.format(tel,tt,lt))
                    fit4, = ax.plot(xspace, fit_y_data_weighted_12, color='blue', linewidth=6.5, label='CLAS12 B: {} +/- {}'.format(round(-1*b,2),round(b_err,2)))
                    plt.legend()
                    plt.xlabel("-t (GeV$^2$)")
                    plt.ylabel("Differential Cross Section $\\frac{d\sigma_U}{dt}$")
                    #plt.plot(tbins[1:],means_on_t_base,marker="+",ms=20)
                    #plt.plot(tbins[1:],means_on_t_base_6,marker="+",ms=20)
                    plt.title("T Dependence of Cross Section, Q2 = {}, xB = {}".format(qmin,xmin))
                    ax.set_yscale("log")
                    print(y_err_12)
                    print(y_err_6)
                    plt.ylim(1e0,5e2)
                    #plt.show()
                    #sys.exit()
                    plt.savefig("tdepfigs/fig_{}_{}.png".format(qmin,xmin))
                    plt.close()
            
            if spare_plot:
                fig, ax = plt.subplots(figsize =(36, 17)) 
                plt.savefig("tdepfigs/fig_{}_{}.png".format(qmin,xmin))
                plt.close()

            
            
            base_x.append(means_on_t_base)
            base_x_6.append(means_on_t_base_6)
        base_x_q.append(base_x)
        base_x_q_6.append(base_x_6)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "20"
    fig, ax = plt.subplots(figsize =(36, 17)) 
    #plt.errorbar(xb_dep,b_values_6,yerr=b_errs_6,linestyle="None",marker="x",ms=12,color="blue",label="CLAS6")
    #plt.errorbar(xb_dep,b_values_12,yerr=b_errs_12,linestyle="None",marker="x",ms=12,color="red",label="CLAS12") 
    #print(q2_12_array)
    #print(q2_6_array)


    phi = np.linspace(0, 2*np.pi, 60)
    x = np.sin(phi)
    y = np.cos(phi)
    rgb_cycle = np.vstack((            # Three sinusoids
        .5*(1.+np.cos(phi          )), # scaled to [0,1]
        .5*(1.+np.cos(phi+2*np.pi/3)), # 120° phase shifted.
        .5*(1.+np.cos(phi-2*np.pi/3)))).T # Shape = (60,3)
    

    rgb_cycle = rgb_cycle[0:len(b_values_6)]

    norm_max_12 = max(q2_12_array)
    norm_max_6 = max(q2_6_array)
    norm_min_12 = min(q2_12_array)
    norm_min_6 = min(q2_6_array)

    norm_max = max((norm_max_6,norm_max_12))
    norm_min = min((norm_min_6,norm_min_12))

    print(q2_6_array/norm_max)

    colors_6 = []
    colors_12 = []


    import colorsys

    #print(colorsys.rgb_to_hls(222, 42, 42))
    #print(colorsys.rgb_to_hls(42, 222, 99))
    #print(colorsys.rgb_to_hls(51, 42, 222))
    for q_val in q2_6_array:
        #print(int(q_val/norm_max))
        col_new = colorsys.hls_to_rgb(q_val/norm_max,132.0, -0.6870229007633588)
        #print(int(col_new[0]))
        #print(int(col_new[1]))
        #print(int(col_new[2]))
        
        #col_new = (int(col_new[0]),int(col_new[1]),int(col_new[2]*255))
        col_new = (col_new[0]/256,col_new[1]/256,col_new[2]/256)
        colors_6.append(col_new)
    
    for q_val in q2_12_array:
        #print(int(q_val/norm_max))
        col_new = colorsys.hls_to_rgb(q_val/norm_max,132.0, -0.6870229007633588)
        #print(int(col_new[0]))
        #print(int(col_new[1]))
        #print(int(col_new[2]))
        
        #col_new = (int(col_new[0]),int(col_new[1]),int(col_new[2]*255))
        col_new = (col_new[0]/256,col_new[1]/256,col_new[2]/256)
        colors_12.append(col_new)

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    #cm = plt.cm.get_cmap('RdYlBu')
    cm = ListedColormap(colors_6)
    count_6 = 0
    for x,y,err,color in zip(xb_6_array,b_values_6,b_errs_6,colors_6):
        if count_6 == 0:
            ax.errorbar(x,y,yerr=err,linestyle="None",marker="v",ms=15,color=color,label="CLAS6")
        else:
            ax.errorbar(x,y,yerr=err,linestyle="None",marker="v",ms=15,color=color)
        count_6 +=1

    q2_12_array_normed = q2_12_array/norm_max
    sc = plt.scatter(xb_6_array,b_values_6,c=q2_12_array, marker="v", vmin=norm_min, vmax=norm_max, s=1, cmap=cm)
    
    count_12 = 0
    for x,y,err,color in zip(xb_12_array,b_values_12,b_errs_12,colors_12):
        if count_12 == 0:
            ax.errorbar(x,y,yerr=err,linestyle="None",marker="o",ms=15,color=color,label="CLAS12")
        else:
            ax.errorbar(x,y,yerr=err,linestyle="None",marker="o",ms=15,color=color)
        count_12 +=1
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "20"
    cbar = plt.colorbar()
    #plt.scatter(xb_6_array,b_values_6,c="red", marker="v", vmin=norm_min, vmax=norm_max, s=195, cmap=cm,label="CLAS6")

    #plt.scatter(xb_12_array,b_values_12,c="blue", marker="o", vmin=norm_min, vmax=norm_max, s=195, cmap=cm,label="CLAS12")
    
    #cbar.ax.get_yaxis().set_ticks([])
    #for j, lab in enumerate(['$0$','$1$','$2$','$>3$']):
    #    cbar.ax.text(.5, (2 * j + 1) / 8.0, lab, ha='center', va='center')
    #cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Q$^2$ (GeV$^2$)')#, rotation=270)

    #plt.errorbar(xb_6_array,b_values_6,yerr=b_errs_6,linestyle="None",marker="v",ms=12,c=rgb_cycle,label="CLAS6")
    #plt.errorbar(xb_12_array,b_values_12,yerr=b_errs_12,linestyle="None",marker="o",ms=12,c="blue",label="CLAS12") 
    #plt.plot(tbins[1:],means_on_t_base,marker="+",ms=20)
    #plt.plot(tbins[1:],means_on_t_base_6,marker="+",ms=20)
    plt.title("B Parameter of Exp. Fit to t Dependence of CLAS12 and CLAS6 Cross Sections".format(qmin,xmin))
    plt.xlabel("xB")
    plt.ylim(0.0,3)
    plt.xlim(0.1,0.6)

    plt.ylabel("B fit parameter")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "20"
    #ax.set_yscale("log")
    plt.legend()
    plt.show()

