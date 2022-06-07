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

# 1.) Necessary imports.    
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
E = 10.6
Clas6_Sim_BeamTime = 11922445
Clas12_Sim_BeamTime = 16047494
Clas12_exp_luminosity = 5.5E40
fs = filestruct.fs()


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

# 2.) Define fit function.
def fit_function(phi,A,B,C):
    #A + B*np.cos(2*phi) +C*np.cos(phi)
    rads = phi*np.pi/180
    #return (A * np.exp(-x/beta) + B * np.exp(-1.0 * (x - mu)**2 / (2 * sigma**2)))
    #A = T+L, B=TT, C=LT
    #A = black, B=blue, C=red
    return A + B*np.cos(2*rads) + C*np.cos(rads)

df = pd.read_pickle("full_xsection.pkl")

"""
Index(['qmin', 'xmin', 'tmin', 'pmin', 'qave_exp', 'xave_exp', 'tave_exp',
       'pave_exp', 'counts_exp', 'qave_rec', 'xave_rec', 'tave_rec',
       'pave_rec', 'counts_rec', 'qave_gen', 'xave_gen', 'tave_gen',
       'pave_gen', 'counts_gen', 'counts_10600GeV', 'counts_5776GeV',
       'Energy_ratio', 'q', 'x', 't', 'p', 'dsdtdp', 'stat', 'sys', 'qmax',
       'xmax', 'tmax', 'pmax', 'gamma_exp', 'epsi_exp', 'gamma6_sim',
       'gamma12_sim', 'xsec_sim_12', 'xsec_sim_6', 'xsec_ratio_sim', 'binvol',
       'acc_corr', 'xsec', 'xsec_corr', 'xsec_corr_red', 'xsec_corr_red_nb',
       'xsec_ratio_exp', 'xsec_ratio_exp_corr', 'uncert_counts_exp',
       'uncert_counts_rec', 'uncert_counts_gen', 'uncert_counts_10600GeV',
       'uncert_counts_5776GeV', 'uncert_xsec', 'uncert_acc_corr',
       'uncert_xsec_corr_red_nb', 'uncert_xsec_ratio_exp',
       'uncert_xsec_ratio_exp_corr'],
      dtype='object')

      	self.xBbins = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7,0.85,1]
		self.q2bins =  [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,7.0,9.0,14.0]
		self.tbins =  [0.09,0.15,0.2,0.3,0.4,0.6,1,1.5,2,3,4.5,6,12]

xsec_corr
gamma_exp
epsi_exp

"""
df.loc[:,"xsec_corr_nb_gamma"] = df["xsec_corr"]*1E33/df["gamma_exp"]

df.loc[:,"uncert_xsec_corr_nb_gamma"] = np.sqrt(  np.square(df["uncert_xsec"]/df["xsec"]) + np.square(df["uncert_acc_corr"]/df["acc_corr"]))*df["xsec_corr_nb_gamma"]

df = df[df.acc_corr > 0.01]

df = df.query("qmin == 3 and xmin == 0.4 and tmin == 0.6")
df = df[["acc_corr","pave_exp","gamma_exp","epsi_exp","xsec_corr_nb_gamma","uncert_xsec_corr_nb_gamma"]]


print(df)


fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(df["pave_exp"],df["xsec_corr_nb_gamma"])

epsi_mean = df["epsi_exp"].mean()



binscenters = df["pave_exp"]
data_entries = df["xsec_corr_nb_gamma"]

popt, pcov = curve_fit(fit_function, xdata=binscenters[data_entries>0], ydata=data_entries[data_entries>0], p0=[2.0, -1, -1],
                    sigma=df["uncert_xsec_corr_nb_gamma"], absolute_sigma=True)

a,b,c = popt[0],popt[1],popt[2]
overal_normalization_factor = 2
#ic(a_err,b_err,c_err)
#ic.disable()

tel = a*6.28*overal_normalization_factor
tt = b/epsi_mean*6.28*overal_normalization_factor
lt = c/np.sqrt(2*epsi_mean*(1+epsi_mean))*6.28*overal_normalization_factor

print(tel,tt,lt)



# 6.)
# Generate enough x values to make the curves look smooth.

#fit_y_data_1 = fit_function(binscenters, *popt)
xmax = 360
xspace = np.linspace(0, xmax, 1000)
fit_y_data = fit_function(xspace, *popt)

#ic(fit_y_data_1)

fit1, = ax.plot(xspace, fit_y_data, color='darkorange', linewidth=2.5, label='Fitted function')
plt.show()

sys.exit()






#print(fit_function(45,1,0,1))

def getPhiFit_prebinned(xval,qval,phi_bins,bin_counts,phi_title,plot_dir,args,bin_corr_fact,bin_corr_fact_uncert):
    ic.disable()
    if args.v:
        ic.enable() 

    xmin = 0
    xmax = 360
    #print("fitting {}".format(phi_title))
    
    data_entries_uncorrected = bin_counts
    data_entries = bin_corr_fact*data_entries_uncorrected
    data_entries_err = bin_corr_fact_uncert

    bins = phi_bins
    #ic.enable()
    ic(bins)

    data_errors = np.sqrt(data_entries)
    data_errors = [1/err if err>0 else err+1 for err in data_errors]
    
    total_data_errors = np.sqrt(np.square(data_entries_err)+np.square(data_errors))

    
    #print(data_entries)

    #ic.enable()
    ic(data_entries)
    if (max(data_entries) == 0):
        #print("No data in this plot, saving and returning 0")

        plt.text(150, 0, "No Data")
        #print("No data")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        #binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])

        #bar1 = ax.bar(binscenters, data_entries, width=bins[1] - bins[0], color='navy', label='Histogram entries')

        #plt.hist(phi_vals, bins =np.linspace(0, 360, 20), range=[0,360])# cmap = plt.cm.nipy_spectral)

        plot_title = plot_dir + phi_title+".png"
        plt.savefig(plot_title)
        plt.close()
        #print("plot saved to {}".format(plot_title))
        
        return ["nofit","nofit","nofit","nofit"]
    else:
        ic(bins)
        bins = np.append(bins,360)
        binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
        #ic.enable()
        ic(binscenters)
        #sys.exit()
        ic.disable()
        ic(binscenters)
        ic(data_entries)
        # 5.) Fit the function to the histogram data.
        popt_0, pcov = curve_fit(fit_function, xdata=binscenters, ydata=data_entries, p0=[2.0, 2, 0.3],
                    sigma=total_data_errors, absolute_sigma=True)

        popt, pcov = curve_fit(fit_function, xdata=binscenters, ydata=data_entries, p0=[popt_0[0],popt_0[1],popt_0[2]],
                    sigma=total_data_errors, absolute_sigma=True)
        #print(popt) #popt contains the values for A, B, C

        

        a,b,c = popt[0],popt[1],popt[2]
        #ic(a_err,b_err,c_err)
        #ic.disable()
        
        # 6.)
        # Generate enough x values to make the curves look smooth.
       
        fit_y_data_1 = fit_function(binscenters, *popt)

        ic(fit_y_data_1)

        

        chisq0 = stats.chisquare(f_obs=data_entries, f_exp=fit_y_data_1)
        chisq = stats.chisquare(f_obs=np.array(data_entries, dtype=np.float64), f_exp=np.array(fit_y_data_1, dtype=np.float64))

        qmod = 1
        amod = np.sqrt(np.square(chisq[0]))/20

        #ic.enable()
        ic(amod)
        if amod > 1:
            qmod = amod
        
        a_err = np.sqrt(pcov[0][0])*qmod
        b_err = np.sqrt(pcov[1][1])*qmod
        c_err = np.sqrt(pcov[2][2])*qmod


        sums=[]
        for ind,val in enumerate(fit_y_data_1):
            diff2 = (data_entries[ind]-val)**2
            s1 = diff2/val
            sums.append(s1)

        manchisq = np.sum(sums)

        ###ic.enable()
        if chisq0[0]<0:
            ic(manchisq)
            ic(chisq0[0])
        if not (chisq0[0] == chisq[0]):
            print("ERROR MISMATCH")
            print(chisq0[0])
            print(chisq[0])
            print(manchisq)

        ic.disable()

        p = chisq[1]
        chisq = chisq[0]

        ic(chisq)
        ic(p)


        xspace = np.linspace(0, xmax, 1000)
        fit_y_data = fit_function(xspace, *popt)

        ##ic.enable()
        ic(fit_y_data)
        
        y_manual = []
        for ind, val in enumerate(xspace):
            ic(val,a,b,c)
            y_one = fit_function(val,a,b,c)
            ic(y_one)
            y_manual.append(y_one)


        
        #7
        # Plot the histogram and the fitted function.

        fig = plt.figure()
        ax = fig.add_subplot(111)
        

        bar1 = ax.bar(binscenters, data_entries, width=bins[1] - bins[0], color='navy', label='Histogram entries')
        fit1, = ax.plot(xspace, fit_y_data, color='darkorange', linewidth=2.5, label='Fitted function')

        # Make the plot nicer.
        plt.xlim(xmin,xmax)
        #plt.ylim(0,5)
        plt.xlabel(r'phi')
        plt.ylabel(r'Number of entries')

        plot_title = plot_dir + phi_title+".png"
        plt.title(phi_title)
        #plt.legend(loc='best')

        fit_params = "A: {:2.2f} +/- {:2.2f}\n B:{:2.2f} +/- {:2.2f}\n C:{:2.2f} +/- {:2.2f}\n Chi:{:2.2f} \n p:{:2.2f}".format(a,a_err,b,b_err,c,c_err,chisq,p)

        extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        ax.legend([bar1, fit1, extra], ("Data","A+Bcos(2Phi)+Ccos(Phi)",fit_params))

        #plt.text(120, max(data_entries)/1.3, fit_params)

        
        plt.savefig(plot_title)
        #plt.show()
        if xval == 0.38:
               if qval == 2.5:
                   plt.show()

        plt.close()
        #print("plot saved to {}".format(plot_title))

        return popt, pcov, chisq, p

def plotPhi_duo(phi_bins,bin_counts_0,bin_counts_1,phi_title,plot_dir,args,saveplot=True):
    ic.disable()
    if args.v:
        ic.enable() 

    xmin = 0
    xmax = 360
    #print("fitting {}".format(phi_title))
    
    data_entries_0 = bin_counts_0
    data_entries_1 = bin_counts_1
    bins = phi_bins

    data_errors_0 = np.sqrt(data_entries_0)
    data_errors_0 = [1/err if err>0 else err+1 for err in data_errors_0]

    data_errors_1 = np.sqrt(data_entries_1)
    data_errors_1 = [1/err if err>0 else err+1 for err in data_errors_1]


    #print(data_entries)

    if (max(data_entries_0) == 0) and(max(data_entries_1) == 0):
        #print("No data in this plot, saving and returning 0")

        plt.text(150, 0, "No Data")
        #print("No data")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])

        bar1 = ax.bar(binscenters, data_entries_0, width=bins[1] - bins[0], color='navy', label='Histogram entries')

        #plt.hist(phi_vals, bins =np.linspace(0, 360, 20), range=[0,360])# cmap = plt.cm.nipy_spectral)

        plot_title = plot_dir + phi_title+".png"
        if saveplot:
            plt.savefig(plot_title)
            plt.close()
        else:
            plt.show()
            plt.close()

        return ["nofit","nofit","nofit","nofit"]
    else:
        ic(bins)
        binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])

        ic(binscenters)
        # 5.) Fit the function to the histogram data.
        #popt, pcov = curve_fit(fit_function, xdata=binscenters, ydata=data_entries, p0=[2.0, 2, 0.3],
         #           sigma=data_errors, absolute_sigma=True)
        #print(popt) #popt contains the values for A, B, C

        ##a_err = np.sqrt(pcov[0][0])
        ##b_err = np.sqrt(pcov[1][1])
        #c_err = np.sqrt(pcov[2][2])

        #a,b,c = popt[0],popt[1],popt[2]
        #ic(a_err,b_err,c_err)
        #ic.disable()
        
        # 6.)
        # Generate enough x values to make the curves look smooth.
       
        #fit_y_data_1 = fit_function(binscenters, *popt)

        #ic(fit_y_data_1)

        

        #chisq0 = stats.chisquare(f_obs=data_entries, f_exp=fit_y_data_1)
        #chisq = stats.chisquare(f_obs=np.array(data_entries, dtype=np.float64), f_exp=np.array(fit_y_data_1, dtype=np.float64))

        #sums=[]
        #for ind,val in enumerate(fit_y_data_1):
        #    diff2 = (data_entries[ind]-val)**2
        #    s1 = diff2/val
        #    sums.append(s1)

       # manchisq = np.sum(sums)

        ###ic.enable()
        #if chisq0[0]<0:
        #    ic(manchisq)
        #    ic(chisq0[0])
        #if not (chisq0[0] == chisq[0]):
        #    print("ERROR MISMATCH")
        #    print(chisq0[0])
        #    print(chisq[0])
        #    print(manchisq)

        ic.disable()

       # p = chisq[1]
       # chisq = chisq[0]

        ##ic(chisq)
        #ic(p)


        #xspace = np.linspace(0, xmax, 1000)
        #fit_y_data = fit_function(xspace, *popt)

        ##ic.enable()
        #ic(fit_y_data)
        
        #y_manual = []
        #for ind, val in enumerate(xspace):
        #    ic(val,a,b,c)
        #    y_one = fit_function(val,a,b,c)
        #    ic(y_one)
        #    y_manual.append(y_one)


        
        #7
        # Plot the histogram and the fitted function.

        fig = plt.figure()
        ax = fig.add_subplot(111)
        


        highPower = data_entries_0
        lowPower = data_entries_1


        #plt.bar(binscenters, highPower,  
        #        color='b', label='LUND Events')
        #plt.bar(binscenters,  lowPower, color='r', alpha=0.5, label='Sim Events')




        #ic.enable()
        #ic(binscenters)
        #ic(data_entries_0)
        #ic(data_entries_1)
        bar0 = ax.bar(binscenters, data_entries_1, width=bins[1] - bins[0], color='red', label='Raw Counts')
        bar1 = ax.bar(binscenters, data_entries_0, width=bins[1] - bins[0], color='black', label='With Acceptance Corr.')
       # fit1, = ax.plot(xspace, fit_y_data, color='darkorange', linewidth=2.5, label='Fitted function')

        # Make the plot nicer.
        plt.xlim(xmin,xmax)
        #plt.ylim(0,5)
        plt.xlabel(r'phi')
        plt.ylabel(r'Number of entries')

        plot_title = plot_dir + phi_title+".png"
        plt.title(phi_title)
        #plt.legend(loc='best')
        plt.legend()


        #fit_params = "A: {:2.2f} +/- {:2.2f}\n B:{:2.2f} +/- {:2.2f}\n C:{:2.2f} +/- {:2.2f}\n Chi:{:2.2f} \n p:{:2.2f}".format(a,a_err,b,b_err,c,c_err,chisq,p)


        #plt.text(120, max(data_entries)/1.3, fit_params)

        
        if saveplot:
            plt.savefig(plot_title)
            plt.close()
        else:
            #plt.show()
            plt.close()
        #print("plot saved to {}".format(plot_title))

def plotPhi_single(phi_bins,bin_counts_0,phi_title,plot_dir,args,saveplot=True):
    ic.disable()
    if args.v:
        ic.enable() 

    xmin = 0
    xmax = 360
    #print("fitting {}".format(phi_title))
    
    data_entries_0 = bin_counts_0
    bins = phi_bins

    data_errors_0 = np.sqrt(data_entries_0)
    data_errors_0 = [1/err if err>0 else err+1 for err in data_errors_0]


    #print(data_entries)

    if (max(data_entries_0) == 0):
        #print("No data in this plot, saving and returning 0")

        plt.text(150, 0, "No Data")
        #print("No data")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])[0]

        bar1 = ax.bar(binscenters, data_entries_0, width=bins[1] - bins[0], color='navy', label='Histogram entries')

        #plt.hist(phi_vals, bins =np.linspace(0, 360, 20), range=[0,360])# cmap = plt.cm.nipy_spectral)

        plot_title = plot_dir + phi_title+".png"
        if saveplot:
            plt.savefig(plot_title)
            plt.close()
        else:
            plt.show()
            plt.close()

        return ["nofit","nofit","nofit","nofit"]
    else:
        ic.disable()
        bins.append(360)
        ic(bins)
        binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(0,len(bins)-1)])

        ic(binscenters)
        # 5.) Fit the function to the histogram data.
        #popt, pcov = curve_fit(fit_function, xdata=binscenters, ydata=data_entries, p0=[2.0, 2, 0.3],
         #           sigma=data_errors, absolute_sigma=True)
        #print(popt) #popt contains the values for A, B, C

        ##a_err = np.sqrt(pcov[0][0])
        ##b_err = np.sqrt(pcov[1][1])
        #c_err = np.sqrt(pcov[2][2])

        #a,b,c = popt[0],popt[1],popt[2]
        #ic(a_err,b_err,c_err)
        #ic.disable()
        
        # 6.)
        # Generate enough x values to make the curves look smooth.
       
        #fit_y_data_1 = fit_function(binscenters, *popt)

        #ic(fit_y_data_1)

        

        #chisq0 = stats.chisquare(f_obs=data_entries, f_exp=fit_y_data_1)
        #chisq = stats.chisquare(f_obs=np.array(data_entries, dtype=np.float64), f_exp=np.array(fit_y_data_1, dtype=np.float64))

        #sums=[]
        #for ind,val in enumerate(fit_y_data_1):
        #    diff2 = (data_entries[ind]-val)**2
        #    s1 = diff2/val
        #    sums.append(s1)

       # manchisq = np.sum(sums)

        ###ic.enable()
        #if chisq0[0]<0:
        #    ic(manchisq)
        #    ic(chisq0[0])
        #if not (chisq0[0] == chisq[0]):
        #    print("ERROR MISMATCH")
        #    print(chisq0[0])
        #    print(chisq[0])
        #    print(manchisq)

        ic.disable()

       # p = chisq[1]
       # chisq = chisq[0]

        ##ic(chisq)
        #ic(p)


        #xspace = np.linspace(0, xmax, 1000)
        #fit_y_data = fit_function(xspace, *popt)

        ##ic.enable()
        #ic(fit_y_data)
        
        #y_manual = []
        #for ind, val in enumerate(xspace):
        #    ic(val,a,b,c)
        #    y_one = fit_function(val,a,b,c)
        #    ic(y_one)
        #    y_manual.append(y_one)


        
        #7
        # Plot the histogram and the fitted function.

        fig = plt.figure()
        ax = fig.add_subplot(111)
        




        #plt.bar(binscenters, highPower,  
        #        color='b', label='LUND Events')
        #plt.bar(binscenters,  lowPower, color='r', alpha=0.5, label='Sim Events')



        #ic.enable()
        ic(binscenters)
        ic(np.array(data_entries_0))

        bar0 = ax.bar(binscenters, np.array(data_entries_0), width=bins[1] - bins[0], color='red', label='Sim input')
       # fit1, = ax.plot(xspace, fit_y_data, color='darkorange', linewidth=2.5, label='Fitted function')

        # Make the plot nicer.
        plt.xlim(xmin,xmax)
        #plt.ylim(0,5)
        plt.xlabel(r'phi')
        plt.ylabel(r'Number of entries')

        plot_title = plot_dir + phi_title+".png"
        plt.title(phi_title)
        #plt.legend(loc='best')
        plt.legend()


        #fit_params = "A: {:2.2f} +/- {:2.2f}\n B:{:2.2f} +/- {:2.2f}\n C:{:2.2f} +/- {:2.2f}\n Chi:{:2.2f} \n p:{:2.2f}".format(a,a_err,b,b_err,c,c_err,chisq,p)


        #plt.text(120, max(data_entries)/1.3, fit_params)

        
        if saveplot:
            #plt.show()
            plt.savefig(plot_title)
            plt.close()
        else:
            plt.show()
            plt.close()
        #print("plot saved to {}".format(plot_title))


def getPhiFit(phi_vals,phi_title,plot_dir,args):
    ic.disable()
    if args.v:
        ic.enable() 

    xmin = 0
    xmax = 360
    #print("fitting {}".format(phi_title))
    
    data = phi_vals
    bins_x = np.linspace(xmin, xmax, 20)
    data_entries, bins = np.histogram(data,bins=bins_x)
    data_errors = np.sqrt(data_entries)
    data_errors = [1/err if err>0 else err+1 for err in data_errors]
    
    ic(data_entries)
    ic(data_errors)

    

    if (max(data_entries) == 0):
        #print("No data in this plot, saving and returning 0")

        plt.text(150, 0, "No Data")

        plt.hist(phi_vals, bins =np.linspace(0, 360, 20), range=[0,360])# cmap = plt.cm.nipy_spectral)

        plot_title = plot_dir + phi_title+".png"
        plt.savefig(plot_title)
        plt.close()
        #print("plot saved to {}".format(plot_title))
        
        return ["nofit","nofit","nofit","nofit"]
    else:
        binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])

        ic(binscenters)
        # 5.) Fit the function to the histogram data.
        popt, pcov = curve_fit(fit_function, xdata=binscenters, ydata=data_entries, p0=[2.0, 2, 0.3],
                    sigma=data_errors, absolute_sigma=True)
        #print(popt) #popt contains the values for A, B, C

        a_err = np.sqrt(pcov[0][0])
        b_err = np.sqrt(pcov[1][1])
        c_err = np.sqrt(pcov[2][2])

        a,b,c = popt[0],popt[1],popt[2]
        #ic(a_err,b_err,c_err)
        #ic.disable()
        
        # 6.)
        # Generate enough x values to make the curves look smooth.
       
        fit_y_data_1 = fit_function(binscenters, *popt)

        ic(fit_y_data_1)

        

        chisq0 = stats.chisquare(f_obs=data_entries, f_exp=fit_y_data_1)
        chisq = stats.chisquare(f_obs=np.array(data_entries, dtype=np.float64), f_exp=np.array(fit_y_data_1, dtype=np.float64))

        sums=[]
        for ind,val in enumerate(fit_y_data_1):
            diff2 = (data_entries[ind]-val)**2
            s1 = diff2/val
            sums.append(s1)

        manchisq = np.sum(sums)

        ###ic.enable()
        if chisq0[0]<0:
            ic(manchisq)
            ic(chisq0[0])
        if not (chisq0[0] == chisq[0]):
            print("ERROR MISMATCH")
            print(chisq0[0])
            print(chisq[0])
            print(manchisq)

        ic.disable()

        p = chisq[1]
        chisq = chisq[0]

        ic(chisq)
        ic(p)


        xspace = np.linspace(0, xmax, 1000)
        fit_y_data = fit_function(xspace, *popt)

        ###ic.enable()
        ic(fit_y_data)
        
        y_manual = []
        for ind, val in enumerate(xspace):
            ic(val,a,b,c)
            y_one = fit_function(val,a,b,c)
            ic(y_one)
            y_manual.append(y_one)


        
        #7
        # Plot the histogram and the fitted function.

        fig = plt.figure()
        ax = fig.add_subplot(111)
        

        bar1 = ax.bar(binscenters, data_entries, width=bins[1] - bins[0], color='navy', label='Histogram entries')
        fit1, = ax.plot(xspace, fit_y_data, color='darkorange', linewidth=2.5, label='Fitted function')

        # Make the plot nicer.
        plt.xlim(xmin,xmax)
        plt.ylim(0,300)
        plt.xlabel(r'phi')
        plt.ylabel(r'Number of entries')

        plot_title = plot_dir + phi_title+".png"
        plt.title(phi_title)
        #plt.legend(loc='best')

        fit_params = "A: {:2.2f} +/- {:2.2f}\n B:{:2.2f} +/- {:2.2f}\n C:{:2.2f} +/- {:2.2f}\n Chi:{:2.2f} \n p:{:2.2f}".format(a,a_err,b,b_err,c,c_err,chisq,p)

        extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        ax.legend([bar1, fit1, extra], ("Data","A+Bcos(2Phi)+Ccos(Phi)",fit_params))

        #plt.text(120, max(data_entries)/1.3, fit_params)

        
        plt.savefig(plot_title)
        #plt.show()
        plt.close()
        #print("plot saved to {}".format(plot_title))

        return popt, pcov, chisq, p

# 3.) Generate exponential and gaussian data and histograms.
