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

from matplotlib import pylab as plt

import numpy as np
from scipy.optimize import minimize

def resid(pars):
    return ((y-fit_function(x,pars))**2).sum()

def resid_weighted(pars):
    return (((y-fit_function(x,pars))**2)/sigma).sum()


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


df = pd.read_pickle("struct_funcs.pkl")
#df.to_csv("struct_funcs.csv")


df = df.query("tel_c12<450 and tel_c12>-450")

q2bins,xBbins, tbins, phibins = fs.q2bins, fs.xBbins, fs.tbins, fs.phibins

qrange = [q2bins[0], q2bins[-1]]
xBrange = [xBbins[0], xBbins[-1]]
trange = [tbins[0], tbins[-1]]

#data_vals = []


sf_data_vals = []

i = 0
for qmin,qmax in zip(q2bins[0:-1],q2bins[1:]):
    print(" \n Q2 bin: {} to {}".format(qmin,qmax))
    #query = "q2 >{} and q2 < {}".format(qmin,qmax)
    #df_q = df.query(query)

    for xmin,xmax in zip(xBbins[0:-1],xBbins[1:]):
        #print("        xB bin: {} to {}".format(xmin,xmax))
        #query = "xb>{} and xb<{}".format(xmin,xmax)
        #df_qx = df_q.query(query)

            query = "qmin == {} and xmin == {} ".format(qmin,xmin)

            df_small = df.query(query)
            if not df_small.empty:
                print(df_small)


                
                plt.rcParams["font.size"] = "20"

                fig, ax = plt.subplots(figsize =(14, 10)) 


                #plt.errorbar(binscenters, data_entries, yerr=sigma, color="blue",fmt="o",label="CLAS6 Data")
                plt.errorbar(df_small['tC6'],df_small['telC6'], yerr=df_small['tel-statC6'], color="black",fmt="-x",label="CLAS6 $d\sigma_T/dt+\epsilon d\sigma_L/dt$")
                plt.errorbar(df_small['tC12'],df_small['tel_c12'], yerr=df_small['tel_c12_err'], color="black",fmt="--s",label="CLAS12 $d\sigma_T/dt+\epsilon d\sigma_L/dt$")
                plt.errorbar(df_small['tC6'],df_small['ltC6'], yerr=df_small['lt-statC6'], color="blue",fmt="-x",label="CLAS6 $d\sigma_{TT}/dt$")
                plt.errorbar(df_small['tC12'],df_small['tt_c12'], yerr=df_small['tt_c12_err'], color="blue",fmt="--s",label="CLAS12 $d\sigma_{TT}/dt$")
                plt.errorbar(df_small['tC6'],df_small['ttC6'], yerr=df_small['tt-statC6'], color="red",fmt="-x",label="CLAS6 $d\sigma_{LT}/dt$")
                plt.errorbar(df_small['tC12'],df_small['lt_c12'], yerr=df_small['lt_c12_err'], color="red",fmt="--s",label="CLAS12 $d\sigma_{LT}/dt$")





                # print(binscenters_c12, data_entries_c12)
                # plt.errorbar(binscenters_c12, data_entries_c12, yerr=sigma_c12, color="green",fmt="x",label="CLAS12 Data")


                # plt.rcParams["font.size"] = "20"

                # fit2, = ax.plot(xspace, fit_y_data_weighted, color='red', linewidth=2.5, label='CLAS6 Fit:         t+l:{:.0f} tt:{:.0f} lt:{:.0f}'.format(pub_tel,pub_tt,pub_lt))
                # #fit3, = ax.plot(xspace, fit_y_data_weighted_new, color='black', linewidth=2.5, label='New CLAS6 Fit: t+l:{:.0f} tt:{:.0f} lt:{:.0f}'.format(tel,tt,lt))
                # fit4, = ax.plot(xspace, fit_y_data_weighted_new_c12, color='green', linewidth=2.5, label='CLAS12 Fit: t+l:{:.0f} tt:{:.0f} lt:{:.0f}'.format(tel_c12,tt_c12,lt_c12))
                
                ax.legend(loc="best")
                ax.set_xlabel("-t (GeV$^2$)")  
                ax.set_ylabel('Structure Functions (nb/GeV$^2$)')  
                title = "Structure Functions at , Q$^2$ = {:.2f}, x$_B$ = {:.2f}".format((df_small['qmin'].values[0]+df_small['qmax'].values[0])/2,(df_small['xmin'].values[0]+df_small['xmax'].values[0])/2)
                plt.title(title)

                plt.annotate("Q$^2_6$={:.2f}, x$_{{{}}}$={:.2f}".format(df_small['qC6'].mean(),"""B,6""",df_small['xC6'].mean()), xy=(0.65, 0.15), xycoords='axes fraction')
                plt.annotate("Q$^2_{{{}}}$={:.2f}, x$_{{{}}}$={:.2f}".format('12',df_small['qC12'].mean(),'B,12',df_small['xC12'].mean()), xy=(0.65, 0.05), xycoords='axes fraction')

                # #plt.savefig("comp_plots/"+title.replace("$","").replace(".","").replace("^","").replace(" ","").replace("=","").replace(",","_")+".png")
                # #plt.savefig("cd_inc_2/"+title.replace("$","").replace(".","").replace("^","").replace(" ","").replace("=","").replace(",","_")+".pdf")
                plt.savefig("structs_pdf/"+title.replace("$","").replace(".","").replace("^","").replace(" ","").replace("=","").replace(",","_")+".pdf")

                #plt.show()
                plt.close()
                #sys.exit()
                


                #sys.exit()                