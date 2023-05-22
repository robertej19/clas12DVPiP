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
E = 10.6
Clas6_Sim_BeamTime = 11922445
Clas12_Sim_BeamTime = 16047494
Clas12_exp_luminosity = 5.5E40
fs = filestruct.fs()



cmap = plt.cm.jet  # define the colormap
# extract all colors from the .jet map
# cmaplist = [cmap(i) for i in range(cmap.N)]
# # force the first color entry to be grey
# cmaplist[0] = (0, 0, 0, 1.0)
# cmaplist[1] = (0, 0, 0, 1.0)
# cmaplist[2] = (0, 0, 0, 1.0)
# cmaplist[3] = (0, 0, 0, 1.0)
#cmaplist[2] = (.5, 0 , 1, .5)
#cmaplist[1] = (0, .5, 1, 1.0)
#cmaplist[3] = (1, 1,0,  1.0)



# # create the new map
# cmap = mpl.colors.LinearSegmentedColormap.from_list(
#     'Custom cmap', cmaplist, cmap.N)


#plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "10"

df_minibin = pd.read_pickle("binned_dvpip/full_xsection.pkl")
""" Columns: 
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
"""

q2bins,xBbins, tbins, phibins = [fs.q2bins, fs.xBbins, fs.tbins, fs.phibins]
q2bins = q2bins[:-6]
#q2bins = [2.5,3.0]
#xBbins = [0.3,0.35]
for qmin,qmax in zip(q2bins[0:-1],q2bins[1:]):
    for xmin,xmax in zip(xBbins[0:-1],xBbins[1:]):
        print(" ON q-{} x-{}".format(qmin, xmin))
        df = df_minibin.query("qmin==@qmin and xmin==@xmin")


        t = df["tmin"].to_numpy()
        p = df["pmin"].to_numpy()

        r = df["counts_exp"].to_numpy()
        run = df["uncert_counts_exp"].to_numpy()

        #r = df["xsec_ratio_exp_corr"].to_numpy()
        #run = df["uncert_xsec_ratio_exp_corr"].to_numpy()

        colors = []
        delta = 0.3
        for rval,runcert in zip(r,run):
            zmin = rval-runcert
            zmax = rval+runcert
            if (zmin<1+delta and zmax>1-delta):
                colors.append(1)
            elif (zmin>1+delta):
                colors.append(3)
            elif (zmax<0.001):
                colors.append(0)
            elif (zmax<1-delta):
                colors.append(2)
            else:
                colors.append(float("NaN"))


        x = np.reshape(p, (len(tbins)-1, len(phibins)-1))
        y = np.reshape(t, (len(tbins)-1, len(phibins)-1))
        z = np.reshape(r, (len(tbins)-1, len(phibins)-1))
        zuncert = np.reshape(run, (len(tbins)-1, len(phibins)-1))
        colors_reshaped = np.reshape(colors, (len(tbins)-1, len(phibins)-1))

        z = np.ma.masked_where(z==0, z)

        cmap.set_bad(color='white')

        fig, ax = plt.subplots(figsize =(36, 17)) 

        #plt.rcParams["font.family"] = "Times New Roman"
        plt.pcolormesh(x,y,z,cmap=cmap)#colorsx,cmap=cmap)#norm=mpl.colors.LogNorm())

        #plt.pcolormesh(x,y,colors_reshaped,cmap=cmap)#,norm=mpl.colors.LogNorm())

        plt.clim(0,1.2)
        

        plt.rcParams["font.size"] = "14"

        for (i, j), zz in np.ndenumerate(z[:,:-1]):
            uncert = zuncert[:,:-1][i,j]
            if (np.isnan(zz) or np.isinf(zz) or zz==0):
                pass
            else:
                #print(i,j)
                ii = x[i][j]+9
                jj = y[i][j]*1.2
                if jj < 0.2:
                    jj = 0.12
                
                ax.text(ii, jj, '{:0.2f} $\pm$ \n {:0.2f}'.format(zz,uncert), ha='center', va='center',
                        bbox=dict(facecolor='white', edgecolor='0.3'))



        plt.rcParams["font.size"] = "20"
        plt.colorbar()

        plt.title("Ratio of CLAS12 to CLAS6 Reduced Cross Sections, Q2 = {:0.2f}, xB = {:0.2f}".format((qmin+qmax)/2,(xmin+xmax)/2))
        ax.set_xlabel('Lepton-Hadron Angle')
        ax.set_ylabel('-t (GeV$^2)$')

        plt.ylim([0.1,2])
        #plt.show()
        plt.savefig("t_phi_ratio_plots/ratio_q2_{:0.2f}_xB_{:0.2f}.png".format((qmin+qmax)/2,(xmin+xmax)/2))
        #sys.exit()
        
        plt.close()
        