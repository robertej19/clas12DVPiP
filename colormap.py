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
#from convert_root_to_pickle import convert_REC_RAD_root_to_pkl
#from convert_root_to_pickle import convert_REC_NORAD_root_to_pkl
#from convert_root_to_pickle import convert_real_to_pkl
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
from new_dvpip_cuts import makeDVpi0P

from bin_events import bin_df


# r = np.random.randn(100,3)
# H, edges = np.histogramdd(r, bins = (5, 8, 4))
# H.shape, edges[0].size, edges[1].size, edges[2].size((5, 8, 4), 6, 9, 5)

# sys.exit()



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


#df_in = pd.read_pickle('full_xsection_inbending_norad_All_All_All_old_bins_no_ang_cuts.pkl')
#df_out = pd.read_pickle('full_xsection_outbending_norad_All_All_All_old_bins_no_ang_cuts.pkl')

# df_in = pd.read_pickle('full_xsection_inbending_norad_All_All_All_old_bins_95_ang_cuts.pkl')
# df_out = pd.read_pickle('full_xsection_outbending_norad_All_All_All_old_bins_95_ang_cuts.pkl')

#df_in = pd.read_pickle('full_xsection_inbending_norad_All_All_All_new_bins.pkl')
#df_out = pd.read_pickle('full_xsection_outbending_norad_All_All_All_new_bins.pkl')

#df_in = pd.read_pickle('/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_inbending_norad_All_All_All_new_f18_in_processing.pkl')
#df_out = pd.read_pickle('/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_outbending_norad_All_All_All_new_f18_in_processing.pkl')

# df_in = pd.read_pickle('/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_inbending_norad_All_All_All_new_f18_in_processing_simple_cuts.pkl')
# df_out = pd.read_pickle('/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_outbending_norad_All_All_All_new_f18_in_processing_simple_cuts.pkl')


#df_in = pd.read_pickle('/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_inbending_norad_All_All_All_new_f18_in_and_out_simple_cuts.pkl')
#df_out = pd.read_pickle('/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_outbending_norad_All_All_All_new_f18_in_and_out_simple_cuts.pkl')

# df_in = pd.read_pickle('/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_inbending_norad_All_All_All_new_f18_in_and_out_advanced_cuts.pkl')
# df_out = pd.read_pickle('/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_outbending_norad_All_All_All_new_f18_in_and_out_advanced_cuts.pkl')

# df_in = pd.read_pickle('/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_inbending_norad_All_All_All_new_f18_in_and_out_simple_9_deg_cuts.pkl')
# df_out = pd.read_pickle('/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_outbending_norad_All_All_All_new_f18_in_and_out_simple_9_deg_cuts.pkl')

# df_in = pd.read_pickle('/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_inbending_norad_All_All_All_new_f18_in_and_out_advanced_9_deg_cuts.pkl')
# df_out = pd.read_pickle('/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_outbending_norad_All_All_All_new_f18_in_and_out_advanced_9_deg_cuts.pkl')


# df_in = pd.read_pickle('/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_inbending_rad_All_All_All_rad_f18_in_and_out_advanced_9_deg_cuts.pkl')
# df_out = pd.read_pickle('/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_outbending_rad_All_All_All_rad_f18_in_and_out_advanced_9_deg_cuts.pkl')

df_in = pd.read_pickle('/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_inbending_rad_All_All_All_rad_f18_in_and_out_advanced_no_ang_cuts.pkl')
df_out = pd.read_pickle('/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_outbending_rad_All_All_All_rad_f18_in_and_out_advanced_no_ang_cuts.pkl')











#for col in df_out.columns:
##    print(col)
#sys.exit()


#xsec_corr_red_nb

jj = True
if jj:
  


    #q2bins,xBbins, tbins, phibins = [fs.q2bins_test, fs.xBbins_test, fs.tbins_test, fs.phibins_test]
    q2bins,xBbins, tbins, phibins = [fs.q2bins, fs.xBbins, fs.tbins, fs.phibins]


    #q2bins = [2.0,2.5]
    #xBbins = [0.3,0.25]
    for qmin in q2bins[:-1]:
        for xmin in xBbins[:-1]:
            #print(" ON q-{} x-{}".format(qmin, xmin))
            # qmin = 1.5
            # xmin = 0.25
            df_om = df_out.query("qmin==@qmin and xmin==@xmin")
            df_im = df_in.query("qmin==@qmin and xmin==@xmin")

            #for tmin,tmax in zip(tbins[0:-1],tbins[1:]):

            #pave_arr = []

            #for pmin,pmax in zip(phibins[0:-1],phibins[1:]):
            #    pave_arr.append((pmin+pmax)/2)


            # for tmin in tbins[0:-1]:
            #     for pmin,pmax in zip(phibins[0:-1],phibins[1:]):
            #         #pave = (pmin+pmax)/2
            #         df_t = df.query("tmin==@tmin and pmin==@pmin")
            #         #print(df_t)
            #         if df_t.shape[0] == 0:
            #             #print("APPENDING ZEROS")
            #             #dict0 = {'qmin':[qmin],'xmin':[xmin],'tmin':[tmin],'pmin':[pmin],'ratio':['nan'],'ratio_uncert':['nan']}
            #             #dict0 = {'qmin':[qmin],'xmin':[xmin],'tmin':[tmin],'pmin':[pmin],'ratio':[0],'ratio_uncert':[0]}
            #             dict0 = {'qmin':[qmin],'xmin':[xmin],'tmin':[tmin],'pmin':[pmin],'ratio':[0],'ratio_uncert':[0]}
                        
            #             df2 = pd.DataFrame(dict0)
            # #            df = pd.concat([df,df2],ignore_index=True)
            #             df = df.append(df2)#,ignore_index=True)

            t = []
            p = []
            r = []
            run = []


            for tind,tmin in enumerate(tbins):
                for pind,pmin in enumerate(phibins):
                    #print(tmin,pmin)
                    if (tind<len(tbins)-1) and (pind<len(phibins)-1):
                        #pass
                        #df_t = df.query("tmin==@tmin and pmin==@pmin")
                        df_t_out = df_om.query("tmin==@tmin and pmin==@pmin")
                        df_t_in = df_im.query("tmin==@tmin and pmin==@pmin")
                        #print(df_t_out)
                        #rval = df_t.counts_low.values[0]
                        rval1 = df_t_in['xsec_corr_red_nb'].values[0]
                        rval2 = df_t_out['xsec_corr_red_nb'].values[0]
                        #print(rval1,rval2)
                        #/df_t_out['xsec_corr_red_nb'].values[0]
                        #print(rval)
                        rval=rval1/rval2
                        runcertval = np.sqrt(np.square(df_t_in['uncert_xsec_corr_red_nb'].values[0]/df_t_in['xsec_corr_red_nb'].values[0])+np.square(df_t_out['uncert_xsec_corr_red_nb'].values[0]/df_t_out['xsec_corr_red_nb'].values[0]))
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

            print(r)
            print(run)

            cmap = plt.cm.jet  # define the colormap
            

            colors = []
            delta = 0.01
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
            

            cmap.set_bad(color='white')

            #print(x)
            ##print(y)
            #print(z)
            fig, ax = plt.subplots(figsize =(36, 17)) 

            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["font.size"] = "20"
            plt.pcolormesh(x,y,colorsx,cmap=cmap)#norm=mpl.colors.LogNorm())
            #plt.clim(0,3)

            #plt.title("Ratio of CLAS12 to CLAS6 Reduced Cross Sections, Q2 = {}, xB = {}".format(qmin,xmin))
            plt.title("Ratio of inbending to outbending cross sections, Q2 = {}, xB = {}".format(qmin,xmin))

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

                print(uncert)
                if uncert == 0 or np.isnan(uncert):
                    pass
                else:

                    #print(i,j)
                    ii = x[i][j]+9
                    jj = y[i][j]*1.2
                    ax.text(ii, jj, '{:0.2f} $\pm$ {:0.1f}'.format(zz,uncert), ha='center', va='center',
                            bbox=dict(facecolor='white', edgecolor='0.3'))


            plt.ylim([0,3])
            #plt.show()
            #sys.exit()
            plt.savefig("rad_f18_in_and_out_advanced_no_ang_cuts/ratio_q2_{}_xB_{}.png".format(qmin,xmin))
            
            #sys.exit()
            
            plt.close()