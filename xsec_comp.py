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


def bin_clas6_sf(df):
    q2bins,xBbins, tbins, phibins = fs.q2bins, fs.xBbins, fs.tbins, fs.phibins
    print(df)

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

# df_sf = pd.read_csv('clas6_structure_funcs.txt', sep=" ")#, header=None)
# df_sf_binned = bin_clas6_sf(df_sf)
# df_sf_binned.to_pickle('clas6_structure_funcs_binned.pkl')
# print(df_sf.query("q2==1.14"))
# print(df_sf_binned.query("q==1.14"))
# sys.exit()
#data.columns = ["a", "b", "c", "etc."]
#print(df_sf['tel-stat'].mean())
#sys.exit()
#df = pd.read_pickle("full_xsection.pkl")
df = pd.read_pickle("final_data_files/full_xsection_CD_Included.pkl")
#df = pd.read_pickle("final_data_files/full_xsection_Sangbaek_rad_CD_sim.pkl")
#df = pd.read_pickle("final_data_files/full_xsection_CD_ONLY.pkl")
#print(df)
#sys.exit()

df_sf_binned = pd.read_pickle('final_data_files/clas6_structure_funcs_binned.pkl')
df_sf_binned = df_sf_binned.apply(pd.to_numeric)
#print(df_sf_binned)


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
df.loc[:,"tot_clas6_uncert"] = np.sqrt(np.square(df["stat"]/df["dsdtdp"]) + np.square(df["sys"]/df["dsdtdp"]))*df["dsdtdp"]

df.loc[:,"epsi_clas6"] = get_gamma(df["x"],df["q"],5.776)[1] #get_gamma((dfout["xmin"]+dfout["xmax"])/2,(dfout["qmin"]+dfout["qmax"])/2)[1]


df.loc[:,"uncert_xsec_corr_nb_gamma"] = np.sqrt(  np.square(df["uncert_xsec"]/df["xsec"]) + np.square(df["uncert_acc_corr"]/df["acc_corr"]))*df["xsec_corr_nb_gamma"]
df.loc[:,"c_12_uncert_ratio"] = df['uncert_xsec_corr_nb_gamma']/df['xsec_corr_nb_gamma']

#df = df[df.acc_corr > 0.01]

df.loc[(df.acc_corr < 0.01),'xsec_corr_nb_gamma']=np.nan


q2bins,xBbins, tbins, phibins = fs.q2bins, fs.xBbins, fs.tbins, fs.phibins

qrange = [q2bins[0], q2bins[-1]]
xBrange = [xBbins[0], xBbins[-1]]
trange = [tbins[0], tbins[-1]]
phirange = [phibins[0], phibins[-1]]

#data_vals = []

print(df.columns)
print(df.head())

df = df[df.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]

#print(phibins)
#sys.exit()

comp_vals = []
p0 = 0
p1 = 139
p2 = 221
p3 = 360
tbins = tbins[0:9]
phibins = [p0,p1,p2,p3]

for phimin,phimax in zip(phibins[0:-1],phibins[1:]):
    print("PHI RANGE: {} - {}".format(phimin,phimax))
    query = "pmin >= {} and pmin < {}".format(phimin,phimax)
    df0 = df.query(query)
    for tmin,tmax in zip(tbins[0:-1],tbins[1:]):
        #print("space")
        print("T RANGE: {} - {}".format(tmin,tmax))
        query = "tmin == {}".format(tmin)
        df2 = df0.query(query)
        nb = df2['xsec_corr_red_nb']*1.15
        x6 = df2['dsdtdp']
        ratio = nb/x6
        df_small = df0.query(query).xsec_ratio_exp
        new_mean = ratio.mean()
        #df_small[~df_small.isin([np.nan, np.inf, -np.inf]).any(1)]
        mean_ratio = df_small.mean()
        #print(mean_ratio,new_mean)
        #sys.exit()
        comp_vals.append([phimin,phimax,tmin,tmax,new_mean])
#sys.exit()

df = pd.DataFrame(comp_vals, columns = ['pmin','pmax','tmin','tmax','mean_ratio'])

df.loc[:,"pmean"] = (df['pmin'] + (df['pmax']))/2
df.loc[:,"tmean"] = (df['tmin'] + (df['tmax']))/2


# # # x = [1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4]
# # # y = [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]
# # # z = [6,2,3,5,8,2,3,2,4,2,3,6,7,3,3,2]
# # # x = np.reshape(x,(4, 4))
# # # y = np.reshape(y,(4, 4))
# # # z = np.reshape(z,(4, 4))
# # # plt.pcolormesh(x, y, z)
# # # plt.show()
# # # sys.exit()

x = phibins
#print(x)
#sys.exit()
y = tbins

xx,yy = np.meshgrid(x,y)
#print(xx)
#sys.exit()
z = df.mean_ratio.values
print(z)
print(x)
print(y)
#print(x)
#print(y)
#print(z)
#x = np.reshape(x, (len(tbins), len(phibins)))
#y = np.reshape(y, (len(tbins), len(phibins)))
z = np.reshape(z, (len(tbins)-1, len(phibins)-1))
#plt.rcParams['pcolor.shading'] = 'flat'
#cmap = mpl.cm.get_cmap("OrRd").copy()

#cmap.set_bad(color='black')

fig, ax = plt.subplots(figsize =(36, 17)) 

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "20"
#plt.pcolormesh(x,y,colorsx,cmap=cmap)#norm=mpl.colors.LogNorm())
#plt.clim(0,3)

#plt.title("Ratio of CLAS12 to CLAS6 Reduced Cross Sections, Q2 = {}, xB = {}".format(qmin,xmin))
plt.title("Ratio of CLAS12 to CLAS6 Reduced Cross Sections over all Q$^2$ and x$_B$")


ax.set_xlabel('Lepton-Hadron Angle')
ax.set_ylabel('-t (GeV$^2)$')

plt.pcolormesh(xx,yy,z)#,cmap=cmap)#norm=mpl.colors.LogNorm())
plt.colorbar()
plt.show()

sys.exit()

sf_data_vals = []

for qmin,qmax in zip(q2bins[0:-1],q2bins[1:]):
    print(" \n Q2 bin: {} to {}".format(qmin,qmax))
    #query = "q2 >{} and q2 < {}".format(qmin,qmax)
    #df_q = df.query(query)

    for xmin,xmax in zip(xBbins[0:-1],xBbins[1:]):
        #print("        xB bin: {} to {}".format(xmin,xmax))
        #query = "xb>{} and xb<{}".format(xmin,xmax)
        #df_qx = df_q.query(query)

        for tmin,tmax in zip(tbins[0:-1],tbins[1:]):
            #print("                 t bin: {} to {}".format(tmin,tmax))
            #query = "t>{} and t<{}".format(tmin,tmax)
            #df_qxt = df_qx.query(query)

            query = "qmin == {} and xmin == {} and tmin == {}".format(qmin,xmin,tmin)

            df_small = df.query(query)
            df_sf_binned_small = df_sf_binned.query(query)

            df_check = df_small[df_small["xsec_corr_nb_gamma"].notnull()]

            if np.isnan(df_sf_binned_small["tel"].values[0]) or df_check.empty:
                pass
            else:
                print(df_small)
                print(df_sf_binned_small)
                sys.exit()
            

            

                epsi_mean_c6 = df_small["epsi_clas6"].mean()

                epsi_mean_c12 = df_small["epsi_exp"].mean()
                mean_xsec_uncer_ratio_c12 = df_small['c_12_uncert_ratio'].mean()


                binscenters_c12 = df_small["pave_exp"]
                data_entries_c12 = df_small["xsec_corr_nb_gamma"]
                sigma_c12 = df_small["uncert_xsec_corr_nb_gamma"]
                binscenters = df_small["p"]
                data_entries = df_small["dsdtdp"]
                sigma = df_small["tot_clas6_uncert"]

                


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
                    qmin,xmin,qmax,xmax,
                    tel_c12_err,tt_c12_err,lt_c12_err,])



                plt.rcParams["font.size"] = "20"

                fig, ax = plt.subplots(figsize =(14, 10)) 


                plt.errorbar(binscenters, data_entries, yerr=sigma, color="red",fmt="o",label="CLAS6 Data")

                print(binscenters_c12, data_entries_c12)
                plt.errorbar(binscenters_c12, data_entries_c12, yerr=sigma_c12, color="blue",fmt="x",label="CLAS12 Data")


                plt.rcParams["font.size"] = "20"

                fit2, = ax.plot(xspace, fit_y_data_weighted, color='red', linewidth=2.5, label='CLAS6 Fit:         t+l:{:.0f} tt:{:.0f} lt:{:.0f}'.format(pub_tel,pub_tt,pub_lt))
                #fit3, = ax.plot(xspace, fit_y_data_weighted_new, color='black', linewidth=2.5, label='New CLAS6 Fit: t+l:{:.0f} tt:{:.0f} lt:{:.0f}'.format(tel,tt,lt))
                fit4, = ax.plot(xspace, fit_y_data_weighted_new_c12, color='blue', linewidth=2.5, label='CLAS12 Fit: t+l:{:.0f} tt:{:.0f} lt:{:.0f}'.format(tel_c12,tt_c12,lt_c12))
                
                ax.legend(loc="best")
                ax.set_xlabel("Phi")  
                ax.set_ylabel('Reduced Cross Section (nb/GeV$^2$)')  
                title = "Cross Section Fit Over Phi, Q$^2$ = {:.2f}, x$_B$ = {:.2f}, t = {:.1f}".format(df_sf_binned_small['q'].values[0],df_sf_binned_small['x'].values[0],df_sf_binned_small['t'].values[0])
                plt.title(title)

                #plt.savefig("comp_plots/"+title.replace("$","").replace(".","").replace("^","").replace(" ","").replace("=","").replace(",","_")+".png")
                #plt.savefig("cd_inc_2/"+title.replace("$","").replace(".","").replace("^","").replace(" ","").replace("=","").replace(",","_")+".pdf")
                plt.savefig("rad_final_tight_cuts/"+title.replace("$","").replace(".","").replace("^","").replace(" ","").replace("=","").replace(",","_")+".png")

                #plt.show()
                plt.close()
                #sys.exit()
                

df_out = pd.DataFrame(sf_data_vals, columns = ['qC6','xC6','tC6','telC6','tel-statC6','tel-sysC6','ltC6','lt-statC6','lt-sysC6','ttC6','tt-statC6',
                                'tt-sysC6','qC12','xC12','tC12','tel_c12','tt_c12','lt_c12','mean_uncert_c12','qmin','xmin',
                                'qmax','xmax','tel_c12_err','tt_c12_err','lt_c12_err'])


df_out.to_pickle("struct_funcs.pkl")
