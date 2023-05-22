import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.patches import Rectangle

# 1.) Necessary imports.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import argparse
import sys
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib as mpl

from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm as cmx
jet = plt.get_cmap('jet')
cNorm  = colors.Normalize(vmin=0, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
scalarMap.to_rgba(0)
# returns (0.0, 0.0, 0.5, 1.0), i.e. blue
scalarMap.to_rgba(0.5)
# returns (0.49019607843137247, 1.0, 0.47754585705249841, 1.0) i.e. green
scalarMap.to_rgba(1)
# returns (0.5, 0.0, 0.0, 1.0) i.e. red

import matplotlib.ticker as tick

#print(scalarMap.to_rgba(1))


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

def resid_weighted_c12(pars):
    return (((y-fit_function(x,pars))**2)/sigma_c12).sum()

def constr0(pars):
    return fit_function(0,pars)

def constr180(pars):
    return fit_function(180,pars)


xmax = 360
xspace = np.linspace(0, xmax, 1000)






dfx = pd.read_pickle('/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_inbending_rad_All_All_All_fall2022DNP_SangbaekCuts_rad_excut_sigma_3.pkl')

df = dfx.query("qmin==6.25 and xmin==0.5 and tmin==0.72")
df = df.query("acc_corr>0.05")
#df_c12_select.to_csv('test_dnp_inbending.csv', index=False)


df.loc[:,"xsec_corr_nb_gamma"] = df["xsec_corr"]*1E33/df["gamma_exp"]



df.loc[:,"uncert_xsec_corr_nb_gamma"] = np.sqrt(  np.square(df["uncert_xsec"]/df["xsec"]) + np.square(df["uncert_acc_corr"]/df["acc_corr"]))*df["xsec_corr_nb_gamma"]

binscenters_c12 = df["pave_exp"]
data_entries_c12 = df["xsec_corr_nb_gamma"]
sigma_c12 = df["uncert_xsec_corr_nb_gamma"]


# plt.errorbar(binscenters_c12,data_entries_c12,yerr=sigma_c12, color='blue',fmt="x", label='CLAS12')
# print(df)
# plt.show()



con1 = {'type': 'ineq', 'fun': constr0}
con2 = {'type': 'ineq', 'fun': constr180}
# con3 = {'type': 'ineq', 'fun': constr270}
cons = [con1,con2]

x = binscenters_c12
y = data_entries_c12
valid = ~(np.isnan(x) | np.isnan(y))

epsi_mean_c12 = df["epsi_exp"].mean()
q_mean_c12 = df['qave_exp'].mean()
x_mean_c12 = df['xave_exp'].mean()
t_mean_c12 = df['tave_exp'].mean()


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


plt.rcParams["font.size"] = "20"

fig, ax = plt.subplots(figsize =(14, 10)) 

#plt.errorbar(binscenters, data_entries, yerr=sigma, color="red",fmt="o",label='CLAS6 Data')#. Bin Averages: Q2: {:.2f} xB: {:.2f} t: {:.2f}'.format(df_sf_binned_small['q'].values[0],df_sf_binned_small['x'].values[0],df_sf_binned_small['t'].values[0]))

plt.errorbar(binscenters_c12, data_entries_c12, yerr=sigma_c12, color="blue",fmt="x",label='CLAS12 Data')#. Bin Averages: Q2: {:.2f} xB: {:.2f} t: {:.2f}'.format(df["qave_exp"].mean(),df["xave_exp"].mean(),df["tave_exp"].mean()))

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


df_GK = pd.read_csv('GK_Model/cross_section_pi0_10600_big.txt', sep='\t', header=0)
    #df_GK_calc = pd.read_csv('GK_Model/cross_section_pi0_10600_big.txt', sep='\t', header=0)
    #df_GK_calc = pd.read_csv('cross_section_pi0_575_new_big_1.txt', sep='\t', header=0)
    # Data Structure:
    #     Q2	xB	mt	sigma_T	sigma_L	sigma_LT	sigma_TT	W	y	epsilon	gammaa	tmin
    #  1.75 	 0.225 	 -0.020 	 nan 	 nan 	 -nan 	 nan 	 2.6282355 	 0.0806671 	 0.9961151 	 0.3190776 	 -0.0574737

#for col in df_GK_calc.columns:
#    print(col)

df_GK['sigma_T'] = pd.to_numeric(df_GK["sigma_T"], errors='coerce')
df_GK['sigma_L'] = pd.to_numeric(df_GK["sigma_L"], errors='coerce')
df_GK['sigma_LT'] = pd.to_numeric(df_GK["sigma_LT"], errors='coerce')
df_GK['sigma_TT'] = pd.to_numeric(df_GK["sigma_TT"], errors='coerce')
df_GK['W'] = pd.to_numeric(df_GK["W"], errors='coerce')
df_GK = df_GK.query('W > 2')

df_GK = df_GK.query('Q2==6.25 and xB==0.525')
print(df_GK)
plt.show()