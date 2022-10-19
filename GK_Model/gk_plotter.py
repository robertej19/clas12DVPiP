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


#print(df_GK)

xB = df_GK['xB'].unique()
Q2 = df_GK['Q2'].unique()

print(xB)
print(Q2)

        
qmin = min(Q2)
qmax = max(Q2)
qsteps = len(Q2)

colorset = Q2/qmax

N = int(qsteps/.002)
cmap = plt.get_cmap('jet',N)

#fig = plt.figure(figsize=(8,6))
#ax1 = fig.add_axes([0.10,0.10,0.70,0.85])

# for i,n in enumerate(np.linspace(0,2,N)):
#     y = np.sin(x)*x**n
#     ax1.plot(x,y,c=cmap(i))

# plt.xlabel('x')
# plt.ylabel('y')

norm = mpl.colors.Normalize(vmin=qmin,vmax=qmax)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])




rainbow = True
if rainbow:
    for xind, xb_val in enumerate(xB):

         # Create figure
        plt.rcParams["font.size"] = "20"
        fig, ax = plt.subplots(figsize =(14, 10))
        plot_title = "GK Cross Section over Q$^2$ at x$_B$={}".format(xb_val)
        plt.title(plot_title)
        #plt.ylim([0,np.max([np.max(GK_curve_c12),np.max(fit_c12_weighted)])*1.3])
        ax.set_xlabel('t [GeV$^2$]')
        ax.set_ylabel(r'$\frac{d\sigma}{dt}$'+ '  [nb/GeV$^2$]')
        ax.set_xlim([0.1,1])

        #colorset = ['red','blue','green','orange','purple','black','purple','black','brown','pink','gray','olive','cyan','brown','pink','gray','olive','cyan','purple','black','brown','pink','gray','olive','cyan']

        #colorset = int(np.ones(len(Q2))

        

        for qind, q2_val in enumerate(Q2):

            print(qind,xind)

            ybounds_factor = 1
            # if q2_val> 2.5:
            #     ybounds_factor = 1/10
            #     if q2_val > 7:
            #         ybounds_factor = 1/100

            query_GK_Model = "Q2=={} and xB=={} ".format(q2_val, xb_val)

            df_GK_calc = df_GK.query(query_GK_Model)
            #print(df_GK_calc)


        
            ax.set_ylim([0.5*ybounds_factor,500*ybounds_factor])

            plt.yscale("log")
            #ax.legend()#[dtedl_2022,dtedl_2014,extra], ("2022 GK fit","2014 GK fit","+ Data",))

            #plt.savefig("GK_Model/gk_c12_plots/reduced_xsec_{}_{}_{}_{}_{}_{}.png".format(qmin,qmax,xmin,xmax,tmin,tmax))


            plt.plot(-1*df_GK_calc['mt'], df_GK_calc['sigma_T']+df_GK_calc['epsilon']*df_GK_calc['sigma_L'],color=scalarMap.to_rgba(colorset[qind]), linewidth=5, label='$\sigma_T+\epsilon\sigma_L$')
            #plt.plot(-1*df_GK_calc['mt'], df_GK_calc['sigma_TT'],'b', linewidth=10, label='$\sigma_{{TT}}$')
            #plt.plot(-1*df_GK_calc['mt'], df_GK_calc['sigma_LT'],'r',  linewidth=10, label='$\sigma_{{LT}}$')

            #ax.legend()

        cbar = plt.colorbar(sm, ticks=np.linspace(qmin,qmax,int(N/2400)), 
             boundaries=np.arange(qmin*0.95,qmax*1.05,0.1))
        cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
        
        plt.show()
        #plt.savefig("gk_big_plots_log_rainbow_xb/fig_{}_xb_{}.png".format(xind,xb_val))
        #plt.close()
else:
    for qind, q2_val in enumerate(Q2):
        for xind, xb_val in enumerate(xB):
            #q2_val = Q2[10]
            #xb_val = xB[2]
            print(qind,xind)

            ybounds_factor = 1
            # if q2_val> 2.5:
            #     ybounds_factor = 1/10
            #     if q2_val > 7:
            #         ybounds_factor = 1/100

            query_GK_Model = "Q2=={} and xB=={} ".format(q2_val, xb_val)

            df_GK_calc = df_GK.query(query_GK_Model)
            #print(df_GK_calc)


            # Create figure
            plt.rcParams["font.size"] = "20"
            fig, ax = plt.subplots(figsize =(14, 10))
            plot_title = "GK Cross Section at {}<Q$^2$, x$_B$<{}".format(q2_val,xb_val)
            plt.title(plot_title)
            #plt.ylim([0,np.max([np.max(GK_curve_c12),np.max(fit_c12_weighted)])*1.3])
            ax.set_xlabel('t [GeV$^2$]')
            ax.set_ylabel(r'$\frac{d\sigma}{dt}$'+ '  [nb/GeV$^2$]')
            ax.set_xlim([0.1,1])
            ax.set_ylim([0.1*ybounds_factor,1000*ybounds_factor])

            plt.yscale("log")
            #ax.legend()#[dtedl_2022,dtedl_2014,extra], ("2022 GK fit","2014 GK fit","+ Data",))

            #plt.savefig("GK_Model/gk_c12_plots/reduced_xsec_{}_{}_{}_{}_{}_{}.png".format(qmin,qmax,xmin,xmax,tmin,tmax))


            plt.plot(-1*df_GK_calc['mt'], df_GK_calc['sigma_T']+df_GK_calc['epsilon']*df_GK_calc['sigma_L'],'k', linewidth=10, label='$\sigma_T+\epsilon\sigma_L$')
            #plt.plot(-1*df_GK_calc['mt'], df_GK_calc['sigma_TT'],'b', linewidth=10, label='$\sigma_{{TT}}$')
            #plt.plot(-1*df_GK_calc['mt'], df_GK_calc['sigma_LT'],'r',  linewidth=10, label='$\sigma_{{LT}}$')

            #ax.legend()
            #plt.show()

            plt.savefig("gk_big_plots_log/fig_{}_{}_{}_{}.png".format(qind,xind,q2_val,xb_val))
            plt.close()