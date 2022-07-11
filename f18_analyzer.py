import uproot
import pandas as pd
import numpy as np
import argparse
import os, sys
from icecream import ic
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
from utils import filestruct
from matplotlib.patches import Rectangle
pd.set_option('mode.chained_assignment', None)

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



a = 8
b = 338


qmin = 4.0
xmin = 0.3
tmin = 0.6

plt.rcParams["font.size"] = "20"

fig, ax = plt.subplots(figsize =(14, 10)) 


df2 = pd.read_pickle("/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_inbending_rad_All_All_All_rad_f18_new_simple_excuts_with_rangeexcut_sigma_2.pkl")
df3 = pd.read_pickle("/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_inbending_rad_All_All_All_rad_f18_new_simple_excuts_with_rangeexcut_sigma_3.pkl")
df4 = pd.read_pickle("/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_inbending_rad_All_All_All_rad_f18_new_simple_excuts_with_rangeexcut_sigma_4.pkl")




df22 = df2.query("qmin=={} and xmin=={} and tmin=={}".format(qmin,xmin,tmin))
df33 = df3.query("qmin=={} and xmin=={} and tmin=={}".format(qmin,xmin,tmin))
df44 = df4.query("qmin=={} and xmin=={} and tmin=={}".format(qmin,xmin,tmin))



binscenters = df33["pave_exp"]-3
data_entries = df33["xsec_corr_red_nb"]
sys_hi= np.abs(df44["xsec_corr_red_nb"]-df33["xsec_corr_red_nb"])
sys_low = np.abs(df22["xsec_corr_red_nb"]-df33["xsec_corr_red_nb"])
stat_sigma = df33["uncert_xsec_corr_red_nb"]



print(sys_hi)
print(sys_low)
print(df44["xsec_corr_red_nb"])
print(df33["xsec_corr_red_nb"])
print(df22["xsec_corr_red_nb"])


upper_total_error = np.sqrt(sys_hi**2 + stat_sigma**2)
lower_total_error = np.sqrt(sys_low**2 + stat_sigma**2)

asymmetric_error_sys = np.array(list(zip(sys_low, sys_hi))).T


asymmetric_error_total = np.array(list(zip(upper_total_error, lower_total_error))).T

mssize = 15
mssize2 = 10

incolor = "orange"
outcolor = "blue"
intotal = "red"
outtotal = "red"
outstat = "green"
instat = "green"

plt.errorbar(binscenters, data_entries, yerr=asymmetric_error_total, color=incolor,fmt=".",label='tight cuts',ecolor = intotal,markersize=mssize)#. Bin Averages: Q2: {:.2f} xB: {:.2f} t: {:.2f}'.format(df_small["qave_exp"].mean(),df_small["xave_exp"].mean(),df_small["tave_exp"].mean()))
plt.errorbar(binscenters, data_entries, yerr=stat_sigma, color=incolor,fmt=".",label='tight cuts',ecolor = instat,markersize=mssize)#. Bin Averages: Q2: {:.2f} xB: {:.2f} t: {:.2f}'.format(df_small["qave_exp"].mean(),df_small["xave_exp"].mean(),df_small["tave_exp"].mean()))


# 

# df = df2.query("qmin=={} and xmin=={} and tmin=={}".format(qmin,xmin,tmin))
# df33 = df3.query("qmin=={} and xmin=={} and tmin=={}".format(qmin,xmin,tmin))
# df44 = df4.query("qmin=={} and xmin=={} and tmin=={}".format(qmin,xmin,tmin))


# #for col in df.columns:
# #    print(col)
# #sys.exit()

# plt.rcParams["font.size"] = "20"

# binscenters = df["pave_exp"]
# data_entries = df["xsec_corr_red_nb"]
# sigma = df["uncert_xsec_corr_red_nb"]


# #plt.errorbar(binscenters, data_entries, yerr=sigma, color="red",fmt="o",label='CLAS6 Data')#. Bin Averages: Q2: {:.2f} xB: {:.2f} t: {:.2f}'.format(df_sf_binned_small['q'].values[0],df_sf_binned_small['x'].values[0],df_sf_binned_small['t'].values[0]))

# plt.errorbar(binscenters,  df22["xsec_corr_red_nb"], yerr=df22["uncert_xsec_corr_red_nb"], color="blue",fmt="x",label='tight cuts')#. Bin Averages: Q2: {:.2f} xB: {:.2f} t: {:.2f}'.format(df_small["qave_exp"].mean(),df_small["xave_exp"].mean(),df_small["tave_exp"].mean()))
# plt.errorbar(binscenters, df33["xsec_corr_red_nb"], yerr=df33["uncert_xsec_corr_red_nb"], color="red",fmt="x",label='nominal cuts')#. Bin Averages: Q2: {:.2f} xB: {:.2f} t: {:.2f}'.format(df_small["qave_exp"].mean(),df_small["xave_exp"].mean(),df_small["tave_exp"].mean()))
# plt.errorbar(binscenters, df44["xsec_corr_red_nb"], yerr=df44["uncert_xsec_corr_red_nb"], color="green",fmt="x",label='loose cuts')#. Bin Averages: Q2: {:.2f} xB: {:.2f} t: {:.2f}'.format(df_small["qave_exp"].mean(),df_small["xave_exp"].mean(),df_small["tave_exp"].mean()))

# plt.title("Cross Section, Outbending, Q2 = 2.25, xB = 0.34, t = 0.34")
# plt.xlabel("Phi")
# plt.ylabel("Cross Section (nb)")


# plt.legend()
plt.ylim([0,10])


df2 = pd.read_pickle("/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_outbending_rad_All_All_All_rad_f18_new_simple_excuts_with_rangeexcut_sigma_2.pkl")
df3 = pd.read_pickle("/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_outbending_rad_All_All_All_rad_f18_new_simple_excuts_with_rangeexcut_sigma_3.pkl")
df4 = pd.read_pickle("/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_outbending_rad_All_All_All_rad_f18_new_simple_excuts_with_rangeexcut_sigma_4.pkl")

df22 = df2.query("qmin=={} and xmin=={} and tmin=={}".format(qmin,xmin,tmin))
df33 = df3.query("qmin=={} and xmin=={} and tmin=={}".format(qmin,xmin,tmin))
df44 = df4.query("qmin=={} and xmin=={} and tmin=={}".format(qmin,xmin,tmin))


plt.rcParams["font.size"] = "20"

binscenters = df33["pave_exp"]
data_entries = df33["xsec_corr_red_nb"]
sys_hi= np.abs(df44["xsec_corr_red_nb"]-df33["xsec_corr_red_nb"])
sys_low = np.abs(df22["xsec_corr_red_nb"]-df33["xsec_corr_red_nb"])
stat_sigma = df33["uncert_xsec_corr_red_nb"]

upper_total_error = np.sqrt(sys_hi**2 + stat_sigma**2)
lower_total_error = np.sqrt(sys_low**2 + stat_sigma**2)

asymmetric_error_sys = np.array(list(zip(sys_hi, sys_low))).T

asymmetric_error_total = np.array(list(zip(upper_total_error, lower_total_error))).T

plt.errorbar(binscenters, data_entries, yerr=asymmetric_error_sys, color=outcolor,fmt="^",label='tight cuts',ecolor = outtotal,markersize=mssize2)#. Bin Averages: Q2: {:.2f} xB: {:.2f} t: {:.2f}'.format(df_small["qave_exp"].mean(),df_small["xave_exp"].mean(),df_small["tave_exp"].mean()))
plt.errorbar(binscenters, data_entries, yerr=stat_sigma, color=outcolor,fmt="^",label='tight cuts',ecolor = outstat,markersize=mssize2)#. Bin Averages: Q2: {:.2f} xB: {:.2f} t: {:.2f}'.format(df_small["qave_exp"].mean(),df_small["xave_exp"].mean(),df_small["tave_exp"].mean()))

#plt.errorbar(binscenters, df33["xsec_corr_red_nb"], yerr=df33["uncert_xsec_corr_red_nb"], color="yellow",fmt="x",label='nominal cuts')#. Bin Averages: Q2: {:.2f} xB: {:.2f} t: {:.2f}'.format(df_small["qave_exp"].mean(),df_small["xave_exp"].mean(),df_small["tave_exp"].mean()))
#plt.errorbar(binscenters, df44["xsec_corr_red_nb"], yerr=df44["uncert_xsec_corr_red_nb"], color="purple",fmt="x",label='loose cuts')#. Bin Averages: Q2: {:.2f} xB: {:.2f} t: {:.2f}'.format(df_small["qave_exp"].mean(),df_small["xave_exp"].mean(),df_small["tave_exp"].mean()))



#plt.errorbar(binscenters, data_entries, yerr=sigma, color="red",fmt="o",label='CLAS6 Data')#. Bin Averages: Q2: {:.2f} xB: {:.2f} t: {:.2f}'.format(df_sf_binned_small['q'].values[0],df_sf_binned_small['x'].values[0],df_sf_binned_small['t'].values[0]))

# plt.errorbar(binscenters, data_entries, yerr=sigma, color="black",fmt="x",label='tight cuts')#. Bin Averages: Q2: {:.2f} xB: {:.2f} t: {:.2f}'.format(df_small["qave_exp"].mean(),df_small["xave_exp"].mean(),df_small["tave_exp"].mean()))
# plt.errorbar(binscenters, df33["xsec_corr_red_nb"], yerr=df33["uncert_xsec_corr_red_nb"], color="yellow",fmt="x",label='nominal cuts')#. Bin Averages: Q2: {:.2f} xB: {:.2f} t: {:.2f}'.format(df_small["qave_exp"].mean(),df_small["xave_exp"].mean(),df_small["tave_exp"].mean()))
# plt.errorbar(binscenters, df44["xsec_corr_red_nb"], yerr=df44["uncert_xsec_corr_red_nb"], color="purple",fmt="x",label='loose cuts')#. Bin Averages: Q2: {:.2f} xB: {:.2f} t: {:.2f}'.format(df_small["qave_exp"].mean(),df_small["xave_exp"].mean(),df_small["tave_exp"].mean()))


qmina = 4.25
xmina = 0.34
tmina = 0.8

plt.title("Cross Section, In&Outbending, Q2 = {}, xB = {}, t = {}".format(qmina,xmina,tmina))
plt.xlabel("Phi")
plt.ylabel("Cross Section (nb)")


#fit_params = "words"
#extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
#ax.legend([extra], (fit_params))

from matplotlib.patches import Patch
from matplotlib.lines import Line2D


incolor = "orange"
outcolor = "blue"
intotal = "red"
outtotal = "red"
outstat = "green"
instat = "green"

legend_elements = [Line2D([0], [0], marker=".", linestyle="None", color=incolor, label='Inbending',
                          markerfacecolor=incolor, markersize=mssize),
                   Line2D([0], [0], marker="^",linestyle="None", color=outcolor, label='Outbending',
                          markerfacecolor=outcolor, markersize=mssize2),
                    Line2D([0], [0], color=instat, lw=4, label="Stat Only"),
                    Line2D([0], [0], color=intotal, lw=4, label="Stat + Sys"),
                   ]

# Create the figure
ax.legend(handles=legend_elements, loc='best')

ax.annotate('*Inb. Bins Shifted 3 Degrees to Left', xy =(120, 2.5),
                xytext =(60, 2.5))
  

plt.show()

#plt.legend()



#plt.plot(df["phi"],df[""])

plt.show()



sys.exit()

fs = filestruct.fs()

#df = pd.read_pickle("/mnt/d/GLOBUS/CLAS12/APS2022/pickled_data/1933_rad_Fall_2018_Outbending_100_recon.pkl")#1933_rad_Fall_2018_Outbending_100_gen.pkl")
#df = pd.read_pickle("interactive/dataArrays/full_xsection_outbending_rad_All_All_All_compare_c12_c6_bin_averages.pkl")
#df0 = pd.read_pickle("/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_outbending_rad_All_All_All_for_aps_2022_plots_simplecuts.pkl")
df0 = pd.read_pickle("/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_outbending_norad_All_All_All_for_aps_gen_plots_norad_bigplots.pkl")
for coln in df0.columns:
    print(coln)


qmin = 1.5
xmin = 0.3
tmin = 0.3

df0.replace([np.inf, -np.inf], np.nan, inplace=True)

q2bins,xBbins, tbins, phibins = fs.q2bins[0:8], fs.xBbins[0:12], np.array(fs.tbins[1:]), fs.phibins

for qmin,qmax in zip(q2bins[0:-1],q2bins[1:]):
    print(" \n Q2 bin: {} to {}".format(qmin,qmax))
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


            df = df0.query("qmin=={} and xmin=={} and tmin=={}".format(qmin,xmin,tmin))

            #print(df)
            #y_data = df["GenPtheta"]
            #x_data = df["Nmg"]
            #x_data = df["Genphi1"]
            #y_data = df["Ptheta"]
            #x_data = df["phi1"]
            #var_names = ["phi","theta"]
            #vars = ["vz"]
            #ranges = None
            #ranges = [-8,2,100]
            #ranges = [[0,360,100],[0,100,100]]
            #make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
            #            saveplot=False,pics_dir="none",plot_title="none",logger=False,first_label="rad",
            #            filename="ExamplePlot",units=["",""],extra_data=None)

            # make_histos.plot_1dhist(x_data,vars,ranges=ranges,second_x="none",logger=False,first_label="norad",second_label="norad",
            #             saveplot=False,pics_dir="none",plot_title="none",first_color="blue",sci_on=False)

            w=16
            plt.rcParams["font.size"] = "30"

            fig,ax = plt.subplots(figsize=(17,14))
            x_data = (df["pmin"]+df["pmax"])/2
            #y_data = df["counts_gen"]
            #plt.bar(x_data,y_data,width=w,label="N$_{Gen}$")
            y_data = df["counts_exp"]/df["acc_corr"]
            plt.ylabel("Acc. Corrected Counts")
            plt.xlabel("Phi Bin")
            #y_data = df["acc_corr"]
            #ax.set_yscale("log")
            plt.bar(x_data,y_data,color='green',width=w,label="Acc. Corrected Counts")
            plt.legend()
            #plt.title("Counnts Gen and Rec, Q^2$_{min}$={}, X$_{min}$={}, T$_{min}$={}".format(qmin,xmin,tmin))
            title = "Acc. Corrected Counts, Q$^2_{{{}}}$ = {:.2f}, x$_{{{}}}$ = {:.2f}, t$_{{{}}}$ = {:.1f}".format("min",qmin,"B,min",xmin,"min",tmin)
            plt.title(title)
            plt.savefig("acc_corr_counts/N_gen_rec_Q2_{}_X_{}_T_{}.png".format(qmin,xmin,tmin))
            #plt.show()
            plt.close()

        # # # # #df1 = pd.read_pickle("final_data_files/full_xsection_Sangbaek_rad_CD_sim.pkl")
        # # # # df2 = pd.read_pickle("final_data_files/full_xsection_CD_Included.pkl")

        # # # # df1.to_csv("sang.csv")
        # # # # df2.to_csv("bobby.csv")


        # # # # sys.exit()


        # # # #df = pd.read_pickle("pickled_data/genOnly_sangbaek_sim_genONLY.pkl")
        # # # df = pd.read_pickle("pickled_data/sangbaek_sim_rec_with_cd.pkl")
        # # # #df = pd.read_pickle("/mnt/d/GLOBUS/CLAS12/simulations/production/Fall_2018_Inbending/Raw_Root_Files/Norad/Recon/5000_20210731_2317_norad_recon.root")
        # # # print(df.shape)

        # # # #y_data = df["GenPtheta"]
        # # # #x_data = df["Pvz"]
        # # # #x_data = df["Genphi1"]
        # # # y_data = df["Ptheta"]
        # # # x_data = df["phi1"]
        # # # var_names = ["phi","theta"]
        # # # vars = ["vz"]
        # # # ranges = [-8,2,100]
        # # # ranges = [[0,360,100],[0,100,100]]
        # # # make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
        # # #             saveplot=False,pics_dir="none",plot_title="none",logger=False,first_label="rad",
        # # #             filename="ExamplePlot",units=["",""],extra_data=None)

        # # # # make_histos.plot_1dhist(x_data,vars,ranges=ranges,second_x="none",logger=False,first_label="norad",second_label="norad",
        # # # #             saveplot=False,pics_dir="none",plot_title="none",first_color="blue",sci_on=False)

        # # # sys.exit()

        # # # df_exp = pd.read_pickle("binned_dvpip/f18_in_dvpp_exp_binned.pkl")
        # # # df_rec = pd.read_pickle("binned_dvpip/f18_in_dvpp_rec_binned.pkl")
        # # # df_gen = pd.read_pickle("binned_dvpip/f18_in_dvpp_gen_binned.pkl")

        # # # df_exp = df_exp[df_exp.counts>10]
        # # # df_rec = df_rec[df_rec.counts>10]

        # # # print(df_exp.head(10))
        # # # print(df_rec.head(10))
        # # # print(df_gen.head(10))


        # # # sys.exit()
        # # # #df_loc_gen = "pickled_data/f18_in_gen.pkl"
        # # # df_loc_rec = "pickled_data/f18_in_rec.pkl"
        # # # df_loc_exp = "pickled_data/f18_in_exp.pkl"


        # # # #df_gen = pd.read_pickle(df_loc_gen)
        # # # df_rec = pd.read_pickle(df_loc_rec)
        # # # df_exp = pd.read_pickle(df_loc_exp)


        # # # #ic(df_gen.shape)
        # # # ic(df_rec.shape)
        # # # ic(df_exp.shape)

        # # # df_dvpp_rec = makeDVpi0(df_rec)
        # # # df_dvpp_exp = makeDVpi0(df_exp)

        # # # #print(df_gen.columns)
        # # # ic(df_dvpp_exp.shape)
        # # # ic(df_dvpp_rec.shape)




        # # # sys.exit()

        # # # #raw_f2018_in_data_epgg_no_cuts_no_corrections.pkl
        # # # #raw_f2018_in_data_epgg_no_cuts_with_corrections.pkl
        # # # df_epgg = pd.read_pickle("raw_f2018_in_data_epgg_no_cuts_no_corrections.pkl")
        # # # #df_epgg = pd.read_pickle("raw_f2018_in_data_epgg_no_cuts_with_corrections.pkl")
        # # # print(df_epgg.shape)
        # # # #print(df_epgg.Psector)

        # # # df_epgg.loc[:,"DeltaT"] = df_epgg['t1'] - df_epgg['t2']

        # # # df_dvpi0p = df_epgg




        # # # """
        # # # Old cuts:
        # # # cut_mmep = df_epgg.loc[:, "MM2_ep"] < 0.7  # mmep
        # # # cut_meepgg = df_epgg.loc[:, "ME_epgg"] < 0.7  # meepgg
        # # # cut_mpt = df_epgg.loc[:, "MPt"] < 0.2  # mpt
        # # # cut_recon = df_epgg.loc[:, "reconPi"] < 2  # recon gam angle
        # # # cut_pi0upper = df_epgg.loc[:, "Mpi0"] < 0.2
        # # # cut_pi0lower = df_epgg.loc[:, "Mpi0"] > 0.07
        # # # cut_sector = (df_epgg.loc[:, "Esector"]!=df_epgg.loc[:, "Gsector"]) & (df_epgg.loc[:, "Esector"]!=df_epgg.loc[:, "Gsector2"])

        # # # df_dvpi0 = df_epgg.loc[cut_xBupper & cut_xBlower & cut_Q2 & cut_W & cut_mmep & cut_meepgg &
        # # #                     cut_mpt & cut_recon & cut_pi0upper & cut_pi0lower & cut_sector, :]


        # # # """
        # # # cut_mmep1_FD = df_dvpi0p["MM2_ep"] < 0.7  # mmep
        # # # cut_meepgg1_FD = df_dvpi0p["ME_epgg"] < 0.7  # meepgg
        # # # cut_mpt_FD = df_dvpi0p["MPt"] < 0.2  # mpt
        # # # cut_recon_FD = df_dvpi0p["reconPi"] < 2  # recon gam angle
        # # # cut_pi_mass_min = df_dvpi0p["Mpi0"] > 0.07
        # # # cut_pi_mass_max = df_dvpi0p["Mpi0"] < 0.20
        # # # cut_Esector1 = df_dvpi0p["Esector"]!=df_dvpi0p["Gsector"]
        # # # cut_Esector2 = df_dvpi0p["Esector"]!=df_dvpi0p["Gsector2"]
        # # # cut_Psector_FD = df_dvpi0p.Psector<7

        # # # cut_total = cut_Psector_FD & cut_mmep1_FD & cut_meepgg1_FD & cut_mpt_FD & cut_recon_FD & cut_pi_mass_min & cut_pi_mass_max & cut_Esector1 & cut_Esector2

        # # # df_dvpi0p = df_dvpi0p[cut_total]


        # # # df_dvpi0p.to_pickle("f18_fd_only_dvpi0p_no_corrs_old_cuts.pkl")

        # # # print(df_dvpi0p.shape)

        # # # """
        # # # #common cuts
        # # # cut_xBupper = df_dvpi0p["xB"] < 1  # xB
        # # # cut_xBlower = df_dvpi0p["xB"] > 0  # xB
        # # # cut_Q2 = df_dvpi0p["Q2"] > 1  # Q2
        # # # cut_W = df_dvpi0p["W"] > 2  # W
        # # # cut_Ee = df_dvpi0p["Ee"] > 2  # Ee
        # # # cut_Ge = df_dvpi0p["Ge"] > 3  # Ge
        # # # #cut_Esector = True
        # # # cut_Esector = df_dvpi0p["Esector"]!=df_dvpi0p["Gsector"]
        # # # cut_pi_mass_min = df_dvpi0p["Mpi0"] > 0.07
        # # # cut_pi_mass_max = df_dvpi0p["Mpi0"] < 0.20
        # # # cut_p_FD = df_dvpi0p["Psector"] < 3000
        # # # #cut_Ppmax = df_dvpi0p.Pp < 0.8  # Pp
        # # # # cut_Vz = np.abs(df_dvpi0p["Evz"] - df_dvpi0p["Pvz"]) < 2.5 + 2.5 / mag([df_dvpi0p["Ppx"], pi0SimInb_forDVCS["Ppy"], pi0SimInb_forDVCS["Ppz"]])
        # # # cut_common = cut_xBupper & cut_xBlower & cut_Q2 & cut_W & cut_Ee & cut_Ge & cut_Esector & cut_pi_mass_min & cut_pi_mass_max & cut_p_FD

        # # # df_dvpi0p = df_dvpi0p[cut_common]

        # # # print(df_dvpi0p.shape)


        # # # cut_Pp1_FD = df_dvpi0p.Pp > 0.35  # Pp
        # # # cut_Psector_FD = df_dvpi0p.Psector<7
        # # # cut_Ptheta_FD = df_dvpi0p.Ptheta>2.477
        # # # cut_Gsector_FD = df_dvpi0p.Gsector<7
        # # # cut_mmep2_FD = df_dvpi0p["MM2_ep"] > -0.569  # mmep
        # # # cut_mpi01_FD = df_dvpi0p["Mpi0"] < 0.167  # mpi0
        # # # cut_mpi02_FD = df_dvpi0p["Mpi0"] > 0.104  # mpi0
        # # # cut_mmegg1_FD = df_dvpi0p["MM2_egg"] < 1.771  # mmegg
        # # # cut_mmegg2_FD = df_dvpi0p["MM2_egg"] > -0.0598  # mmegg
        # # # cut_meepgg1_FD = df_dvpi0p["ME_epgg"] < 0.805  # meepgg
        # # # cut_meepgg2_FD = df_dvpi0p["ME_epgg"] > -0.813  # meepgg
        # # # cut_mpt_FD = df_dvpi0p["MPt"] < 0.231  # mpt
        # # # cut_recon_FD = df_dvpi0p["reconPi"] < 1.098  # recon gam angle
        # # # cut_mmepgg2_FD = np.abs(df_dvpi0p["MM2_epgg"]) > -0.0287  # mmepgg

        # # # cut_FD = (cut_Psector_FD & cut_mmep1_FD)
        # # # #cut_FD = (cut_Pp1_FD & cut_Psector_FD & cut_Ptheta_FD & cut_Gsector_FD &
        # # # #            cut_mmep1_FD & cut_mmep2_FD & cut_mpi01_FD & cut_mpi02_FD & 
        # # # #            cut_mmegg1_FD & cut_mmegg2_FD & cut_meepgg1_FD & cut_meepgg2_FD &
        # # # #            cut_mpt_FD & cut_recon_FD & cut_mmepgg1_FD & cut_mmepgg2_FD)

        # # # df_dvpi0p = df_dvpi0p[cut_FD]

        # # # """

        # # # sys.exit()



        # # # x_data = df_dvpi0p["Ppz"]
        # # # plot_title = "Proton Z Momentum vs. t"

        # # # #plot_title = "F 2018 Inbending, epgg, no exclusivity cuts"

        # # # y_data = df_dvpi0p["t1"]
        # # # y_range = [0,1,100]





        # # # # y_data = df["Ptheta"]
        # # # # y_range = [0,50,100]

        # # # var_names = ["t2","Ppz"]

        # # # ranges = [[0,2,100],y_range]
        # # # make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
        # # #             saveplot=False,pics_dir="none",plot_title=plot_title,logger=False,first_label="rad",
        # # #             filename="ExamplePlot",units=["GeV","GeV^2"])





        # # # sys.exit()

        # # # fs = filestruct.fs()


        # # # from utils.epg import epgFromROOT

        # # # fname = "test_new_filter.root"
        # # # converter = epgFromROOT(fname, entry_stop = None, mc = False, rec = False)
        # # # df_after_cuts = converter.df_epgg

        # # # """

        # # # """

        # # # print(df_after_cuts.head(4))



        # # # sys.exit()

        # # # df = pd.read_pickle("F18_All_DVPi0_Events.pkl")

        # # # df_epgg = df.filter(['Epx','Epy','Epz','Ppx','Ppy','Ppz','Gpx','Gpy','Gpz','Gpx2','Gpy2','Gpz2'], axis=1)

        # # # print(df.columns)

        # # # # useful objects
        # # # ele = [df_epgg['Epx'], df_epgg['Epy'], df_epgg['Epz']]
        # # # df_epgg.loc[:, 'Ep'] = mag(ele)
        # # # df_epgg.loc[:, 'Ee'] = getEnergy(ele, me)
        # # # df_epgg.loc[:, 'Etheta'] = getTheta(ele)
        # # # df_epgg.loc[:, 'Ephi'] = getPhi(ele)

        # # # pro = [df_epgg['Ppx'], df_epgg['Ppy'], df_epgg['Ppz']]
        # # # df_epgg.loc[:, 'Pp'] = mag(pro)
        # # # df_epgg.loc[:, 'Pe'] = getEnergy(pro, M)
        # # # df_epgg.loc[:, 'Ptheta'] = getTheta(pro)
        # # # df_epgg.loc[:, 'Pphi'] = getPhi(pro)

        # # # gam = [df_epgg['Gpx'], df_epgg['Gpy'], df_epgg['Gpz']]
        # # # df_epgg.loc[:, 'Gp'] = mag(gam)
        # # # df_epgg.loc[:, 'Ge'] = getEnergy(gam, 0)
        # # # df_epgg.loc[:, 'Gtheta'] = getTheta(gam)
        # # # df_epgg.loc[:, 'Gphi'] = getPhi(gam)

        # # # gam2 = [df_epgg['Gpx2'], df_epgg['Gpy2'], df_epgg['Gpz2']]
        # # # df_epgg.loc[:, 'Gp2'] = mag(gam2)
        # # # df_epgg.loc[:,'Ge2'] = getEnergy(gam2, 0)
        # # # df_epgg.loc[:, 'Gtheta2'] = getTheta(gam2)
        # # # df_epgg.loc[:, 'Gphi2'] = getPhi(gam2)

        # # # Ppt = mag([df_epgg['Ppx'], df_epgg['Ppy'], 0])

        # # # pi0 = vecAdd(gam, gam2)
        # # # VGS = [-df_epgg['Epx'], -df_epgg['Epy'], pbeam - df_epgg['Epz']]
        # # # v3l = cross(beam, ele)
        # # # v3h = cross(pro, VGS)
        # # # costheta = cosTheta(VGS, gam)

        # # # v3g = cross(VGS, gam)
        # # # VmissPi0 = [-df_epgg["Epx"] - df_epgg["Ppx"], -df_epgg["Epy"] -
        # # #             df_epgg["Ppy"], pbeam - df_epgg["Epz"] - df_epgg["Ppz"]]
        # # # VmissP = [-df_epgg["Epx"] - df_epgg["Gpx"] - df_epgg["Gpx2"], -df_epgg["Epy"] -
        # # #             df_epgg["Gpy"] - df_epgg["Gpy2"], pbeam - df_epgg["Epz"] - df_epgg["Gpz"] - df_epgg["Gpz2"]]
        # # # Vmiss = [-df_epgg["Epx"] - df_epgg["Ppx"] - df_epgg["Gpx"] - df_epgg["Gpx2"],
        # # #             -df_epgg["Epy"] - df_epgg["Ppy"] - df_epgg["Gpy"] - df_epgg["Gpy2"],
        # # #             pbeam - df_epgg["Epz"] - df_epgg["Ppz"] - df_epgg["Gpz"] - df_epgg["Gpz2"]]

        # # # df_epgg.loc[:, 'Mpx'], df_epgg.loc[:, 'Mpy'], df_epgg.loc[:, 'Mpz'] = Vmiss

        # # # # binning kinematics
        # # # df_epgg.loc[:,'Q2'] = -((ebeam - df_epgg['Ee'])**2 - mag2(VGS))
        # # # df_epgg.loc[:,'nu'] = (ebeam - df_epgg['Ee'])
        # # # df_epgg.loc[:,'xB'] = df_epgg['Q2'] / 2.0 / M / df_epgg['nu']
        # # # df_epgg.loc[:,'t1'] = 2 * M * (df_epgg['Pe'] - M)
        # # # df_epgg.loc[:,'t2'] = (M * df_epgg['Q2'] + 2 * M * df_epgg['nu'] * (df_epgg['nu'] - np.sqrt(df_epgg['nu'] * df_epgg['nu'] + df_epgg['Q2']) * costheta))\
        # # # / (M + df_epgg['nu'] - np.sqrt(df_epgg['nu'] * df_epgg['nu'] + df_epgg['Q2']) * costheta)

        # # # df_epgg.loc[:,'W'] = np.sqrt(np.maximum(0, (ebeam + M - df_epgg['Ee'])**2 - mag2(VGS)))
        # # # df_epgg.loc[:,'MPt'] = np.sqrt((df_epgg["Epx"] + df_epgg["Ppx"] + df_epgg["Gpx"] + df_epgg["Gpx2"])**2 +
        # # #                             (df_epgg["Epy"] + df_epgg["Ppy"] + df_epgg["Gpy"] + df_epgg["Gpy2"])**2)

        # # # # exclusivity variables
        # # # df_epgg.loc[:,'MM2_ep'] = (-M - ebeam + df_epgg["Ee"] +
        # # #                         df_epgg["Pe"])**2 - mag2(VmissPi0)
        # # # df_epgg.loc[:,'MM2_egg'] = (-M - ebeam + df_epgg["Ee"] +
        # # #                         df_epgg["Ge"] + df_epgg["Ge2"])**2 - mag2(VmissP)
        # # # df_epgg.loc[:,'MM2_epgg'] = (-M - ebeam + df_epgg["Ee"] + df_epgg["Pe"] +
        # # #                         df_epgg["Ge"] + df_epgg["Ge2"])**2 - mag2(Vmiss)
        # # # df_epgg.loc[:,'ME_epgg'] = (M + ebeam - df_epgg["Ee"] - df_epgg["Pe"] - df_epgg["Ge"] - df_epgg["Ge2"])
        # # # df_epgg.loc[:,'Mpi0'] = pi0InvMass(gam, gam2)
        # # # df_epgg.loc[:,'reconPi'] = angle(VmissPi0, pi0)
        # # # df_epgg.loc[:,"Pie"] = df_epgg['Ge'] + df_epgg['Ge2']

        # # # df_epgg.loc[:,"DeltaT"] = df_epgg['t1'] - df_epgg['t2']




        # # # x_data = df_epgg["Pp"]
        # # # plot_title = "Proton Z Momentum vs. t"

        # # # #plot_title = "F 2018 Inbending, epgg, no exclusivity cuts"

        # # # y_data = df_epgg["DeltaT"]
        # # # y_range = [-0.4,0.4,70]





        # # # # y_data = df["Ptheta"]
        # # # # y_range = [0,50,100]

        # # # var_names = ["t2","Ppz"]

        # # # ranges = [[0,2,70],y_range]
        # # # make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
        # # #             saveplot=False,pics_dir="none",plot_title=plot_title,logger=False,first_label="rad",
        # # #             filename="ExamplePlot",units=["GeV","GeV^2"])
