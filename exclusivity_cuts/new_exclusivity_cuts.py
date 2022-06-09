import os, sys
import pandas as pd
import numpy as np
from utils import make_histos
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def fit_function(x, A, beta, B, mu, sigma):
        return (A * np.exp(-x/beta) + B * np.exp(-1.0 * (x - mu)**2 / (2 * sigma**2)))


def fit_distribution():

        
        # 3.) Generate exponential and gaussian data and histograms.
        data = np.random.exponential(scale=2.0, size=100000)
        data2 = np.random.normal(loc=3.0, scale=0.3, size=15000)
        bins = np.linspace(0, 6, 61)
        data_entries_1, bins_1 = np.histogram(data, bins=bins)
        data_entries_2, bins_2 = np.histogram(data2, bins=bins)

        # 4.) Add histograms of exponential and gaussian data.
        data_entries = data_entries_1 + data_entries_2
        binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])

        # 5.) Fit the function to the histogram data.
        popt, pcov = curve_fit(fit_function, xdata=binscenters, ydata=data_entries, p0=[20000, 2.0, 2000, 3.0, 0.3])
        print(popt)

        # 6.)
        # Generate enough x values to make the curves look smooth.
        xspace = np.linspace(0, 6, 100000)

        # Plot the histogram and the fitted function.
        plt.bar(binscenters, data_entries, width=bins[1] - bins[0], color='navy', label=r'Histogram entries')
        plt.plot(xspace, fit_function(xspace, *popt), color='darkorange', linewidth=2.5, label=r'Fitted function')

        # Make the plot nicer.
        plt.xlim(0,6)
        plt.xlabel(r'x axis')
        plt.ylabel(r'Number of entries')
        plt.title(r'Exponential decay with gaussian peak')
        plt.legend(loc='best')
        plt.show()
        plt.clf()

def makeDVpi0P(df_epgg, pol = "inbending",data_type="exp",proton_loc="All",photon1_loc="All",photon2_loc="All",simple_exclusivity_cuts=False):

        #Variables listing:

        ex_vars = ["Mpi0",
                "reconPi",
                "MPt",
                "MM2_epgg",
                "MM2_ep",
                "MM2_egg",
                "ME_epgg"]
        

        other_vars = ["xB",
                "Q2",
                "W",
                "Etheta",
                "Ptheta",
                "Pphi",
                "Gpz",
                "Gpz2",
                "Psector",
                "Gsector",
                "Gsector2"]

        # for x_key in ex_vars:
        #         x_data = df_epgg[x_key]
        #         output_dir = "./"
        #         make_histos.plot_1dhist(x_data,[x_key,],ranges="none",second_x=False,first_label=data_type,logger=True,
        #                         saveplot=True,pics_dir=output_dir+"hists_1D/no_cuts/",plot_title=x_key+", "+data_type+"Before Cuts")
        

        #common cuts:
        df_epgg.loc[:, "closeness"] = np.abs(df_epgg.loc[:, "Mpi0"] - .1349766)
        cut_xBupper = df_epgg.loc[:, "xB"] < 1  # xB
        cut_xBlower = df_epgg.loc[:, "xB"] > 0  # xB
        cut_Q2 = (df_epgg.loc[:, "Q2"] > 1)# & (df_epgg.loc[:, "Q2"] <1.5)  # Q2
        cut_W = df_epgg.loc[:, "W"] > 2  # W

        df_epgg = df_epgg.loc[cut_xBupper & cut_xBlower & cut_Q2 & cut_W]

         #Collaboration approved
        cut_pi0upper = df_epgg.loc[:, "Mpi0"] < 0.168
        cut_pi0lower = df_epgg.loc[:, "Mpi0"] > 0.096
        cut_Mpi0 = cut_pi0lower & cut_pi0upper

        cut_recon = df_epgg.loc[:, "reconPi"] < 2  # recon gam angle

        cut_mpt_hi = df_epgg.loc[:, "MPt"] < 0.2  # mpt
        cut_mpt_low = df_epgg.loc[:, "MPt"] > -0.2  # mpt
        cut_mpt = cut_mpt_hi & cut_mpt_low

        cut_mmepgg_hi = np.abs(df_epgg["MM2_epgg"]) < 0.05#0.0440  # mmepgg
        cut_mmepgg_low = np.abs(df_epgg["MM2_epgg"]) > -0.05#-0.0478  # mmepgg
        cut_mmepgg = cut_mmepgg_hi & cut_mmepgg_low
        #Extra

        cut_mmep_hi = df_epgg.loc[:, "MM2_ep"] < 0.2#0.7  # mmep
        cut_mmep_low = df_epgg.loc[:, "MM2_ep"] > -0.196#-0.66  # mmep
        cut_mmep = cut_mmep_hi & cut_mmep_low

        cut_mmegg_hi = df_epgg.loc[:, "MM2_egg"] < 1.6 #0.95  # mm_egg
        cut_mmegg_low = df_epgg.loc[:, "MM2_egg"] > 0.16 # 0.8  # mm_egg
        cut_mmegg = cut_mmegg_hi & cut_mmegg_low

        cut_meepgg_hi = df_epgg.loc[:, "ME_epgg"] < 0.5  # meepgg
        cut_meepgg_low = df_epgg.loc[:, "ME_epgg"] > -0.5  # meepgg
        cut_meepgg = cut_meepgg_hi & cut_meepgg_low


        ex_cuts = [cut_Mpi0,
                cut_recon,
                cut_mpt,
                cut_mmepgg,
                cut_mmep,
                cut_mmegg,
                cut_meepgg]

        ex_cuts_names = ["cut_Mpi0",
                "cut_recon",
                "cut_mpt",
                "cut_mmepgg",
                "cut_mmep",
                "cut_mmegg",
                "cut_meepgg"]

        ex_cuts_dict = {"cut_Mpi0":cut_Mpi0,
                "cut_recon":cut_recon,
                "cut_mpt":cut_mpt,
                "cut_mmepgg":cut_mmepgg,
                "cut_mmep":cut_mmep,
                "cut_mmegg":cut_mmegg,
                "cut_meepgg":cut_meepgg}

        ex_cuts_ranges = [ [0.05,0.25,100],
                 [-0.1,5,100],
                 [0,0.5,100],
                 [-0.06,0.06,100],
                 [-1,1,100],
                 [-0.2,2.0,100],
                 [-1,1,100]]

        ex_cuts2 = [cut_Mpi0]

        var_names = []
        mu_values = []
        sigmasquared_values = []

        for xind,x0_key in enumerate(ex_cuts_names):
                df_sample = df_epgg
                #print(xind)
                #print(x0_key)
                cut_key_list = [x for x in ex_cuts_names if ex_cuts_names.index(x) != xind]
                #print(cut_key_list)
                for key in cut_key_list:
                        df_sample = df_sample.loc[ex_cuts_dict[key]]
                


                # sys.exit()
                
                # for cut in cut_list:
                #         df_sample = df_sample.loc[cut]
                if 0 == 1:
                        for x_index,x_key in enumerate(ex_vars):
                                x_data = df_sample[x_key]
                                output_dir = "./"
                                ranges = ex_cuts_ranges[x_index]

                                make_histos.plot_1dhist(x_data,[x_key,],ranges=ranges,second_x=False,first_label=data_type,logger=False,
                                                saveplot=True,pics_dir=output_dir+"hists_1D/all_cuts_but_{}/".format(x0_key),plot_title=x_key+", "+data_type+" All Cuts Except "+x0_key)
                if 1 == 1:
                        
                        
                        x_data = df_sample[ex_vars[ex_cuts_names.index(x0_key)]]



                        output_dir = "./"
                        ranges = ex_cuts_ranges[xind]



                        popt, pcov = make_histos.plot_1dhist(x_data,[x0_key,],ranges=ranges,second_x=False,first_label=data_type,logger=False,
                                        saveplot=True,pics_dir=output_dir+"hists_1D/",plot_title=ex_vars[ex_cuts_names.index(x0_key)]+", "+data_type+" All Cuts Except "+ex_vars[ex_cuts_names.index(x0_key)],fitdata=True)

                        # print(ex_vars[ex_cuts_names.index(x0_key)],
                        #                 "A = {}, Mu = {}, SigmaSquared = {}".format(*popt),
                        #                 "CovMatrix = {}".format(pcov))

                        var_names.append(ex_vars[ex_cuts_names.index(x0_key)])
                        mu_values.append(popt[1])
                        sigmasquared_values.append(popt[2])

                               
#                df_sample = df_epgg.loc[cut_xBupper & cut_xBlower & cut_Q2 & cut_W]

        print(type(mu_values[0]))
        q = np.column_stack([var_names,mu_values,np.sqrt(sigmasquared_values)])
        #print(q)
        df = pd.DataFrame (q, columns = ['var_name', 'mu', 'sigma'])
        #print (type(df.mu.values(0)))
        df['mu'] = df['mu'].astype(float)
        df['sigma'] = df['sigma'].astype(float)


        df.loc[:,"mu+3sigma"] = df.loc[:,"mu"] + 3*df.loc[:,"sigma"]
        df.loc[:,"mu-3sigma"] = df.loc[:,"mu"] - 3*df.loc[:,"sigma"]

        df.loc[:,"mu+2sigma"] = df.loc[:,"mu"] + 2*df.loc[:,"sigma"]
        df.loc[:,"mu-2sigma"] = df.loc[:,"mu"] - 2*df.loc[:,"sigma"]

        df.loc[:,"mu+4sigma"] = df.loc[:,"mu"] + 4*df.loc[:,"sigma"]
        df.loc[:,"mu-4sigma"] = df.loc[:,"mu"] - 4*df.loc[:,"sigma"]

        print(df)
        df.to_pickle("temporary_exclusivity_variances.pkl")

        sys.exit()
                

                # df_dvpi0 = df_epgg.loc[cut_xBupper & cut_xBlower & cut_Q2 & cut_W & 
                #                         cut_Ptheta_low_mid_hi &
                #                         cut_Pphi_low &
                #                         cut_Pphi_hi &
                #                         cut_pi0upper & 
                #                         cut_pi0lower & 
                #                         cut_mmep_hi & 
                #                         cut_mmep_low & 
                #                         cut_mmegg_hi & 
                #                         cut_mmegg_low & 
                #                         cut_meepgg_hi & 
                #                         cut_meepgg_low & 
                #                         cut_mpt_hi & 
                #                         cut_mpt_low & 
                #                         cut_recon & 
                #                         cut_mmepgg_hi & 
                #                         cut_mmepgg_low & 
                #                         cut_proton &
                #                         cut_photon1 &
                #                         cut_Gpz2_low &
                #                         cut_Gpz_hi &
                #                         cut_etheta_discrep &
                #                         cut_photon2, :]



        for x_key in ex_vars:
                x_data = df_epgg[x_key]
                output_dir = "./"
                make_histos.plot_1dhist(x_data,[x_key,],ranges="none",second_x=False,first_label=data_type,logger=False,
                                saveplot=True,pics_dir=output_dir+"hists_1D/all_cuts/",plot_title=x_key+", "+data_type+"All Cuts Except "+x_key)



        df_dvpi0 = df_dvpi0.sort_values(by=['closeness', 'Psector', 'Gsector'], ascending = [True, True, True])
        df_dvpi0 = df_dvpi0.loc[~df_dvpi0.event.duplicated(), :]
        df_dvpi0 = df_dvpi0.sort_values(by='event')        

        return df_dvpi0

        if simple_exclusivity_cuts:

                #Angle cuts
                cut_etheta_discrep = df_epgg.loc[:, "Etheta"] > 9 # W
                cut_Ptheta_low = df_epgg.loc[:, "Ptheta"] > 5  # W
                cut_Ptheta_hi = df_epgg.loc[:, "Ptheta"] < 65  # W
                cut_Ptheta_mid = df_epgg.loc[:, "Ptheta"] < 42#35  # W
                cut_Ptheta_mid2 = df_epgg.loc[:, "Ptheta"] >42  # W
                cut_Pphi_low = df_epgg.loc[:, "Pphi"] > -360  # W
                cut_Pphi_hi = df_epgg.loc[:, "Pphi"] < 360  # W
                cut_Gpz2_low = df_epgg.loc[:, "Gpz2"] > 0.4#0.4  # Gpz2
                cut_Gpz_hi = df_epgg.loc[:, "Gpz"] < 6000  # Gpz2
                cut_Ptheta_low_mid_hi = (cut_Ptheta_low & cut_Ptheta_mid)|(cut_Ptheta_mid2 & cut_Ptheta_hi)

                #Optional cuts
                cut_FD_proton = (df_epgg.loc[:, "Psector"]<7)# & (df_epgg.loc[:, "Ptheta"]<35)
                cut_CD_proton = (df_epgg.loc[:, "Psector"]>7)# & (df_epgg.loc[:, "Ptheta"]>45) & (df_epgg.loc[:, "Ptheta"]<65)
                
                cut_FD_photon1 = df_epgg.loc[:, "Gsector"]<7 #Gsector
                cut_FT_photon1 = df_epgg.loc[:, "Gsector"]>7 #Gsector
                cut_FD_photon2 = df_epgg.loc[:, "Gsector2"]<7 #Gsector
                cut_FT_photon2 = df_epgg.loc[:, "Gsector2"]>7 #Gsector


                if proton_loc == "FD":
                        cut_proton = cut_FD_proton
                elif proton_loc == "CD":
                        cut_proton = cut_CD_proton
                else:
                        cut_proton = True
                
                if photon1_loc == "FD":
                        cut_photon1 = cut_FD_photon1
                elif photon1_loc == "FT":
                        cut_photon1 = cut_FT_photon1
                else:
                        cut_photon1 = True

                if photon2_loc == "FD":
                        cut_photon2 = cut_FD_photon2
                elif photon2_loc == "FT":
                        cut_photon2 = cut_FT_photon2
                else:
                        cut_photon2 = True

                df_dvpi0 = df_epgg.loc[cut_xBupper & cut_xBlower & cut_Q2 & cut_W & 
                                        cut_Ptheta_low_mid_hi &
                                        cut_Pphi_low &
                                        cut_Pphi_hi &
                                        cut_pi0upper & 
                                        cut_pi0lower & 
                                        cut_mmep_hi & 
                                        cut_mmep_low & 
                                        cut_mmegg_hi & 
                                        cut_mmegg_low & 
                                        cut_meepgg_hi & 
                                        cut_meepgg_low & 
                                        cut_mpt_hi & 
                                        cut_mpt_low & 
                                        cut_recon & 
                                        cut_mmepgg_hi & 
                                        cut_mmepgg_low & 
                                        cut_proton &
                                        cut_photon1 &
                                        cut_Gpz2_low &
                                        cut_Gpz_hi &
                                        cut_etheta_discrep &
                                        cut_photon2, :]




        else:
                #make dvpi0 pairs
                df_dvpi0p = df_epgg

                Ge2Threshold = 0.4
                CD_Ptheta_ub = 65
                CD_Ptheta_lb = 42
                FD_Ptheta_ub = 35
                FD_Ptheta_lb = 5

                #common cuts
                cut_xBupper = df_dvpi0p.loc[:, "xB"] < 1  # xB
                cut_xBlower = df_dvpi0p.loc[:, "xB"] > 0  # xB
                cut_Q2 = df_dvpi0p.loc[:, "Q2"] > 1  # Q2
                cut_W = df_dvpi0p.loc[:, "W"] > 2  # W
                cut_Ee = df_dvpi0p["Ee"] > 2  # Ee
                cut_Ge2 = df_dvpi0p["Ge2"] > Ge2Threshold  # Ge2 Threshold.
                cut_Esector = (df_dvpi0p["Esector"]!=df_dvpi0p["Gsector"]) & (df_dvpi0p["Esector"]!=df_dvpi0p["Gsector2"])
                cut_Psector = ~( ((df_dvpi0p["Pstat"]//10)%10>0) & (df_dvpi0p["Psector"]==df_dvpi0p["Gsector"]) ) & ~( ((df_dvpi0p["Pstat"]//10)%10>0) & (df_dvpi0p["Psector"]==df_dvpi0p["Gsector2"]) )
                cut_Ppmax = df_dvpi0p.Pp < 1.6  # Pp
                cut_Pthetamin = df_dvpi0p.Ptheta > 0  # Ptheta
                # cut_Vz = np.abs(df_dvcs["Evz"] - df_dvcs["Pvz"]) < 2.5 + 2.5 / mag([df_dvcs["Ppx"], df_dvcs["Ppy"], df_dvcs["Ppz"]])

                cut_etheta_discrep = df_dvpi0p.loc[:, "Etheta"] > 0.0009 # W

                cut_common = cut_xBupper & cut_xBlower & cut_Q2 & cut_W & cut_Ee & cut_Ge2 & cut_Esector & cut_Psector & cut_Ppmax & cut_Pthetamin & cut_etheta_discrep

                df_dvpi0p = df_dvpi0p[cut_common]

                # proton reconstruction quality
                # cut_FD_proton = (df_epgg.loc[:, "Psector"]<7) & (df_epgg.loc[:, "Ptheta"]<35)
                # cut_CD_proton = (df_epgg.loc[:, "Psector"]>7) & (df_epgg.loc[:, "Ptheta"]>45) & (df_epgg.loc[:, "Ptheta"]<65)
                # cut_proton = (cut_FD_proton)|(cut_CD_proton)
                cut_proton = 1

                df_dvpi0p.loc[:, "config"] = 0

                if pol == "inbending":
                        #CDFT
                        cut_Pp1_CDFT = df_dvpi0p.Pp > 0.3  # Pp
                        cut_Psector_CDFT = df_dvpi0p.Psector>7
                        cut_Ptheta1_CDFT = df_dvpi0p.Ptheta<CD_Ptheta_ub
                        cut_Ptheta2_CDFT = df_dvpi0p.Ptheta>CD_Ptheta_lb
                        cut_Gsector_CDFT = df_dvpi0p.Gsector>7
                        cut_GFid_CDFT = df_dvpi0p.GFid==1
                        cut_GFid2_CDFT = df_dvpi0p.GFid2==1
                        cut_PFid_CDFT = df_dvpi0p.PFid==1
                        cut_mpi01_CDFT = df_dvpi0p["Mpi0"] < 0.149#0.157  # mpi0
                        cut_mpi02_CDFT = df_dvpi0p["Mpi0"] > 0.126#0.118  # mpi0
                        cut_mmep1_CDFT = df_dvpi0p["MM2_ep"] < 0.610#0.914  # mmep
                        cut_mmep2_CDFT = df_dvpi0p["MM2_ep"] > -0.384#-0.715  # mmep
                        cut_mmegg1_CDFT = df_dvpi0p["MM2_egg"] < 1.641#2.155  # mmegg
                        cut_mmegg2_CDFT = df_dvpi0p["MM2_egg"] > 0.0974#-0.417  # mmegg
                        cut_meepgg1_CDFT = df_dvpi0p["ME_epgg"] < 0.481#0.799  # meepgg
                        cut_meepgg2_CDFT = df_dvpi0p["ME_epgg"] > -0.474#-0.792  # meepgg
                        cut_mpt_CDFT = df_dvpi0p["MPt"] < 0.1272#0.189  # mpt
                        cut_recon_CDFT = df_dvpi0p["reconPi"] < 0.955  # recon gam angle
                        cut_coplanarity_CDFT = df_dvpi0p["coplanarity"] < 9.259#15.431  # coplanarity angle
                        cut_mmepgg1_CDFT = df_dvpi0p["MM2_epgg"] < 0.02564#0.0440  # mmepgg
                        cut_mmepgg2_CDFT = df_dvpi0p["MM2_epgg"] > -0.02944#-0.0478  # mmepgg

                        cut_CDFT = (cut_Pp1_CDFT & cut_Psector_CDFT & cut_Ptheta1_CDFT & cut_Ptheta2_CDFT & cut_Gsector_CDFT & 
                                        cut_GFid_CDFT & cut_GFid2_CDFT & cut_PFid_CDFT &
                                        cut_mmep1_CDFT & cut_mmep2_CDFT & cut_mpi01_CDFT & cut_mpi02_CDFT & 
                                        cut_mmegg1_CDFT & cut_mmegg2_CDFT & cut_meepgg1_CDFT & cut_meepgg2_CDFT &
                                        cut_mpt_CDFT & cut_recon_CDFT & cut_coplanarity_CDFT & cut_mmepgg1_CDFT & cut_mmepgg2_CDFT)


                        #CD
                        cut_Pp1_CD = df_dvpi0p.Pp > 0.3  # Pp
                        cut_Psector_CD = df_dvpi0p.Psector>7
                        cut_Ptheta1_CD = df_dvpi0p.Ptheta<CD_Ptheta_ub
                        cut_Ptheta2_CD = df_dvpi0p.Ptheta>CD_Ptheta_lb
                        cut_Gsector_CD = (df_dvpi0p.Gsector<7) & (df_dvpi0p.Gsector>0)
                        cut_Gsector2_CD = (df_dvpi0p.Gsector2<7) & (df_dvpi0p.Gsector2>0)
                        cut_GFid_CD = df_dvpi0p.GFid==1
                        cut_GFid2_CD = df_dvpi0p.GFid2==1
                        cut_PFid_CD = df_dvpi0p.PFid==1
                        cut_mpi01_CD = df_dvpi0p["Mpi0"] < 0.162  # mpi0
                        cut_mpi02_CD = df_dvpi0p["Mpi0"] > 0.107  # mpi0
                        cut_mmep1_CD = df_dvpi0p["MM2_ep"] < 0.354  # mmep
                        cut_mmep2_CD = df_dvpi0p["MM2_ep"] > -0.283  # mmep
                        cut_mmegg1_CD = df_dvpi0p["MM2_egg"] < 1.922  # mmegg
                        cut_mmegg2_CD = df_dvpi0p["MM2_egg"] > 0.007  # mmegg
                        cut_meepgg1_CD = df_dvpi0p["ME_epgg"] < 0.822  # meepgg
                        cut_meepgg2_CD = df_dvpi0p["ME_epgg"] > -0.677  # meepgg
                        cut_mpt_CD = df_dvpi0p["MPt"] < 0.176  # mpt
                        cut_recon_CD = df_dvpi0p["reconPi"] < 1.476  # recon gam angle
                        cut_coplanarity_CD = df_dvpi0p["coplanarity"] < 10.203  # coplanarity angle
                        cut_mmepgg1_CD = df_dvpi0p["MM2_epgg"] < 0.0208  # mmepgg
                        cut_mmepgg2_CD = df_dvpi0p["MM2_epgg"] > -0.0250  # mmepgg

                        cut_CD = (cut_Pp1_CD & cut_Psector_CD & cut_Ptheta1_CD & cut_Ptheta2_CD & cut_Gsector_CD & cut_Gsector2_CD & 
                                        cut_GFid_CD & cut_GFid2_CD & cut_PFid_CD &
                                        cut_mmep1_CD & cut_mmep2_CD & cut_mpi01_CD & cut_mpi02_CD & 
                                        cut_mmegg1_CD & cut_mmegg2_CD & cut_meepgg1_CD & cut_meepgg2_CD &
                                        cut_mpt_CD & cut_recon_CD & cut_coplanarity_CD & cut_mmepgg1_CD & cut_mmepgg2_CD)

                        #FD
                        cut_Pp1_FD = df_dvpi0p.Pp > 0.42  # Pp
                        cut_Psector_FD = df_dvpi0p.Psector<7
                        cut_Ptheta1_FD = df_dvpi0p.Ptheta<FD_Ptheta_ub
                        cut_Ptheta2_FD = df_dvpi0p.Ptheta>FD_Ptheta_lb
                        cut_Gsector_FD = (df_dvpi0p.Gsector<7) & (df_dvpi0p.Gsector>0)
                        cut_Gsector2_FD = (df_dvpi0p.Gsector2<7) & (df_dvpi0p.Gsector2>0)
                        cut_GFid_FD = df_dvpi0p.GFid==1
                        cut_GFid2_FD = df_dvpi0p.GFid2==1
                        cut_PFid_FD = df_dvpi0p.PFid==1
                        cut_mpi01_FD = df_dvpi0p["Mpi0"] < 0.178  # mpi0
                        cut_mpi02_FD = df_dvpi0p["Mpi0"] > 0.0910  # mpi0
                        cut_mmep1_FD = df_dvpi0p["MM2_ep"] < 0.335  # mmep
                        cut_mmep2_FD = df_dvpi0p["MM2_ep"] > -0.271  # mmep
                        cut_mmegg1_FD = df_dvpi0p["MM2_egg"] < 1.762  # mmegg
                        cut_mmegg2_FD = df_dvpi0p["MM2_egg"] > 0.117  # mmegg
                        cut_meepgg1_FD = df_dvpi0p["ME_epgg"] < 0.816 # meepgg
                        cut_meepgg2_FD = df_dvpi0p["ME_epgg"] > -0.685  # meepgg
                        cut_mpt_FD = df_dvpi0p["MPt"] < 0.180  # mpt
                        cut_recon_FD = df_dvpi0p["reconPi"] < 1.363  # recon gam angle
                        cut_coplanarity_FD = df_dvpi0p["coplanarity"] < 9.190  # coplanarity angle
                        cut_mmepgg1_FD = df_dvpi0p["MM2_epgg"] < 0.0189  # mmepgg
                        cut_mmepgg2_FD = df_dvpi0p["MM2_epgg"] > -0.0224  # mmepgg

                        cut_FD = (cut_Pp1_FD & cut_Psector_FD & cut_Ptheta1_FD & cut_Ptheta2_FD & cut_Gsector_FD & cut_Gsector2_FD &
                                        cut_GFid_FD & cut_GFid2_FD & cut_PFid_FD &
                                        cut_mmep1_FD & cut_mmep2_FD & cut_mpi01_FD & cut_mpi02_FD & 
                                        cut_mmegg1_FD & cut_mmegg2_FD & cut_meepgg1_FD & cut_meepgg2_FD &
                                        cut_mpt_FD & cut_recon_FD & cut_coplanarity_FD & cut_mmepgg1_FD & cut_mmepgg2_FD)

                elif pol == "outbending":
                        #CDFT
                        cut_Pp1_CDFT = df_dvpi0p.Pp > 0.3  # Pp
                        cut_Psector_CDFT = df_dvpi0p.Psector>7
                        cut_Ptheta1_CDFT = df_dvpi0p.Ptheta<CD_Ptheta_ub
                        cut_Ptheta2_CDFT = df_dvpi0p.Ptheta>CD_Ptheta_lb
                        cut_Gsector_CDFT = df_dvpi0p.Gsector>7
                        cut_GFid_CDFT = df_dvpi0p.GFid==1
                        cut_GFid2_CDFT = df_dvpi0p.GFid2==1
                        cut_PFid_CDFT = df_dvpi0p.PFid==1
                        cut_mpi01_CDFT = df_dvpi0p["Mpi0"] < 0.151#0.160  # mpi0
                        cut_mpi02_CDFT = df_dvpi0p["Mpi0"] > 0.124#0.115  # mpi0
                        cut_mmep1_CDFT = df_dvpi0p["MM2_ep"] < 0.575#0.892  # mmep
                        cut_mmep2_CDFT = df_dvpi0p["MM2_ep"] > -0.378#-0.694  # mmep
                        cut_mmegg1_CDFT = df_dvpi0p["MM2_egg"] < 1.665#2.184  # mmegg
                        cut_mmegg2_CDFT = df_dvpi0p["MM2_egg"] > 0.107#-0.412  # mmegg
                        cut_meepgg1_CDFT = df_dvpi0p["ME_epgg"] < 0.514#0.844  # meepgg
                        cut_meepgg2_CDFT = df_dvpi0p["ME_epgg"] > -0.476#-0.806  # meepgg
                        cut_mpt_CDFT = df_dvpi0p["MPt"] < 0.146#0.210  # mpt
                        cut_recon_CDFT = df_dvpi0p["reconPi"] < 1.114#1.630  # recon gam angle
                        cut_coplanarity_CDFT = df_dvpi0p["coplanarity"] < 10.69#17.817  # coplanarity angle
                        cut_mmepgg1_CDFT = df_dvpi0p["MM2_epgg"] < 0.0324#0.0549  # mmepgg
                        cut_mmepgg2_CDFT = df_dvpi0p["MM2_epgg"] > -0.035#-0.0575  # mmepgg

                        cut_CDFT = (cut_Pp1_CDFT & cut_Psector_CDFT & cut_Ptheta1_CDFT & cut_Ptheta2_CDFT & cut_Gsector_CDFT & 
                                        cut_GFid_CDFT & cut_GFid2_CDFT & cut_PFid_CDFT &
                                        cut_mmep1_CDFT & cut_mmep2_CDFT & cut_mpi01_CDFT & cut_mpi02_CDFT & 
                                        cut_mmegg1_CDFT & cut_mmegg2_CDFT & cut_meepgg1_CDFT & cut_meepgg2_CDFT &
                                        cut_mpt_CDFT & cut_recon_CDFT & cut_coplanarity_CDFT & cut_mmepgg1_CDFT & cut_mmepgg2_CDFT)


                        #CD
                        cut_Pp1_CD = df_dvpi0p.Pp > 0.3  # Pp
                        cut_Psector_CD = df_dvpi0p.Psector>7
                        cut_Ptheta1_CD = df_dvpi0p.Ptheta<CD_Ptheta_ub
                        cut_Ptheta2_CD = df_dvpi0p.Ptheta>CD_Ptheta_lb
                        cut_Gsector_CD = (df_dvpi0p.Gsector<7) & (df_dvpi0p.Gsector>0)
                        cut_Gsector2_CD = (df_dvpi0p.Gsector2<7) & (df_dvpi0p.Gsector2>0)
                        cut_GFid_CD = df_dvpi0p.GFid==1
                        cut_GFid2_CD = df_dvpi0p.GFid2==1
                        cut_PFid_CD = df_dvpi0p.PFid==1
                        cut_mpi01_CD = df_dvpi0p["Mpi0"] < 0.163  # mpi0
                        cut_mpi02_CD = df_dvpi0p["Mpi0"] > 0.106  # mpi0
                        cut_mmep1_CD = df_dvpi0p["MM2_ep"] < 0.294  # mmep
                        cut_mmep2_CD = df_dvpi0p["MM2_ep"] > -0.218  # mmep
                        cut_mmegg1_CD = df_dvpi0p["MM2_egg"] < 1.876  # mmegg
                        cut_mmegg2_CD = df_dvpi0p["MM2_egg"] > -0.0142  # mmegg
                        cut_meepgg1_CD = df_dvpi0p["ME_epgg"] < 0.700  # meepgg
                        cut_meepgg2_CD = df_dvpi0p["ME_epgg"] > -0.597  # meepgg
                        cut_mpt_CD = df_dvpi0p["MPt"] < 0.194  # mpt
                        cut_recon_CD = df_dvpi0p["reconPi"] < 1.761  # recon gam angle
                        cut_coplanarity_CD = df_dvpi0p["coplanarity"] < 9.530  # coplanarity angle
                        cut_mmepgg1_CD = df_dvpi0p["MM2_epgg"] < 0.0182  # mmepgg
                        cut_mmepgg2_CD = df_dvpi0p["MM2_epgg"] > -0.0219  # mmepgg

                        cut_CD = (cut_Pp1_CD & cut_Psector_CD & cut_Ptheta1_CD & cut_Ptheta2_CD & cut_Gsector_CD & cut_Gsector2_CD & 
                                        cut_GFid_CD & cut_GFid2_CD & cut_PFid_CD &
                                        cut_mmep1_CD & cut_mmep2_CD & cut_mpi01_CD & cut_mpi02_CD & 
                                        cut_mmegg1_CD & cut_mmegg2_CD & cut_meepgg1_CD & cut_meepgg2_CD &
                                        cut_mpt_CD & cut_recon_CD & cut_coplanarity_CD & cut_mmepgg1_CD & cut_mmepgg2_CD)

                        #FD
                        cut_Pp1_FD = df_dvpi0p.Pp > 0.5  # Pp
                        cut_Psector_FD = df_dvpi0p.Psector<7
                        cut_Ptheta1_FD = df_dvpi0p.Ptheta<FD_Ptheta_ub
                        cut_Ptheta2_FD = df_dvpi0p.Ptheta>FD_Ptheta_lb
                        cut_Gsector_FD = (df_dvpi0p.Gsector<7) & (df_dvpi0p.Gsector>0)
                        cut_Gsector2_FD = (df_dvpi0p.Gsector2<7) & (df_dvpi0p.Gsector2>0)
                        cut_GFid_FD = df_dvpi0p.GFid==1
                        cut_GFid2_FD = df_dvpi0p.GFid2==1
                        cut_PFid_FD = df_dvpi0p.PFid==1
                        cut_mpi01_FD = df_dvpi0p["Mpi0"] < 0.164  # mpi0
                        cut_mpi02_FD = df_dvpi0p["Mpi0"] > 0.105  # mpi0
                        cut_mmep1_FD = df_dvpi0p["MM2_ep"] < 0.323  # mmep
                        cut_mmep2_FD = df_dvpi0p["MM2_ep"] > -0.256  # mmep
                        cut_mmegg1_FD = df_dvpi0p["MM2_egg"] < 1.828  # mmegg
                        cut_mmegg2_FD = df_dvpi0p["MM2_egg"] > 0.0491  # mmegg
                        cut_meepgg1_FD = df_dvpi0p["ME_epgg"] < 0.754  # meepgg
                        cut_meepgg2_FD = df_dvpi0p["ME_epgg"] > -0.583  # meepgg
                        cut_mpt_FD = df_dvpi0p["MPt"] < 0.177  # mpt
                        cut_recon_FD = df_dvpi0p["reconPi"] < 1.940  # recon gam angle
                        cut_coplanarity_FD = df_dvpi0p["coplanarity"] < 7.498  # coplanarity angle
                        cut_mmepgg1_FD = df_dvpi0p["MM2_epgg"] < 0.0195  # mmepgg
                        cut_mmepgg2_FD = df_dvpi0p["MM2_epgg"] > -0.0240  # mmepgg

                        cut_FD = (cut_Pp1_FD & cut_Psector_FD & cut_Ptheta1_FD & cut_Ptheta2_FD & cut_Gsector_FD & cut_Gsector2_FD &
                                        cut_GFid_FD & cut_GFid2_FD & cut_PFid_FD & 
                                        cut_mmep1_FD & cut_mmep2_FD & cut_mpi01_FD & cut_mpi02_FD & 
                                        cut_mmegg1_FD & cut_mmegg2_FD & cut_meepgg1_FD & cut_meepgg2_FD &
                                        cut_mpt_FD & cut_recon_FD & cut_coplanarity_FD & cut_mmepgg1_FD & cut_mmepgg2_FD)

                        df_dvpi0p.loc[cut_CDFT, "config"] = 3
                        df_dvpi0p.loc[cut_CD, "config"] = 2
                        df_dvpi0p.loc[cut_FD, "config"] = 1

                        df_dvpi0p = df_dvpi0p[df_dvpi0p.config>0]

                        #For an event, there can be two gg's passed conditions above.
                        #Take only one gg's that makes pi0 invariant mass
                        #This case is very rare.
                        #For now, duplicated proton is not considered.
                        df_dvpi0p = df_dvpi0p.sort_values(by=['closeness', 'Psector', 'Gsector'], ascending = [True, True, True])
                        df_dvpi0p = df_dvpi0p.loc[~df_dvpi0p.event.duplicated(), :]
                        df_dvpi0p = df_dvpi0p.sort_values(by='event') 

                        print("Number of events before cuts: {}".format(df_epgg.shape[0])) 
                        print("Number of events after cuts: {}".format(df_dvpi0p.shape[0])) 
                return df_dvpi0p

if __name__ == "__main__":
    test_df = pd.read_pickle("test_df_epgg.pkl")
    #print(test_df)
    df_dvpi0p_exp = makeDVpi0P(test_df)

    #df_exp = pd.read_pickle("new_exp_convert_outbend.pkl",pol="outbending")
    #df_rec = pd.read_pickle("new_rec_convert_outbend_rad.pkl",pol="outbending")

    #df_dvpi0p_exp = makeDVpi0P(df_exp)
    #df_dvpi0p_rec = makeDVpi0P(df_rec)

    ##df_dvpi0p_exp.to_pickle("new_exp_dvpi0p_outbend.pkl")
    #df_dvpi0p_rec.to_pickle("new_rec_dvpi0p_outbend_rad.pkl")
