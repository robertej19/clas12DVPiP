import pandas as pd
import numpy as np
import argparse, os, sys
from utils import filestruct

fs = filestruct.fs()

def makeDVpi0P(df_epgg, pol = "inbending",proton_loc="All",photon1_loc="All",photon2_loc="All",simple_exclusivity_cuts=False,
                                  unique_identifyer="", datafilename="temporary_exclusivity_variances_",use_generic_cuts=True,
                                  sigma_multiplier=3):


    if simple_exclusivity_cuts:

        df_epgg.loc[:, "closeness"] = np.abs(df_epgg.loc[:, "Mpi0"] - .1349766)

        #Mandatory cuts
        cut_xBupper = df_epgg.loc[:, "xB"] < 1  # xB
        cut_xBlower = df_epgg.loc[:, "xB"] > 0  # xB
        cut_Q2 = (df_epgg.loc[:, "Q2"] > 1)# & (df_epgg.loc[:, "Q2"] <1.5)  # Q2
        cut_W = df_epgg.loc[:, "W"] > 2  # W

        cut_etheta_discrep = df_epgg.loc[:, "Etheta"] > 9 # W


        cut_Ptheta_low = df_epgg.loc[:, "Ptheta"] > 5  # W
        cut_Ptheta_hi = df_epgg.loc[:, "Ptheta"] < 65  # W
        cut_Ptheta_mid = df_epgg.loc[:, "Ptheta"] < 42#35  # W
        cut_Ptheta_mid2 = df_epgg.loc[:, "Ptheta"] >42  # W


        cut_Pphi_low = df_epgg.loc[:, "Pphi"] > -360  # W
        cut_Pphi_hi = df_epgg.loc[:, "Pphi"] < 360  # W


        cut_Ptheta_low_mid_hi = (cut_Ptheta_low & cut_Ptheta_mid)|(cut_Ptheta_mid2 & cut_Ptheta_hi)


        cut_Gpz2_low = df_epgg.loc[:, "Gpz2"] > 0.4#0.4  # Gpz2
        cut_Gpz_hi = df_epgg.loc[:, "Gpz"] < 6000  # Gpz2




        #Collaboration approved
        cut_pi0upper = df_epgg.loc[:, "Mpi0"] < 0.168
        cut_pi0lower = df_epgg.loc[:, "Mpi0"] > 0.096
        cut_recon = df_epgg.loc[:, "reconPi"] < 2  # recon gam angle
        cut_mpt_hi = df_epgg.loc[:, "MPt"] < 0.2  # mpt
        cut_mpt_low = df_epgg.loc[:, "MPt"] > -0.2  # mpt
        cut_mmepgg_hi = np.abs(df_epgg["MM2_epgg"]) < 0.05#0.0440  # mmepgg
        cut_mmepgg_low = np.abs(df_epgg["MM2_epgg"]) > -0.05#-0.0478  # mmepgg

        #Extra

        cut_mmep_hi = df_epgg.loc[:, "MM2_ep"] < 0.2#0.7  # mmep
        cut_mmep_low = df_epgg.loc[:, "MM2_ep"] > -0.196#-0.66  # mmep
        cut_mmegg_hi = df_epgg.loc[:, "MM2_egg"] < 1.6 #0.95  # mm_egg
        cut_mmegg_low = df_epgg.loc[:, "MM2_egg"] > 0.16 # 0.8  # mm_egg
        cut_meepgg_hi = df_epgg.loc[:, "ME_epgg"] < 0.5  # meepgg
        cut_meepgg_low = df_epgg.loc[:, "ME_epgg"] > -0.5  # meepgg






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

        df_dvpi0 = df_dvpi0.sort_values(by=['closeness', 'Psector', 'Gsector'], ascending = [True, True, True])
        df_dvpi0 = df_dvpi0.loc[~df_dvpi0.event.duplicated(), :]
        df_dvpi0 = df_dvpi0.sort_values(by='event')        

        return df_dvpi0


    else:
        print("Executing advanced DVPiP Cuts")
        print("polarity is {}".format(pol))
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

        print("USING POLARITY {}".format(pol))
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


        parser = argparse.ArgumentParser(description="Get args",formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument("-f","--fname", help="a single root file to convert into pickles")
        parser.add_argument("-o","--out", help="a single pickle file name as an output", default="excuting.pkl")

        args = parser.parse_args()


        input_dir = fs.inb_norad_rec_epgg_dir
        output_dir = fs.inb_norad_rec_dvpip_dir

        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        for file in files:
                print("Ex. cut on {}".format(input_dir+file))
                output_name = output_dir+"dvpip_events_"+file
                print("saving to {}".format(output_name))
                df_dvpi0p = makeDVpi0P(pd.read_pickle(input_dir+file),pol = "inbending")
                df_dvpi0p.to_pickle(output_name)
               

#     #df = makeDVpi0P(pd.read_pickle(args.fname))
    
#     #dir = "/mnt/d/GLOBUS/CLAS12/Thesis/1_potential_dvpip_events/inb/rec/"
#     dir = "/mnt/d/GLOBUS/CLAS12/Thesis/1_potential_dvpip_events/outb/rec/"


    
#     #fname = "norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon_fid_corr_smear"
#     fname = "norad_10000_20230703_1814_Fall_2018_Outbending_100_50nA_recon_fid_corr_smear"

#     df = makeDVpi0P(pd.read_pickle(dir+fname+".pkl"))

#     print(df)
    
#     #df.to_pickle("final_inbending_exclusive.pkl")
#     #df.to_pickle("/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/inb/rec/{}.pkl".format(fname))
#     df.to_pickle("/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/outb/rec/{}.pkl".format(fname))


    #df_exp = pd.read_pickle("new_exp_convert_outbend.pkl",pol="outbending")
    #df_rec = pd.read_pickle("new_rec_convert_outbend_rad.pkl",pol="outbending")

    #df_dvpi0p_exp = makeDVpi0P(df_exp)
    #df_dvpi0p_rec = makeDVpi0P(df_rec)

    #df_dvpi0p_exp.to_pickle("new_exp_dvpi0p_outbend.pkl")
    #df_dvpi0p_rec.to_pickle("new_rec_dvpi0p_outbend_rad.pkl")
