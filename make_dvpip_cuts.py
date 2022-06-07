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

def makeDVpi0(df_epgg,proton_loc="All",photon1_loc="All",photon2_loc="All"):

        df_epgg.loc[:, "closeness"] = np.abs(df_epgg.loc[:, "Mpi0"] - .1349766)

        #Mandatory cuts
        cut_xBupper = df_epgg.loc[:, "xB"] < 1  # xB
        cut_xBlower = df_epgg.loc[:, "xB"] > 0  # xB
        cut_Q2 = (df_epgg.loc[:, "Q2"] > 1)# & (df_epgg.loc[:, "Q2"] <1.5)  # Q2
        cut_W = df_epgg.loc[:, "W"] > 2  # W

        cut_Ptheta_low = df_epgg.loc[:, "Ptheta"] > 0  # W
        cut_Ptheta_hi = df_epgg.loc[:, "Ptheta"] < 65  # W
        cut_Ptheta_mid = df_epgg.loc[:, "Ptheta"] < 35  # W
        cut_Ptheta_mid2 = df_epgg.loc[:, "Ptheta"] >45  # W

        cut_Pphi_low = df_epgg.loc[:, "Pphi"] > -360  # W
        cut_Pphi_hi = df_epgg.loc[:, "Pphi"] < 360  # W


        cut_Ptheta_low_mid_hi = (cut_Ptheta_low & cut_Ptheta_mid)|(cut_Ptheta_mid2 & cut_Ptheta_hi)



        cut_Gpz2_low = df_epgg.loc[:, "Gpz2"] > 0#0.4  # Gpz2
        cut_Gpz_hi = df_epgg.loc[:, "Gpz"] < 6000  # Gpz2

        cut_pi0upper = df_epgg.loc[:, "Mpi0"] < 0.2
        cut_pi0lower = df_epgg.loc[:, "Mpi0"] > 0.07
        cut_mmep_hi = df_epgg.loc[:, "MM2_ep"] < 0.2#0.7  # mmep
        cut_mmep_low = df_epgg.loc[:, "MM2_ep"] > -0.196#-0.66  # mmep
        cut_mmegg_hi = df_epgg.loc[:, "MM2_egg"] < 1.6 #0.95  # mm_egg
        cut_mmegg_low = df_epgg.loc[:, "MM2_egg"] > 0.16 # 0.8  # mm_egg
        cut_meepgg_hi = df_epgg.loc[:, "ME_epgg"] < 0.7  # meepgg
        cut_meepgg_low = df_epgg.loc[:, "ME_epgg"] > -0.7  # meepgg
        cut_mpt_hi = df_epgg.loc[:, "MPt"] < 0.2  # mpt
        cut_mpt_low = df_epgg.loc[:, "MPt"] > -0.2  # mpt
        cut_recon = df_epgg.loc[:, "reconPi"] < 2  # recon gam angle
        cut_mmepgg_hi = np.abs(df_epgg["MM2_epgg"]) < 0.05#0.0440  # mmepgg
        cut_mmepgg_low = np.abs(df_epgg["MM2_epgg"]) > -0.05#-0.0478  # mmepgg

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
                                cut_photon2, :]

        #cut_sector = (df_epgg.loc[:, "Esector"]!=df_epgg.loc[:, "Gsector"]) & (df_epgg.loc[:, "Esector"]!=df_epgg.loc[:, "Gsector2"])
        #cut_Vz = np.abs(df_epgg["Evz"] - df_epgg["Pvz"]) < 2.5 + 2.5 / mag([df_epgg["Ppx"], df_epgg["Ppy"], df_epgg["Ppz"]])

        

        # proton reconstruction quality
        #cut_proton = (df_epgg.loc[:, "Psector"]<7) & (df_epgg.loc[:, "Ptheta"]<35)
        # # # cut_proton = True
        # # # cut_FD_proton = (df_epgg.loc[:, "Psector"]<7)# & (df_epgg.loc[:, "Ptheta"]<35)
        # # # cut_CD_proton = (df_epgg.loc[:, "Psector"]>7)# & (df_epgg.loc[:, "Ptheta"]>45) & (df_epgg.loc[:, "Ptheta"]<65)

        # # # cut_FD_photon = df_epgg.loc[:, "Gsector"]<7 #Gsector
        # # # cut_FT_photon = df_epgg.loc[:, "Gsector"]>7 #Gsector
        # # # cut_FD_photon2 = df_epgg.loc[:, "Gsector2"]<7 #Gsector
        # # # cut_FT_photon2 = df_epgg.loc[:, "Gsector2"]>7 #Gsector



        # # # #cut_proton = (cut_FD_proton)|(cut_CD_proton)
        # # # #cut_proton = cut_FD_proton
        # # # #cut_proton = cut_CD_proton

               

        # # # # Exclusivity cuts
        # # # cut_mmep_hi = df_epgg.loc[:, "MM2_ep"] < 0.3#0.1  # mmep
        # # # cut_mmep_low = df_epgg.loc[:, "MM2_ep"] > -0.2#-0.06  # mmep


        # # # cut_mmegg_hi = df_epgg.loc[:, "MM2_egg"] < 1.2 #0.95  # mm_egg
        # # # cut_mmegg_low = df_epgg.loc[:, "MM2_egg"] > 0.6 # 0.8  # mm_egg

        
        # # # cut_meepgg = df_epgg.loc[:, "ME_epgg"] < 0.1  # meepgg
        # # # cut_meepgg_neg = df_epgg.loc[:, "ME_epgg"] > -0.1  # meepgg

        # # # cut_mpt = df_epgg.loc[:, "MPt"] < 0.1  # mpt
        # # # cut_recon = df_epgg.loc[:, "reconPi"] < 2  # recon gam angle
        # # # cut_pi0upper = df_epgg.loc[:, "Mpi0"] < 0.2#0.161 #0.2
        # # # cut_pi0lower = df_epgg.loc[:, "Mpi0"] > 0.07# 0.115 #0.07
        # # # #cut_sector = (df_epgg.loc[:, "Esector"]!=df_epgg.loc[:, "Gsector"]) & (df_epgg.loc[:, "Esector"]!=df_epgg.loc[:, "Gsector2"])
        # # # #cut_Vz = np.abs(df_epgg["Evz"] - df_epgg["Pvz"]) < 2.5 + 2.5 / mag([df_epgg["Ppx"], df_epgg["Ppy"], df_epgg["Ppz"]])

        # # # df_dvpi0 = df_epgg.loc[cut_xBupper & cut_mmep_hi & cut_mmep_low &
        # # #                 cut_FD_photon & cut_FT_photon2 & 
        # # #                 cut_proton_mom & cut_gamma_mom & cut_gamma_mom2 &
        # # #                 cut_mmegg_hi & cut_mmegg_low & cut_meepgg & cut_meepgg_neg & 
        # # #                 cut_meepgg_neg & cut_xBlower & cut_Q2 & cut_W & 
        # # #                 cut_proton & cut_mpt & cut_recon & 
        # # #                 cut_pi0upper & cut_pi0lower, :]

        #For an event, there can be two gg's passed conditions above.
        #Take only one gg's that makes pi0 invariant mass
        #This case is very rare.
        #For now, duplicated proton is not considered.
        df_dvpi0 = df_dvpi0.sort_values(by=['closeness', 'Psector', 'Gsector'], ascending = [True, True, True])
        df_dvpi0 = df_dvpi0.loc[~df_dvpi0.event.duplicated(), :]
        df_dvpi0 = df_dvpi0.sort_values(by='event')        

        return df_dvpi0


if __name__ == "__main__":
        df_loc_rec = "pickled_data/f18_bkmrg_in_rec.pkl"
        df_rec = pd.read_pickle(df_loc_rec)
        ic(df_rec.shape)
        df_dvpp_rec = makeDVpi0(df_rec)
        ic(df_dvpp_rec.shape)
        df_dvpp_rec.to_pickle("pickled_dvpip/f18_bkmrg_in_dvpp_rec_noseccut.pkl")














