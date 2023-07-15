import pandas as pd
import numpy as np
import os, sys
from icecream import ic
import matplotlib.pyplot as plt

from utils import filestruct, const, make_histos


import os
from PIL import Image
import numpy as np
import re

"""
df_noFID = pd.read_pickle("protonRec_noFID.pkl")
df_nosmear = pd.read_pickle("protonRec_nosmear.pkl")
df_withFID = pd.read_pickle("protonRec_withFID.pkl")
df_yessmear = pd.read_pickle("protonRec_yessmear.pkl")
"""

dir_base ="/mnt/d/GLOBUS/CLAS12/Thesis/1_potential_dvpip_events/inb/rec/smear_corr_files/"

"""
Load these all into dataframes:
norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon_fid_corr_nosmear.pkl
norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon_fid_corr_smear.pkl
norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon_fid_nocorr_nosmear.pkl
norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon_nofid_nocorr_nosmear.pkl
"""
test = 0
if test:
    df_norad_nf_nc_ns = pd.read_pickle("quicktest.pkl")
else:
    #df_norad_nf_nc_ns = pd.read_pickle(dir_base+"norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon_nofid_nocorr_nosmear.pkl")#.head(100000)
    #df_norad_nf_nc_ns.to_pickle("quicktest.pkl")

    
    #df_norad_f_nc_ns = pd.read_pickle(dir_base+"norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon_fid_nocorr_nosmear.pkl")
    df_norad_f_c_ns = pd.read_pickle(dir_base+"norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon_fid_corr_nosmear.pkl")
    #df_norad_f_c_s = pd.read_pickle(dir_base+"norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon_fid_corr_smear.pkl")

#print(df_norad_nf_nc_ns.columns.values)

"""
['Epx' 'Epy' 'Epz' 'Eedep' 'Esector' 'event' 'Ep' 'ESamplFrac' 'Ppx' 'Ppy'
 'Ppz' 'Pstat' 'Psector' 'Pp' 'Ptheta' 'Pphi' 'PFid' 'Gpx' 'Gpy' 'Gpz'
 'Gedep' 'Gsector' 'GFid' 'Gp' 'Gtheta' 'Gphi' 'GSamplFrac' 'Gpx2' 'Gpy2'
 'Gpz2' 'Gedep2' 'Gsector2' 'GFid2' 'Gp2' 'Gtheta2' 'Gphi2' 'GSamplFrac2'
 'Ee' 'Etheta' 'Ephi' 'Pe' 'Ge' 'Ge2' 'Mpx' 'Mpy' 'Mpz' 'Q2' 'nu' 'xB' 'y'
 't1' 't2' 'W' 'MPt' 'phi1' 'phi2' 'MM2_ep' 'MM2_egg' 'MM2_epgg' 'ME_epgg'
 'Mpi0' 'reconPi' 'Pie' 'coplanarity' 'coneAngle1' 'coneAngle2'
 'openingAngle' 'closeness' 'GenEpx' 'GenEpy' 'GenEpz' 'GenEp' 'GenEtheta'
 'GenEphi' 'GenPpx' 'GenPpy' 'GenPpz' 'GenPp' 'GenPtheta' 'GenPphi'
 'GenGpx' 'GenGpy' 'GenGpz' 'GenGp' 'GenGp2' 'GenGpx2' 'GenGpy2' 'GenGpz2'
 'GenGtheta' 'GenGphi' 'GenGtheta2' 'GenGphi2' 'GenEe' 'GenPe' 'GenGe'
 'GenGe2' 'GenPipx' 'GenPipy' 'GenPipz' 'GenPip' 'GenPie' 'GenPitheta'
 'GenPiphi' 'GenMpx' 'GenMpy' 'GenMpz' 'GenQ2' 'Gennu' 'Geny' 'GenxB'
 'Gent1' 'Gent2' 'GenW' 'Genphi1' 'Genphi2' 'GenMPt' 'GenMM2_epgg'
 'GenMM2_ep' 'GenMM2_egg' 'GenME_epgg' 'GenconeAngle' 'GenreconGam'
 'Gencoplanarity' 'GenconeAngle1' 'GenconeAngle2' 'GenopeningAngle']
"""

df = df_norad_f_c_ns

x_data = df['Pp']
y_data = df['GenPp']-df['Pp']
var_names = ['Proton Momentum','$\delta$ Momentum (Gen-Rec)']
ranges = [[0,3,200],[-.1,.1,200]]

make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
            saveplot=True,pics_dir="none",plot_title="Proton Momentum After Correction",logger=False,first_label="rad",
            filename="ExamplePlot",units=["GeV/c","GeV/c"],extra_data=None)