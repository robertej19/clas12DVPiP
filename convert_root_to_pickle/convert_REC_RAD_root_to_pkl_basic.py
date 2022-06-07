#!/usr/bin/env python3
"""
A simple script to save data in pickle.
"""

import uproot
import pandas as pd
import numpy as np
import argparse
from copy import copy
from utils.const import *
from utils.physics import *

M = 0.938272081 # target mass
me = 0.5109989461 * 0.001 # electron mass
m_pi0 = 0.1349768
ebeam = 10.604 # beam energy
pbeam = np.sqrt(ebeam * ebeam - me * me) # beam electron momentum
beam = [0, 0, pbeam] # beam vector
target = [0, 0, 0] # target vector







class root2pickle():
    #class to read root to make epg pairs, inherited from epg
    def __init__(self, fname, entry_start = None, entry_stop = None, pol = "inbending", gen = "pi0rad", raw = False, detRes = False, dvcs = False):
        self.fname = fname
        self.readEPGG(entry_start = entry_start, entry_stop = entry_stop, pol = pol, gen = gen, detRes = detRes)
        self.saveDVpi0vars()

    def readFile(self):
        #read root using uproot
        self.file = uproot.open(self.fname)
        self.tree = self.file["T"]

    def closeFile(self):
        #close file for saving memory
        self.file = None
        self.tree = None

    def readEPGG(self, entry_start = None, entry_stop = None, gen = "pi0rad", pol = "inbending", detRes = False):
        #save data into df_epg, df_epgg for parent class epg
        self.readFile()

        # data frames and their keys to read Z part
        df_electronGen = pd.DataFrame()
        df_protonGen = pd.DataFrame()
        df_gammaGen = pd.DataFrame()
        eleKeysGen = ["GenEpx", "GenEpy", "GenEpz"]
        if detRes:
            eleKeysGen = ["GenEpx", "GenEpy", "GenEpz", "GenEvx", "GenEvy", "GenEvz"]
        proKeysGen = ["GenPpx", "GenPpy", "GenPpz"]
        gamKeysGen = ["GenGpx", "GenGpy", "GenGpz"]
        pi0KeysGen = ["GenPipx", "GenPipy", "GenPipz"]

        # read keys
        for key in eleKeysGen:
            df_electronGen[key] = self.tree[key].array(library="pd", entry_start = entry_start, entry_stop=entry_stop)
        for key in proKeysGen:
            df_protonGen[key] = self.tree[key].array(library="pd", entry_start = entry_start, entry_stop=entry_stop)
        for key in gamKeysGen:
            df_gammaGen[key] = self.tree[key].array(library="pd", entry_start = entry_start, entry_stop=entry_stop)

        #convert data type to standard double
        # df_electronGen = df_electronGen.astype({"GenEpx": float, "GenEpy": float, "GenEpz": float, "GenEvx": float, "GenEvy": float, "GenEvz": float})
        df_electronGen = df_electronGen.astype({"GenEpx": float, "GenEpy": float, "GenEpz": float})
        df_protonGen = df_protonGen.astype({"GenPpx": float, "GenPpy": float, "GenPpz": float})
        df_gammaGen = df_gammaGen.astype({"GenGpx": float, "GenGpy": float, "GenGpz": float})

        #set up a dummy index for merging
        df_electronGen.loc[:,'event'] = df_electronGen.index
        df_protonGen.loc[:,'event'] = df_protonGen.index
        df_gammaGen.loc[:,'event'] = df_gammaGen.index.get_level_values('entry')

        #sort columns for readability
        if detRes:
            df_electronGen = df_electronGen.loc[:, ["event", "GenEpx", "GenEpy", "GenEpz", "GenEvx", "GenEvy", "GenEvz"]]
        else:
            df_electronGen = df_electronGen.loc[:, ["event", "GenEpx", "GenEpy", "GenEpz"]]

        #spherical coordinates
        eleGen = [df_electronGen["GenEpx"], df_electronGen["GenEpy"], df_electronGen["GenEpz"]]
        df_electronGen.loc[:, 'GenEp'] = mag(eleGen)
        df_electronGen.loc[:, 'GenEtheta'] = getTheta(eleGen)
        df_electronGen.loc[:, 'GenEphi'] = getPhi(eleGen)

        proGen = [df_protonGen["GenPpx"], df_protonGen["GenPpy"], df_protonGen["GenPpz"]]
        df_protonGen.loc[:, 'GenPp'] = mag(proGen)
        df_protonGen.loc[:, 'GenPtheta'] = getTheta(proGen)
        df_protonGen.loc[:, 'GenPphi'] = getPhi(proGen)

        df_MC = pd.merge(df_electronGen, df_protonGen, how='inner', on='event')

        df_pi0Gen = pd.DataFrame()
        for key in pi0KeysGen:
            df_pi0Gen[key] = self.tree[key].array(library="pd", entry_start = entry_start, entry_stop=entry_stop)
        df_pi0Gen = df_pi0Gen.astype({"GenPipx": float, "GenPipy": float, "GenPipz": float})
        df_pi0Gen.loc[:,'event'] = df_pi0Gen.index
        #two g's to one gg.
        pi0Gen = [df_pi0Gen["GenPipx"], df_pi0Gen["GenPipy"], df_pi0Gen["GenPipz"]]
        df_pi0Gen.loc[:, 'GenPip'] = mag(pi0Gen)
        df_pi0Gen.loc[:, 'GenPitheta'] = getTheta(pi0Gen)
        df_pi0Gen.loc[:, 'GenPiphi'] = getPhi(pi0Gen)

        df_gammaGen = df_gammaGen[df_gammaGen.index.get_level_values('subentry')==0]
        gamGen = [df_gammaGen["GenGpx"], df_gammaGen["GenGpy"], df_gammaGen["GenGpz"]]
        df_gammaGen.loc[:, 'GenGp'] = mag(gamGen)
        df_gammaGen.loc[:, 'GenGtheta'] = getTheta(gamGen)
        df_gammaGen.loc[:, 'GenGphi'] = getPhi(gamGen)

        df_MC = pd.merge(df_MC, df_gammaGen, how='inner', on='event')
        df_MC = pd.merge(df_MC, df_pi0Gen, how='inner', on='event')
        
        ele = [df_MC['GenEpx'], df_MC['GenEpy'], df_MC['GenEpz']]
        pro = [df_MC['GenPpx'], df_MC['GenPpy'], df_MC['GenPpz']]
        pi0 = [df_MC['GenPipx'], df_MC['GenPipy'], df_MC['GenPipz']]
        gam = [df_MC['GenGpx'], df_MC['GenGpy'], df_MC['GenGpz']]

        df_MC.loc[:, 'GenEe'] = getEnergy(ele, me)
        df_MC.loc[:, 'GenPe'] = getEnergy(pro, M)
        df_MC.loc[:, 'GenGe'] = getEnergy(gam, 0)
        df_MC.loc[:, 'GenPie'] = getEnergy(pi0, m_pi0)



        Ppt = mag([df_MC['GenPpx'], df_MC['GenPpy'], 0])
        VGS = [-df_MC['GenEpx'], -df_MC['GenEpy'], pbeam - df_MC['GenEpz']]
        v3l = cross(beam, ele)
        v3h = cross(pro, VGS)
        v3g = cross(VGS, pi0)
        VmissG = [-df_MC["GenEpx"] - df_MC["GenPpx"], -df_MC["GenEpy"] - df_MC["GenPpy"],
                    pbeam - df_MC["GenEpz"] - df_MC["GenPpz"]]
        VmissP = [-(df_MC["GenEpx"] + df_MC["GenGpx"]), -(df_MC["GenEpy"] + df_MC["GenGpy"]),
                    -(-pbeam + df_MC["GenEpz"] + df_MC["GenGpz"])]
        Vmiss = [-(df_MC["GenEpx"] + df_MC["GenPpx"] + df_MC["GenGpx"]), -(df_MC["GenEpy"] + df_MC["GenPpy"] + df_MC["GenGpy"]),
                    -(-pbeam + df_MC["GenEpz"] + df_MC["GenPpz"] + df_MC["GenGpz"])]
        costheta = cosTheta(VGS, pi0)
        df_MC.loc[:, 'GenMpx'], df_MC.loc[:, 'GenMpy'], df_MC.loc[:, 'GenMpz'] = Vmiss
        # binning kinematics
        df_MC.loc[:,'GenQ2'] = -((ebeam - df_MC['GenEe'])**2 - mag2(VGS))
        df_MC.loc[:,'Gennu'] = (ebeam - df_MC['GenEe'])
        df_MC.loc[:,'Geny'] = df_MC['Gennu']/ebeam
        df_MC.loc[:,'GenxB'] = df_MC['GenQ2'] / 2.0 / M / df_MC['Gennu']
        df_MC.loc[:,'Gent1'] = 2 * M * (df_MC['GenPe'] - M)
        df_MC.loc[:,'Gent2'] = (M * df_MC['GenQ2'] + 2 * M * df_MC['Gennu'] * (df_MC['Gennu'] - np.sqrt(df_MC['Gennu'] * df_MC['Gennu'] + df_MC['GenQ2']) * costheta))\
        / (M + df_MC['Gennu'] - np.sqrt(df_MC['Gennu'] * df_MC['Gennu'] + df_MC['GenQ2']) * costheta)
        df_MC.loc[:,'GenW'] = np.sqrt(np.maximum(0, (ebeam + M - df_MC['GenEe'])**2 - mag2(VGS)))
        # trento angles
        df_MC.loc[:,'Genphi1'] = angle(v3l, v3h)
        df_MC.loc[:,'Genphi1'] = np.where(dot(v3l, pro) > 0, 360.0 -
                                    df_MC['Genphi1'], df_MC['Genphi1'])
        df_MC.loc[:,'Genphi2'] = angle(v3l, v3g)
        df_MC.loc[:,'Genphi2'] = np.where(dot(v3l, gam) <
                                    0, 360.0 - df_MC['Genphi2'], df_MC['Genphi2'])
        # exclusivity variables
        df_MC.loc[:,'GenMM2_epg'] = (-M - ebeam + df_MC["GenEe"] +
                                df_MC["GenPe"] + df_MC["GenGe"])**2 - mag2(Vmiss)
        df_MC.loc[:,'GenME_epg'] = (M + ebeam - df_MC["GenEe"] - df_MC["GenPe"] - df_MC["GenGe"])
        df_MC.loc[:,'GenMM2_ep'] = (-M - ebeam + df_MC["GenEe"] + df_MC["GenPe"])**2 - mag2(VmissG)
        df_MC.loc[:,'GenMM2_eg'] = (-M - ebeam + df_MC["GenEe"] + df_MC["GenGe"])**2 - mag2(VmissP)
        df_MC.loc[:,'GenMPt'] = np.sqrt((df_MC["GenEpx"] + df_MC["GenPpx"] + df_MC["GenGpx"])**2 +
                                (df_MC["GenEpy"] + df_MC["GenPpy"] + df_MC["GenGpy"])**2)
        df_MC.loc[:,'GenconeAngle'] = angle(ele, gam)
        df_MC.loc[:,'GenreconGam'] = angle(gam, VmissG)
        df_MC.loc[:,'Gencoplanarity'] = angle(v3h, v3g)
                
        self.df_MC = df_MC    #done with saving z



        print("generator mode: ", gen)
        print("debug:: number of events", len(df_electronGen))
        print("debug:: number of all MC df", len(df_MC))

        # data frames and their keys to read X part
        df_electronRec = pd.DataFrame()
        df_protonRec = pd.DataFrame()
        df_gammaRec = pd.DataFrame()
        eleKeysRec = ["Epx", "Epy", "Epz", "Esector"]
        proKeysRec = ["Ppx", "Ppy", "Ppz", "Pstat", "Psector"]
        proKeysRec.extend(["PDc1Hitx", "PDc1Hity", "PDc1Hitz"])
        gamKeysRec = ["Gpx", "Gpy", "Gpz", "GcX", "GcY", "Gsector"]

        # read them
        for key in eleKeysRec:
            df_electronRec[key] = self.tree[key].array(library="pd", entry_start = entry_start, entry_stop=entry_stop)
        for key in proKeysRec:
            df_protonRec[key] = self.tree[key].array(library="pd", entry_start = entry_start, entry_stop=entry_stop)
        for key in gamKeysRec:
            df_gammaRec[key] = self.tree[key].array(library="pd", entry_start = entry_start, entry_stop=entry_stop)

        self.closeFile()

        #convert data type to standard double
        df_electronRec = df_electronRec.astype({"Epx": float, "Epy": float, "Epz": float})
        df_protonRec = df_protonRec.astype({"Ppx": float, "Ppy": float, "Ppz": float})
        df_gammaRec = df_gammaRec.astype({"Gpx": float, "Gpy": float, "Gpz": float, "GcX": float, "GcY": float})


        #set up a dummy index for merging
        df_electronRec.loc[:,'event'] = df_electronRec.index
        df_protonRec.loc[:,'event'] = df_protonRec.index.get_level_values('entry')
        df_gammaRec.loc[:,'event'] = df_gammaRec.index.get_level_values('entry')
        df_gammaRec.loc[:,'GIndex'] = df_gammaRec.index.get_level_values('subentry')

        #save only FD protons and photons
        # df_protonRec = df_protonRec[df_protonRec["Psector"]<7]
        #proton momentum correction
        pro = [df_protonRec['Ppx'], df_protonRec['Ppy'], df_protonRec['Ppz']]
        df_protonRec.loc[:, 'Pp'] = mag(pro)
        df_protonRec.loc[:, 'Ptheta'] = getTheta(pro)
        df_protonRec.loc[:, 'Pphi'] = getPhi(pro)

        #moduli proton phi
        # df_protonRec.loc[:, "Pphi"] = np.where(df_protonRec.loc[:, "Pphi"]>180, df_protonRec.loc[:, "Pphi"] - 360, df_protonRec.loc[:, "Pphi"]) 
        # df_protonRec.loc[:, "Pphi"] = np.where(df_protonRec.loc[:, "Pphi"]<-180, df_protonRec.loc[:, "Pphi"] + 360, df_protonRec.loc[:, "Pphi"]) 

        df_protonRec.loc[:, "Ppx"] = df_protonRec.loc[:, "Pp"]*np.sin(np.radians(df_protonRec.loc[:, "Ptheta"]))*np.cos(np.radians(df_protonRec.loc[:, "Pphi"]))
        df_protonRec.loc[:, "Ppy"] = df_protonRec.loc[:, "Pp"]*np.sin(np.radians(df_protonRec.loc[:, "Ptheta"]))*np.sin(np.radians(df_protonRec.loc[:, "Pphi"]))
        df_protonRec.loc[:, "Ppz"] = df_protonRec.loc[:, "Pp"]*np.cos(np.radians(df_protonRec.loc[:, "Ptheta"]))

        pro = [df_protonRec['Ppx'], df_protonRec['Ppy'], df_protonRec['Ppz']]
        df_protonRec.loc[:, 'Pe'] = getEnergy(pro, M)

        df_gg = pd.merge(df_gammaRec, df_gammaRec,
                         how='outer', on='event', suffixes=("", "2"))

        df_gg = df_gg[df_gg["GIndex"] < df_gg["GIndex2"]]

        df_ep = pd.merge(df_electronRec, df_protonRec, how='outer', on='event')

        df_epgg = pd.merge(df_ep, df_gg, how='outer', on='event')
        df_epgg = df_epgg[~np.isnan(df_epgg["Ppx"])]
        df_epgg = df_epgg[~np.isnan(df_epgg["Gpx"])]
        df_epgg = df_epgg[~np.isnan(df_epgg["Gpx2"])]

        self.df_epgg = df_epgg #temporarily save df_epgg

    def saveDVpi0vars(self):
        #set up pi0 variables
        df_epgg = self.df_epgg

        # useful objects
        ele = [df_epgg['Epx'], df_epgg['Epy'], df_epgg['Epz']]
        df_epgg.loc[:, 'Ep'] = mag(ele)
        df_epgg.loc[:, 'Ee'] = getEnergy(ele, me)
        df_epgg.loc[:, 'Etheta'] = getTheta(ele)
        df_epgg.loc[:, 'Ephi'] = getPhi(ele)

        pro = [df_epgg['Ppx'], df_epgg['Ppy'], df_epgg['Ppz']]

        gam = [df_epgg['Gpx'], df_epgg['Gpy'], df_epgg['Gpz']]
        df_epgg.loc[:, 'Gp'] = mag(gam)
        df_epgg.loc[:, 'Ge'] = getEnergy(gam, 0)
        df_epgg.loc[:, 'Gtheta'] = getTheta(gam)
        df_epgg.loc[:, 'Gphi'] = getPhi(gam)

        gam2 = [df_epgg['Gpx2'], df_epgg['Gpy2'], df_epgg['Gpz2']]
        df_epgg.loc[:, 'Gp2'] = mag(gam2)
        df_epgg.loc[:,'Ge2'] = getEnergy(gam2, 0)
        df_epgg.loc[:, 'Gtheta2'] = getTheta(gam2)
        df_epgg.loc[:, 'Gphi2'] = getPhi(gam2)

        pi0 = vecAdd(gam, gam2)
        VGS = [-df_epgg['Epx'], -df_epgg['Epy'], pbeam - df_epgg['Epz']]
        v3l = cross(beam, ele)
        v3h = cross(pro, VGS)
        v3g = cross(VGS, gam)
        v3pi0 = cross(VGS, pi0)

        VmissPi0 = [-df_epgg["Epx"] - df_epgg["Ppx"], -df_epgg["Epy"] -
                    df_epgg["Ppy"], pbeam - df_epgg["Epz"] - df_epgg["Ppz"]]
        VmissP = [-df_epgg["Epx"] - df_epgg["Gpx"] - df_epgg["Gpx2"], -df_epgg["Epy"] -
                    df_epgg["Gpy"] - df_epgg["Gpy2"], pbeam - df_epgg["Epz"] - df_epgg["Gpz"] - df_epgg["Gpz2"]]
        Vmiss = [-df_epgg["Epx"] - df_epgg["Ppx"] - df_epgg["Gpx"] - df_epgg["Gpx2"],
                    -df_epgg["Epy"] - df_epgg["Ppy"] - df_epgg["Gpy"] - df_epgg["Gpy2"],
                    pbeam - df_epgg["Epz"] - df_epgg["Ppz"] - df_epgg["Gpz"] - df_epgg["Gpz2"]]
        costheta = cosTheta(VGS, gam)

        df_epgg.loc[:, 'Mpx'], df_epgg.loc[:, 'Mpy'], df_epgg.loc[:, 'Mpz'] = Vmiss

        # binning kinematics
        df_epgg.loc[:,'Q2'] = -((ebeam - df_epgg['Ee'])**2 - mag2(VGS))
        df_epgg.loc[:,'nu'] = (ebeam - df_epgg['Ee'])
        df_epgg.loc[:,'xB'] = df_epgg['Q2'] / 2.0 / M / df_epgg['nu']
        df_epgg.loc[:,'t1'] = 2 * M * (df_epgg['Pe'] - M)
        df_epgg.loc[:,'t2'] = (M * df_epgg['Q2'] + 2 * M * df_epgg['nu'] * (df_epgg['nu'] - np.sqrt(df_epgg['nu'] * df_epgg['nu'] + df_epgg['Q2']) * costheta))\
        / (M + df_epgg['nu'] - np.sqrt(df_epgg['nu'] * df_epgg['nu'] + df_epgg['Q2']) * costheta)
        df_epgg.loc[:,'W'] = np.sqrt(np.maximum(0, (ebeam + M - df_epgg['Ee'])**2 - mag2(VGS)))
        df_epgg.loc[:,'MPt'] = np.sqrt((df_epgg["Epx"] + df_epgg["Ppx"] + df_epgg["Gpx"] + df_epgg["Gpx2"])**2 +
                                 (df_epgg["Epy"] + df_epgg["Ppy"] + df_epgg["Gpy"] + df_epgg["Gpy2"])**2)
        # trento angles
        df_epgg.loc[:,'phi1'] = angle(v3l, v3h)
        df_epgg.loc[:,'phi1'] = np.where(dot(v3l, pro) > 0, 360.0 -
                                  df_epgg['phi1'], df_epgg['phi1'])
        df_epgg.loc[:,'phi2'] = angle(v3l, v3g)
        df_epgg.loc[:,'phi2'] = np.where(dot(v3l, gam) <
                                  0, 360.0 - df_epgg['phi2'], df_epgg['phi2'])

        # exclusivity variables
        df_epgg.loc[:,'MM2_ep'] = (-M - ebeam + df_epgg["Ee"] +
                             df_epgg["Pe"])**2 - mag2(VmissPi0)
        df_epgg.loc[:,'MM2_egg'] = (-M - ebeam + df_epgg["Ee"] +
                             df_epgg["Ge"] + df_epgg["Ge2"])**2 - mag2(VmissP)
        df_epgg.loc[:,'MM2_epgg'] = (-M - ebeam + df_epgg["Ee"] + df_epgg["Pe"] +
                             df_epgg["Ge"] + df_epgg["Ge2"])**2 - mag2(Vmiss)
        df_epgg.loc[:,'ME_epgg'] = (M + ebeam - df_epgg["Ee"] - df_epgg["Pe"] - df_epgg["Ge"] - df_epgg["Ge2"])
        df_epgg.loc[:,'Mpi0'] = pi0InvMass(gam, gam2)
        df_epgg.loc[:,'reconPi'] = angle(VmissPi0, pi0)
        df_epgg.loc[:,"Pie"] = df_epgg['Ge'] + df_epgg['Ge2']
        df_epgg.loc[:,'coplanarity'] = angle(v3h, v3pi0)
        df_epgg.loc[:,'coneAngle1'] = angle(ele, gam)
        df_epgg.loc[:,'coneAngle2'] = angle(ele, gam2)

        df_epgg.loc[:, "closeness"] = np.abs(df_epgg.loc[:, "Mpi0"] - .1349766)

        self.df_epgg = df_epgg
        
        df_Rec = self.df_epgg
        df_MC = self.df_MC
        df = pd.merge(df_Rec, df_MC, how = 'inner', on='event')
        self.df = df

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get args",formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-f","--fname", help="a single root file to convert into pickles", default="/Users/sangbaek/Dropbox (MIT)/data/project/merged_9628_files.root")
    parser.add_argument("-o","--out", help="a single pickle file name as an output", default="goodbyeRoot.pkl")
    parser.add_argument("-S","--entry_start", help="entry_start to start reading the root file", default = None)
    parser.add_argument("-s","--entry_stop", help="entry_stop to stop reading the root file", default = None)
    parser.add_argument("-p","--polarity", help="polarity", default = "inbending")
    parser.add_argument("-g","--generator", help="choose dvcs or pi0", default = "pi0norad")
    parser.add_argument("-r","--raw", help="save raw only", default = False, action = "store_true")
    parser.add_argument("-d","--detRes", help="include detector response", action = "store_true")
    parser.add_argument("-D","--dvcs", help="save dvcs overlap", action = "store_true")
    
    args = parser.parse_args()

    if args.entry_start:
        args.entry_start = int(args.entry_start)
    if args.entry_stop:
        args.entry_stop = int(args.entry_stop)

    converter = root2pickle(args.fname, entry_start = args.entry_start, entry_stop = args.entry_stop, pol = args.polarity, gen = args.generator, raw = args.raw, detRes = args.detRes, dvcs = args.dvcs)
    df = converter.df

    df.to_pickle(args.out)