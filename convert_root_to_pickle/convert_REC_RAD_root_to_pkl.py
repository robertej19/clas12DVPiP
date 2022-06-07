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

        if detRes:
            eleKeysRec.extend(["Evx", "Evy", "Evz"])
            eleKeysRec.extend(["EDc1Hitx", "EDc1Hity", "EDc1Hitz", "EDc3Hitx", "EDc3Hity", "EDc3Hitz"])
            eleKeysRec.extend(["Eedep", "Eedep1", "Eedep2", "Eedep3"])
            gamKeysRec.extend(["Gedep", "Gedep1", "Gedep2", "Gedep3"])
            proKeysRec.extend(["Pvz"])
            proKeysRec.extend(["PCvt1Hitx", "PCvt1Hity", "PCvt1Hitz", "PCvt3Hitx", "PCvt3Hity", "PCvt3Hitz", "PCvt5Hitx", "PCvt5Hity", "PCvt5Hitz", "PCvt7Hitx", "PCvt7Hity", "PCvt7Hitz", "PCvt12Hitx", "PCvt12Hity", "PCvt12Hitz"])
            proKeysRec.extend(["PDc1Hitx", "PDc1Hity", "PDc1Hitz", "PDc3Hitx", "PDc3Hity", "PDc3Hitz"])
            eleKeysRec.extend(["startTime"])
            proKeysRec.extend(["PFtof1aTime", "PFtof1bTime", "PFtof2Time", "PCtofTime"])
            proKeysRec.extend(["Pchi2pid", "Pchi2track", "PNDFtrack"])

        # read them
        for key in eleKeysRec:
            df_electronRec[key] = self.tree[key].array(library="pd", entry_start = entry_start, entry_stop=entry_stop)
        for key in proKeysRec:
            df_protonRec[key] = self.tree[key].array(library="pd", entry_start = entry_start, entry_stop=entry_stop)
        for key in gamKeysRec:
            df_gammaRec[key] = self.tree[key].array(library="pd", entry_start = entry_start, entry_stop=entry_stop)

        self.closeFile()

        #convert data type to standard double
        # df_electronRec = df_electronRec.astype({"Epx": float, "Epy": float, "Epz": float, "Evx": float, "Evy": float, "Evz": float})
        # df_protonRec = df_protonRec.astype({"Ppx": float, "Ppy": float, "Ppz": float, "Pvz": float})
        df_electronRec = df_electronRec.astype({"Epx": float, "Epy": float, "Epz": float})
        df_protonRec = df_protonRec.astype({"Ppx": float, "Ppy": float, "Ppz": float})
        # df_protonRec = df_protonRec.astype({"PDc1Hitx": float, "PDc1Hity": float, "PDc1Hitz": float})
        # df_protonRec = df_protonRec.astype({"PDc3Hitx": float, "PDc3Hity": float, "PDc3Hitz": float})
        # df_protonRec = df_protonRec.astype({"PCvt1Hitx": float, "PCvt1Hity": float, "PCvt1Hitz": float})
        # df_protonRec = df_protonRec.astype({"PCvt3Hitx": float, "PCvt3Hity": float, "PCvt3Hitz": float})
        # df_protonRec = df_protonRec.astype({"PCvt5Hitx": float, "PCvt5Hity": float, "PCvt5Hitz": float})
        # df_protonRec = df_protonRec.astype({"PCvt7Hitx": float, "PCvt7Hity": float, "PCvt7Hitz": float})
        # df_protonRec = df_protonRec.astype({"PCvt12Hitx": float, "PCvt12Hity": float, "PCvt12Hitz": float})
        df_gammaRec = df_gammaRec.astype({"Gpx": float, "Gpy": float, "Gpz": float, "GcX": float, "GcY": float})

        #photon fiducial cuts by F.X. Girod
        df_gammaRec.loc[:, "GFidFX"] = 0

        sector_cond = [df_gammaRec.Gsector ==1, df_gammaRec.Gsector ==2, df_gammaRec.Gsector ==3, df_gammaRec.Gsector ==4, df_gammaRec.Gsector ==5, df_gammaRec.Gsector ==6]
        psplit = np.select(sector_cond, [87, 82, 85, 77, 78, 82])
        tleft = np.select(sector_cond, [58.7356, 62.8204, 62.2296, 53.7756, 58.2888, 54.5822])
        tright = np.select(sector_cond, [58.7477, 51.2589, 59.2357, 56.2415, 60.8219, 49.8914])
        sleft = np.select(sector_cond, [0.582053, 0.544976, 0.549788, 0.56899, 0.56414, 0.57343])
        sright = np.select(sector_cond, [-0.591876, -0.562926, -0.562246, -0.563726, -0.568902, -0.550729])
        rleft = np.select(sector_cond, [64.9348, 64.7541, 67.832, 55.9324, 55.9225, 60.0997])
        rright = np.select(sector_cond, [65.424, 54.6992, 63.6628, 57.8931, 56.5367, 56.4641])
        qleft = np.select(sector_cond, [0.745578, 0.606081, 0.729202, 0.627239, 0.503674, 0.717899])
        qright = np.select(sector_cond, [-0.775022, -0.633863, -0.678901, -0.612458, -0.455319, -0.692481])
        #first condition
        ang = np.radians((df_gammaRec.loc[df_gammaRec.Gsector<7, "Gsector"]-1) * 60)
        GcX_rot = df_gammaRec.loc[df_gammaRec.Gsector<7, "GcY"] * np.sin(ang) + df_gammaRec.loc[df_gammaRec.Gsector<7, "GcX"] * np.cos(ang)
        GcY_rot = df_gammaRec.loc[df_gammaRec.Gsector<7, "GcY"] * np.cos(ang) - df_gammaRec.loc[df_gammaRec.Gsector<7, "GcX"] * np.sin(ang)

        df_gammaRec.loc[df_gammaRec.Gsector<7, "GcX"] = GcX_rot
        df_gammaRec.loc[df_gammaRec.Gsector<7, "GcY"] = GcY_rot

        cond1_1 = df_gammaRec.GcX >= psplit
        cond1_2 = df_gammaRec.GcY < sleft * (df_gammaRec.GcX - tleft)
        cond1_3 = df_gammaRec.GcY > sright * (df_gammaRec.GcX - tright)
        cond1_4 = df_gammaRec.Gsector < 7
        cond1 = cond1_1 & cond1_2 & cond1_3 & cond1_4
        df_gammaRec.loc[cond1, "GFidFX"] = 1
        #second condition else if the first
        # cond2_0 = df_gammaRec.GFidFX == 0 # not necessary, because cond2_1 rules out the first (S. Lee)
        cond2_1 = df_gammaRec.GcX < psplit
        cond2_2 = df_gammaRec.GcY < qleft * (df_gammaRec.GcX - rleft)
        cond2_3 = df_gammaRec.GcY > qright * (df_gammaRec.GcX - rright)
        cond2_4 = df_gammaRec.Gsector < 7
        cond2 = cond2_1 & cond2_2 & cond2_3 & cond2_4
        df_gammaRec.loc[cond2, "GFidFX"] = 1

        df_gammaRec.loc[df_gammaRec.Gsector > 7, "GFidFX"] = 1

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

        df_protonRec.loc[:, "PDc1theta"] = -100000

        if detRes:
            df_protonRec.loc[:, "PDc3theta"] = -100000

            df_electronRec.loc[:, "EDc1theta"] = getTheta([df_electronRec.EDc1Hitx, df_electronRec.EDc1Hity, df_electronRec.EDc1Hitz])
            df_electronRec.loc[:, "EDc3theta"] = getTheta([df_electronRec.EDc3Hitx, df_electronRec.EDc3Hity, df_electronRec.EDc3Hitz])
            df_electronRec.loc[:, "EAngleDiff"] = df_electronRec.loc[:, "EDc3theta"] - df_electronRec.loc[:, "EDc1theta"]

            df_protonRec.loc[:, "PCvt1r"] = -100000
            df_protonRec.loc[:, "PCvt1theta"] = -100000
            df_protonRec.loc[:, "PCvt1phi"] = -100000
            df_protonRec.loc[:, "PCvt3r"] = -100000
            df_protonRec.loc[:, "PCvt3theta"] = -100000
            df_protonRec.loc[:, "PCvt3phi"] = -100000
            df_protonRec.loc[:, "PCvt5r"] = -100000
            df_protonRec.loc[:, "PCvt5theta"] = -100000
            df_protonRec.loc[:, "PCvt5phi"] = -100000
            df_protonRec.loc[:, "PCvt7r"] = -100000
            df_protonRec.loc[:, "PCvt7theta"] = -100000
            df_protonRec.loc[:, "PCvt7phi"] = -100000
            df_protonRec.loc[:, "PCvt12r"] = -100000
            df_protonRec.loc[:, "PCvt12theta"] = -100000
            df_protonRec.loc[:, "PCvt12phi"] = -100000

        df_protonRecFD = df_protonRec.loc[df_protonRec.Psector<7, :]
        df_protonRecCD = df_protonRec.loc[(df_protonRec.Psector>7) & (df_protonRec.Ptheta<75), :]
        df_protonRecOthers = df_protonRec.loc[(df_protonRec.Psector>7) & (df_protonRec.Ptheta>=75), :]

        correction = False

        def corr(x, t):
            x0, x1, x2, x3 = x
            return x0 + x1*np.power(t-np.ones(len(t))*0.3, x3)

        df_protonRecFD = df_protonRecFD.loc[df_protonRec.Pp > 0.3, :]
        df_protonRecFD.loc[:, "PDc1theta"] = getTheta([df_protonRecFD.PDc1Hitx, df_protonRecFD.PDc1Hity, df_protonRecFD.PDc1Hitz])
        if detRes:
            df_protonRecFD.loc[:, "PDc3theta"] = getTheta([df_protonRecFD.PDc3Hitx, df_protonRecFD.PDc3Hity, df_protonRecFD.PDc3Hitz])
        best_params = [-53.14680163254601, 79.61307254040804, 0.3, 0.05739232362022314]
        df_protonRecFD_1 = df_protonRecFD.loc[df_protonRecFD.PDc1theta < corr(best_params, df_protonRecFD.Pp), :]
        df_protonRecFD_2 = df_protonRecFD.loc[df_protonRecFD.PDc1theta >= corr(best_params, df_protonRecFD.Pp), :]

        if detRes:
            df_protonRecCD.loc[:, "PCvt1r"] = mag([df_protonRecCD.PCvt1Hitx, df_protonRecCD.PCvt1Hity, df_protonRecCD.PCvt1Hitz])
            df_protonRecCD.loc[:, "PCvt1theta"] = getTheta([df_protonRecCD.PCvt1Hitx, df_protonRecCD.PCvt1Hity, df_protonRecCD.PCvt1Hitz])
            df_protonRecCD.loc[:, "PCvt1phi"] = getPhi([df_protonRecCD.PCvt1Hitx, df_protonRecCD.PCvt1Hity, df_protonRecCD.PCvt1Hitz])
            df_protonRecCD.loc[:, "PCvt3r"] = mag([df_protonRecCD.PCvt3Hitx, df_protonRecCD.PCvt3Hity, df_protonRecCD.PCvt3Hitz])
            df_protonRecCD.loc[:, "PCvt3theta"] = getTheta([df_protonRecCD.PCvt3Hitx, df_protonRecCD.PCvt3Hity, df_protonRecCD.PCvt3Hitz])
            df_protonRecCD.loc[:, "PCvt3phi"] = getPhi([df_protonRecCD.PCvt3Hitx, df_protonRecCD.PCvt3Hity, df_protonRecCD.PCvt3Hitz])
            df_protonRecCD.loc[:, "PCvt5r"] = mag([df_protonRecCD.PCvt5Hitx, df_protonRecCD.PCvt5Hity, df_protonRecCD.PCvt5Hitz])
            df_protonRecCD.loc[:, "PCvt5theta"] = getTheta([df_protonRecCD.PCvt5Hitx, df_protonRecCD.PCvt5Hity, df_protonRecCD.PCvt5Hitz])
            df_protonRecCD.loc[:, "PCvt5phi"] = getPhi([df_protonRecCD.PCvt5Hitx, df_protonRecCD.PCvt5Hity, df_protonRecCD.PCvt5Hitz])
            df_protonRecCD.loc[:, "PCvt7r"] = mag([df_protonRecCD.PCvt7Hitx, df_protonRecCD.PCvt7Hity, df_protonRecCD.PCvt7Hitz])
            df_protonRecCD.loc[:, "PCvt7theta"] = getTheta([df_protonRecCD.PCvt7Hitx, df_protonRecCD.PCvt7Hity, df_protonRecCD.PCvt7Hitz])
            df_protonRecCD.loc[:, "PCvt7phi"] = getPhi([df_protonRecCD.PCvt7Hitx, df_protonRecCD.PCvt7Hity, df_protonRecCD.PCvt7Hitz])
            df_protonRecCD.loc[:, "PCvt12r"] = mag([df_protonRecCD.PCvt12Hitx, df_protonRecCD.PCvt12Hity, df_protonRecCD.PCvt12Hitz])
            df_protonRecCD.loc[:, "PCvt12theta"] = getTheta([df_protonRecCD.PCvt12Hitx, df_protonRecCD.PCvt12Hity, df_protonRecCD.PCvt12Hitz])
            df_protonRecCD.loc[:, "PCvt12phi"] = getPhi([df_protonRecCD.PCvt12Hitx, df_protonRecCD.PCvt12Hity, df_protonRecCD.PCvt12Hitz])

        #inbending
        if pol == "inbending":
            const_FD = -0.00051894 - 0.00018104 * df_protonRecFD_1.Ptheta
            coeff_FD = 3.29466917*10**(-3) +  5.73663160*10**(-4) * df_protonRecFD_1.Ptheta - 1.40807209 * 10**(-5) * df_protonRecFD_1.Ptheta * df_protonRecFD_1.Ptheta
            CorrectedPp_FD_1 = np.select([df_protonRecFD_1.Pp<1, df_protonRecFD_1.Pp>=1], [const_FD + coeff_FD/df_protonRecFD_1.loc[:, "Pp"] + df_protonRecFD_1.loc[:, "Pp"], np.exp(-2.739 - 3.932*df_protonRecFD_1.Pp) + 0.002907+df_protonRecFD_1.Pp])

            const_FD = -0.16742969 + 0.00697925 * df_protonRecFD_1.Ptheta
            coeff_FD = 0.23352115 - 0.01338697 * df_protonRecFD_1.Ptheta
            CorrectedPtheta_FD_1 = const_FD + coeff_FD/df_protonRecFD_1.loc[:, "Pp"]/df_protonRecFD_1.loc[:, "Pp"] + df_protonRecFD_1.loc[:, "Ptheta"]

            const_FD = 0.21192125 -0.0115175 * df_protonRecFD_1.Ptheta
            coeff_FD = -8.94307411*0.1 + 1.66349766*0.1 * df_protonRecFD_1.Ptheta -8.90617559*0.001 * df_protonRecFD_1.Ptheta * df_protonRecFD_1.Ptheta + 1.64803754*0.0001 * df_protonRecFD_1.Ptheta * df_protonRecFD_1.Ptheta * df_protonRecFD_1.Ptheta
            CorrectedPphi_FD_1 = const_FD + coeff_FD/df_protonRecFD_1.loc[:, "Pp"]/df_protonRecFD_1.loc[:, "Pp"] + df_protonRecFD_1.loc[:, "Pphi"]

            const_FD = -3.03346359*10**(-1) + 1.83368163*10**(-2)*df_protonRecFD_2.Ptheta - 2.86486404*10**(-4)*df_protonRecFD_2.Ptheta*df_protonRecFD_2.Ptheta
            coeff_FD =  2.01023276*10**(-1) - 1.13312215*10**(-2)*df_protonRecFD_2.Ptheta + 1.82487916*10**(-4)*df_protonRecFD_2.Ptheta*df_protonRecFD_2.Ptheta
            CorrectedPp_FD_2 = np.select([df_protonRecFD_2.Pp<1, df_protonRecFD_2.Pp>=1], [const_FD + coeff_FD/df_protonRecFD_2.loc[:, "Pp"] + df_protonRecFD_2.loc[:, "Pp"], np.exp(-1.2 - 4.228*df_protonRecFD_2.Pp) + 0.007502+df_protonRecFD_2.Pp])

            const_FD = 2.04334532 * 10 -1.81052405 * df_protonRecFD_2.Ptheta + 5.32556360*0.01 * df_protonRecFD_2.Ptheta * df_protonRecFD_2.Ptheta -5.23157558*0.0001 * df_protonRecFD_2.Ptheta * df_protonRecFD_2.Ptheta * df_protonRecFD_2.Ptheta
            coeff_FD = 8.74233279 -7.63869344 * 0.1 * df_protonRecFD_2.Ptheta + 2.22376362*0.01 * df_protonRecFD_2.Ptheta * df_protonRecFD_2.Ptheta -2.16457260*0.0001 * df_protonRecFD_2.Ptheta * df_protonRecFD_2.Ptheta * df_protonRecFD_2.Ptheta
            CorrectedPtheta_FD_2 = const_FD + coeff_FD/df_protonRecFD_2.loc[:, "Pp"]/df_protonRecFD_2.loc[:, "Pp"] + df_protonRecFD_2.loc[:, "Ptheta"]

            const_FD = 0.54697831 -0.04896981*df_protonRecFD_2.Ptheta +  0.00111376*df_protonRecFD_2.Ptheta*df_protonRecFD_2.Ptheta
            coeff_FD = -4.06733541*10**2 + 2.43696202*10*df_protonRecFD_2.Ptheta -3.36144736*10**(-1)*df_protonRecFD_2.Ptheta*df_protonRecFD_2.Ptheta
            coeff2_FD = 2.06378660*10 - 1.42866062*df_protonRecFD_2.Ptheta + 2.01085440*10**(-2)*df_protonRecFD_2.Ptheta*df_protonRecFD_2.Ptheta
            CorrectedPphi_FD_2 = const_FD + coeff_FD*np.exp(coeff2_FD*df_protonRecFD_2.loc[:, "Pp"]) + df_protonRecFD_2.loc[:, "Pphi"]

            #CD part
            const_CD = 1.93686914 - 0.116288824*df_protonRecCD.Ptheta + 0.00223685833*df_protonRecCD.Ptheta**2 - 1.40771969 * 10**(-5)*df_protonRecCD.Ptheta**3
            coeff_CD = -0.738047800 + 0.0443343685*df_protonRecCD.Ptheta - 8.50985972*10**(-4)*df_protonRecCD.Ptheta*df_protonRecCD.Ptheta + 5.36810280 * 10**(-6) * df_protonRecCD.Ptheta**3

            CorrectedPp_CD = const_CD + coeff_CD/df_protonRecCD.loc[:, "Pp"] + df_protonRecCD.loc[:, "Pp"]

            const_CD = -1.09849291*100 + 8.86664014 * df_protonRecCD.Ptheta - 0.26643881 * df_protonRecCD.Ptheta**2 + 3.53814210 * 10**(-3) * df_protonRecCD.Ptheta**3 - 1.75297107 * 10**(-5) * df_protonRecCD.Ptheta**4
            coeff_CD = 9.52034523*100 -5.74808292 * 10 * df_protonRecCD.Ptheta + 1.15386949 * df_protonRecCD.Ptheta**2 - 7.57970373 * 0.001 * df_protonRecCD.Ptheta**3
            coeff2_CD = -2.00387313*100 + 1.18979079 * 10 * df_protonRecCD.Ptheta - 2.37730217*0.1 * df_protonRecCD.Ptheta**2 + 1.55153003*0.001*df_protonRecCD.Ptheta**3

            CorrectedPtheta_CD = const_CD + coeff_CD*np.exp(coeff2_CD*df_protonRecCD.loc[:, "Pp"]) + df_protonRecCD.loc[:, "Ptheta"]

            const_CD = 4.94546178 -3.26662886*0.1 * df_protonRecCD.Ptheta +  7.39069603 * 0.001 * df_protonRecCD.Ptheta**2 -6.83599356*10**(-5) * df_protonRecCD.Ptheta**3 + 2.12303103*10**(-7) * df_protonRecCD.Ptheta**4
            coeff_CD = 1.72181613*10**(5) -1.36827111*10**(4) * df_protonRecCD.Ptheta + 4.00923146*10**(2) * df_protonRecCD.Ptheta**2 - 5.12792347 * df_protonRecCD.Ptheta**3 + 2.41793167*10**(-2) * df_protonRecCD.Ptheta**4
            coeff2_CD =  1.20477219*10**(2) -5.86630228 * df_protonRecCD.Ptheta + 7.44007875*10**(-2) * df_protonRecCD.Ptheta**2 -2.42652473*10**(-4) * df_protonRecCD.Ptheta**3
            CorrectedPphi_CD = const_CD + coeff_CD*np.exp(coeff2_CD*df_protonRecCD.loc[:, "Pp"]) + df_protonRecCD.loc[:, "Pphi"]

            correction = True
        elif pol == "outbending":
            #FD part
            const_FD = 0.05083242 -0.00469777*df_protonRecFD_1.Ptheta + 0.0001082*df_protonRecFD_1.Ptheta*df_protonRecFD_1.Ptheta
            coeff_FD = -1.47443264*0.01 + 1.58220893*0.001*df_protonRecFD_1.Ptheta -3.19490013*0.00001*df_protonRecFD_1.Ptheta*df_protonRecFD_1.Ptheta
            CorrectedPp_FD_1 = np.select([df_protonRecFD_1.Pp<1, df_protonRecFD_1.Pp>=1], [const_FD + coeff_FD/df_protonRecFD_1.loc[:, "Pp"] + df_protonRecFD_1.loc[:, "Pp"], np.exp(-2.739 - 3.932*df_protonRecFD_1.Pp) + 0.002907 + df_protonRecFD_1.Pp])

            const_FD = -2.56460305*10 + 3.29877542*df_protonRecFD_1.Ptheta -1.43106886*0.1*df_protonRecFD_1.Ptheta*df_protonRecFD_1.Ptheta + 2.08341898*0.001*df_protonRecFD_1.Ptheta*df_protonRecFD_1.Ptheta*df_protonRecFD_1.Ptheta
            coeff_FD =  9.12532740*10 -1.20100762*10*df_protonRecFD_1.Ptheta + 5.27654711*0.1*df_protonRecFD_1.Ptheta*df_protonRecFD_1.Ptheta -7.72656759*0.001*df_protonRecFD_1.Ptheta*df_protonRecFD_1.Ptheta*df_protonRecFD_1.Ptheta
            CorrectedPtheta_FD_1 = const_FD + coeff_FD/df_protonRecFD_1.loc[:, "Pp"]/df_protonRecFD_1.loc[:, "Pp"] + df_protonRecFD_1.loc[:, "Ptheta"]

            const_FD = -20.4780893 + 1.67020488*df_protonRecFD_1.Ptheta - 0.03419348*df_protonRecFD_1.Ptheta*df_protonRecFD_1.Ptheta
            coeff_FD = 35.02807194 - 2.9098043*df_protonRecFD_1.Ptheta +  0.06037906*df_protonRecFD_1.Ptheta*df_protonRecFD_1.Ptheta
            CorrectedPphi_FD_1 = const_FD + coeff_FD/df_protonRecFD_1.loc[:, "Pp"]/df_protonRecFD_1.loc[:, "Pp"] + df_protonRecFD_1.loc[:, "Pphi"]

            const_FD = 0.09832589 -0.0066463*df_protonRecFD_2.Ptheta + 0.00010312*df_protonRecFD_2.Ptheta*df_protonRecFD_2.Ptheta
            coeff_FD = -9.61421691*0.01 + 6.85638807*0.001*df_protonRecFD_2.Ptheta -9.75766427*0.00001*df_protonRecFD_2.Ptheta*df_protonRecFD_2.Ptheta
            CorrectedPp_FD_2 = np.select([df_protonRecFD_2.Pp<1, df_protonRecFD_2.Pp>=1], [const_FD + coeff_FD/df_protonRecFD_2.loc[:, "Pp"] + df_protonRecFD_2.loc[:, "Pp"], np.exp(-1.871 - 3.063*df_protonRecFD_2.Pp) + 0.007517 + df_protonRecFD_2.Pp])

            const_FD = -1.68873940 + 9.56867163*0.01*df_protonRecFD_2.Ptheta -1.43741464*0.001*df_protonRecFD_2.Ptheta*df_protonRecFD_2.Ptheta
            coeff_FD = 1.49978357*10 -1.40137094*df_protonRecFD_2.Ptheta + 4.38501543*0.01*df_protonRecFD_2.Ptheta*df_protonRecFD_2.Ptheta -4.57982872*0.0001*df_protonRecFD_2.Ptheta*df_protonRecFD_2.Ptheta*df_protonRecFD_2.Ptheta
            CorrectedPtheta_FD_2 = const_FD + coeff_FD/df_protonRecFD_2.loc[:, "Pp"]/df_protonRecFD_2.loc[:, "Pp"] + df_protonRecFD_2.loc[:, "Ptheta"]

            const_FD = 6.75359137 - 0.43199851*df_protonRecFD_2.Ptheta + 0.0068995*df_protonRecFD_2.Ptheta*df_protonRecFD_2.Ptheta
            coeff_FD = -1.68588219 + 1.05609627*0.1*df_protonRecFD_2.Ptheta -1.50452832*0.001*df_protonRecFD_2.Ptheta*df_protonRecFD_2.Ptheta
            CorrectedPphi_FD_2 = const_FD + coeff_FD/df_protonRecFD_2.loc[:, "Pp"]/df_protonRecFD_2.loc[:, "Pp"] + df_protonRecFD_2.loc[:, "Pphi"]
            #CD part
            const_CD = 1.92657376 - 0.113836734*df_protonRecCD.Ptheta + 0.00215038526*df_protonRecCD.Ptheta**2 - 1.32525053 * 10**(-5)*df_protonRecCD.Ptheta**3
            coeff_CD = -0.755650043 + 0.0445538936*df_protonRecCD.Ptheta - 8.38241864*10**(-4)*df_protonRecCD.Ptheta*df_protonRecCD.Ptheta + 5.16887255 * 10**(-6) * df_protonRecCD.Ptheta**3

            CorrectedPp_CD = const_CD + coeff_CD/df_protonRecCD.loc[:, "Pp"] + df_protonRecCD.loc[:, "Pp"]

            const_CD = -5.79024055*10 + 4.67197531 * df_protonRecCD.Ptheta - 0.140156897 * df_protonRecCD.Ptheta**2 + 1.85853057 * 10**(-3) * df_protonRecCD.Ptheta**3 - 9.19989908 * 10**(-6) * df_protonRecCD.Ptheta**4
            coeff_CD = 2.99700765*1000 - 2.18027982 * 10**2 * df_protonRecCD.Ptheta + 5.84757503 * df_protonRecCD.Ptheta**2 - 6.80409195 * 0.01 * df_protonRecCD.Ptheta**3 + 2.89244618 * 0.0001 * df_protonRecCD.Ptheta**4
            coeff2_CD = -1.82237904*100 + 1.10153549 * 10 * df_protonRecCD.Ptheta - 2.24699931*0.1 * df_protonRecCD.Ptheta**2 + 1.49390960*0.001*df_protonRecCD.Ptheta**3

            CorrectedPtheta_CD = const_CD + coeff_CD*np.exp(coeff2_CD*df_protonRecCD.loc[:, "Pp"]) + df_protonRecCD.loc[:, "Ptheta"]

            const_CD = 7.58761670 - 5.28224578*0.1 * df_protonRecCD.Ptheta +  1.31580117 * 0.01 * df_protonRecCD.Ptheta**2 -1.41738951*10**(-4) * df_protonRecCD.Ptheta**3 + 5.62884363*10**(-7) * df_protonRecCD.Ptheta**4
            coeff_CD = 1.07644097*10**(5) - 8.67994639*10**(3) * df_protonRecCD.Ptheta + 2.57187193*10**(2) * df_protonRecCD.Ptheta**2 - 3.31379317 * df_protonRecCD.Ptheta**3 + 1.56896621*10**(-2) * df_protonRecCD.Ptheta**4
            coeff2_CD =  1.92263184*10**(2) -1.00870704 * 10 * df_protonRecCD.Ptheta + 1.56575252*10**(-1) * df_protonRecCD.Ptheta**2 -7.71489734*10**(-4) * df_protonRecCD.Ptheta**3
            CorrectedPphi_CD = const_CD + coeff_CD*np.exp(coeff2_CD*df_protonRecCD.loc[:, "Pp"]) + df_protonRecCD.loc[:, "Pphi"]

            correction = True
        else:
            print("no correction applied")

        if correction:
            print("correction applied for " + pol)

            df_protonRecCD.loc[:, "Pp"] = CorrectedPp_CD
            df_protonRecCD.loc[:, "Ptheta"] = CorrectedPtheta_CD
            df_protonRecCD.loc[:, "Pphi"] = CorrectedPphi_CD

            df_protonRecFD_1.loc[:, "Pp"] = CorrectedPp_FD_1
            df_protonRecFD_1.loc[:, "Ptheta"] = CorrectedPtheta_FD_1
            df_protonRecFD_1.loc[:, "Pphi"] = CorrectedPphi_FD_1
            df_protonRecFD_1.loc[:, "Pband"] = "lower"

            df_protonRecFD_2.loc[:, "Pp"] = CorrectedPp_FD_2
            df_protonRecFD_2.loc[:, "Ptheta"] = CorrectedPtheta_FD_2
            df_protonRecFD_2.loc[:, "Pphi"] = CorrectedPphi_FD_2
            df_protonRecFD_2.loc[:, "Pband"] = "upper"

            df_protonRecFD = pd.concat([df_protonRecFD_1, df_protonRecFD_2])

            df_protonRec = pd.concat([df_protonRecFD, df_protonRecCD, df_protonRecOthers])

        if detRes:
            df_protonRec.loc[:, "PAngleDiff"] = df_protonRec.loc[:, "PDc3theta"] - df_protonRec.loc[:, "PDc1theta"]

        # Turn off smearing, for now
        # # # #smearing photon
        # # # gam = [df_gammaRec['Gpx'], df_gammaRec['Gpy'], df_gammaRec['Gpz']]
        # # # df_gammaRec.loc[:, 'Gp'] = mag(gam)
        # # # df_gammaRec.loc[:, 'Gtheta'] = getTheta(gam)
        # # # df_gammaRec.loc[:, 'Gphi'] = getPhi(gam)
        # # # #FT photon
        # # # df_gammaRec.loc[df_gammaRec["Gsector"]>7, "Gp"] = df_gammaRec.loc[df_gammaRec["Gsector"]>7, "Gp"]*np.random.normal(1, 0.014, len(df_gammaRec.loc[df_gammaRec.Gsector>7]))
        # # # #FD photon
        # # # df_gammaRec.loc[df_gammaRec["Gsector"]<7, "Gp"] = df_gammaRec.loc[df_gammaRec["Gsector"]<7, "Gp"]*np.random.normal(1, 0.035, len(df_gammaRec.loc[df_gammaRec.Gsector<7]))

        # # # df_gammaRec.loc[:, "Gpx"] = df_gammaRec.loc[:, "Gp"]*np.sin(np.radians(df_gammaRec.loc[:, "Gtheta"]))*np.cos(np.radians(df_gammaRec.loc[:, "Gphi"]))
        # # # df_gammaRec.loc[:, "Gpy"] = df_gammaRec.loc[:, "Gp"]*np.sin(np.radians(df_gammaRec.loc[:, "Gtheta"]))*np.sin(np.radians(df_gammaRec.loc[:, "Gphi"]))
        # # # df_gammaRec.loc[:, "Gpz"] = df_gammaRec.loc[:, "Gp"]*np.cos(np.radians(df_gammaRec.loc[:, "Gtheta"]))

        # # # #smearing proton
        # # # #CD proton
        # # # df_protonRec.loc[df_protonRec["Psector"]>7, "Pp"] = df_protonRec.loc[df_protonRec["Psector"]>7, "Pp"]*np.random.normal(1, np.abs(0.16*(1/(1+np.exp(-(df_protonRec.loc[df_protonRec["Psector"]>7, "Pp"]-0.3)/0.1))-0.5)), len(df_protonRec.loc[df_protonRec.Psector>7]))
        # # # df_protonRec.loc[df_protonRec["Psector"]>7, "Ptheta"] = df_protonRec.loc[df_protonRec["Psector"]>7, "Ptheta"] + np.random.normal(0, 0.8, len(df_protonRec.loc[df_protonRec.Psector>7]))
        # # # df_protonRec.loc[df_protonRec["Psector"]>7, "Pphi"] = df_protonRec.loc[df_protonRec["Psector"]>7, "Pphi"] + np.random.normal(0, 2.2, len(df_protonRec.loc[df_protonRec.Psector>7])) 
        # # # #FD proton
        # # # if pol == "inbending":
        # # #     # df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"] = df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"]*np.random.normal(1, np.select([df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"]<1.3, (df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"]>=1.3) & (df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"]<1.9), df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"]>=1.9], [0.075, -7/120*df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"]+7*1.9/120+0.04, 0.04])*(1/(1+np.exp(-(df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"]-0.5)/0.05))), len(df_protonRec.loc[df_protonRec.Psector<7]))
        # # #     df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"] = df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"]*np.random.normal(1, np.select([df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"]<1.3, (df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"]>=1.3) & (df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"]<1.9), df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"]>=1.9], [0.06, -(0.06-0.04)/(1.9-1.3)*df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"]+(0.06-0.04)/(1.9-1.3)*1.9+0.04, 0.04])*(1/(1+np.exp(-(df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"]-0.5)/0.05))), len(df_protonRec.loc[df_protonRec.Psector<7]))
        # # # elif pol == "outbending":
        # # #     df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"] = df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"]*np.random.normal(1, np.select([df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"]<.95, (df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"]>=.95) & (df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"]<1.2), (df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"]>=1.2)&(df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"]<1.575), (df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"]>=1.575) & (df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"]<1.9), (df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"]>1.9)], [0.08, -0.025/(1.2-.95)*df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"]+1.2*0.025/(1.2-.95) + 0.055, 0.055, -0.015/(1.9-1.575)*df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"]+1.9*0.015/(1.9-1.575) + 0.04,0.04])*(1/(1+np.exp(-(df_protonRec.loc[df_protonRec["Psector"]<7, "Pp"]-0.6)/0.05))), len(df_protonRec.loc[df_protonRec.Psector<7]))





        #moduli proton phi
        df_protonRec.loc[:, "Pphi"] = np.where(df_protonRec.loc[:, "Pphi"]>180, df_protonRec.loc[:, "Pphi"] - 360, df_protonRec.loc[:, "Pphi"]) 
        df_protonRec.loc[:, "Pphi"] = np.where(df_protonRec.loc[:, "Pphi"]<-180, df_protonRec.loc[:, "Pphi"] + 360, df_protonRec.loc[:, "Pphi"]) 

        df_protonRec.loc[:, "Ppx"] = df_protonRec.loc[:, "Pp"]*np.sin(np.radians(df_protonRec.loc[:, "Ptheta"]))*np.cos(np.radians(df_protonRec.loc[:, "Pphi"]))
        df_protonRec.loc[:, "Ppy"] = df_protonRec.loc[:, "Pp"]*np.sin(np.radians(df_protonRec.loc[:, "Ptheta"]))*np.sin(np.radians(df_protonRec.loc[:, "Pphi"]))
        df_protonRec.loc[:, "Ppz"] = df_protonRec.loc[:, "Pp"]*np.cos(np.radians(df_protonRec.loc[:, "Ptheta"]))

        pro = [df_protonRec['Ppx'], df_protonRec['Ppy'], df_protonRec['Ppz']]
        df_protonRec.loc[:, 'Pe'] = getEnergy(pro, M)

        df_gg = pd.merge(df_gammaRec, df_gammaRec,
                         how='outer', on='event', suffixes=("", "2"))
        if detRes:
            df_gg = df_gg.loc[:, ~df_gg.columns.duplicated()]
            df_gg.loc[:, "Gedep2_tot"] = df_gg.Gedep12 + df_gg.Gedep22 + df_gg.Gedep32

        df_gg = df_gg[df_gg["GIndex"] < df_gg["GIndex2"]]

        df_protonRec = df_protonRec.drop(["PDc1Hitx", "PDc1Hity", "PDc1Hitz", "PDc1theta"], axis = 1)
        if detRes:
            df_electronRec = df_electronRec.drop(["EDc1Hitx", "EDc1Hity", "EDc1Hitz", "EDc3Hitx", "EDc3Hity", "EDc3Hitz", "EDc1theta", "EDc3theta"], axis = 1)
            df_protonRec = df_protonRec.drop(["PCvt1Hitx", "PCvt1Hity", "PCvt1Hitz", "PCvt3Hitx", "PCvt3Hity", "PCvt3Hitz", "PCvt5Hitx", "PCvt5Hity", "PCvt5Hitz", "PCvt7Hitx", "PCvt7Hity", "PCvt7Hitz", "PCvt12Hitx", "PCvt12Hity", "PCvt12Hitz"], axis = 1)
            df_protonRec = df_protonRec.drop(["PDc3Hitx", "PDc3Hity", "PDc3Hitz", "PDc3theta"], axis = 1)

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