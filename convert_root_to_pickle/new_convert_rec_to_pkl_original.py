#!/usr/bin/env python3
"""
From https://github.com/Sangbaek/clas12DVCS/blob/main/root2pickleEpggRec.py
"""

import uproot
import pandas as pd
import numpy as np
import argparse
from copy import copy
from utils.const import *
from utils.physics import *


class root2pickle():
    '''class to read root to make epg pairs'''
    def __init__(self, fname, entry_start = None, entry_stop = None, pol = "inbending", gen = "norad", raw = False, detRes = False, width = "mid", dvcs = False, smearing = 1, nofid = False):
        '''
            clas init.
            Args
            --------------------
            fname: root file name to be read
            entry_start: the lower bound of root entry
            entry_stop: the upper bound of root entry
            pol: polarity
            gen: generator
            raw: no exclusivity cuts
            detRes: include detector responses in the output (full skim)
            width: data selection window width
            nofid: do not apply fid cuts.
            Attributes
            --------------------
            fname: root file name to be read
            Methods
            --------------------
            determineWidth: to determine event selection window
            readFile: read root file to pandas
            closeFile: nullify the files to save memory
            readEPGG: read and post-process data. fiducial cut/ proton energy loss correction/ smearing.
            saveDVpi0vars: 4 momentum algebra to save DVpi0P vars
            makeDVpi0P: select pi0->2g events.
            makeDVpi0P: select DVpi0P candidates
            save: save output
        '''
        self.fname = fname

        self.determineWidth(width = width)
        self.readEPGG(entry_start = entry_start, entry_stop = entry_stop, pol = pol, gen = gen, detRes = detRes, smearing = smearing, nofid = nofid)
        self.saveDVpi0vars()
        if not raw:
            self.makeDVpi0P(pol = pol)
        self.save(raw = raw, dvcs = dvcs, pol = pol)

    def readFile(self):
        '''read root using uproot'''
        self.file = uproot.open(self.fname)
        self.tree = self.file["T"]

    def closeFile(self):
        '''close file for saving memory'''
        self.file = None
        self.tree = None

    def determineWidth(self, width = "mid"):
        '''determine event selection window'''
        print("determine width level: {}".format(width))
        if width == "default":
            self.Ge2Threshold = Ge2Threshold_default
            # self.cuts_dvpi0p_CDFT_Inb = cuts_dvpi0p_default
            # self.cuts_dvpi0p_CD_Inb = cuts_dvpi0p_default
            # self.cuts_dvpi0p_FD_Inb = cuts_dvpi0p_default
            # self.cuts_dvpi0p_CDFT_Outb = cuts_dvpi0p_default
            # self.cuts_dvpi0p_CD_Outb = cuts_dvpi0p_default
            # self.cuts_dvpi0p_FD_Outb = cuts_dvpi0p_default
        if width == "mid":
            self.Ge2Threshold = Ge2Threshold_mid
            # self.cuts_dvpi0p_CDFT_Inb = cuts_dvpi0p_CDFT_Inb_3sigma
            # self.cuts_dvpi0p_CD_Inb = cuts_dvpi0p_CD_Inb_3sigma
            # self.cuts_dvpi0p_FD_Inb = cuts_dvpi0p_FD_Inb_3sigma
            # self.cuts_dvpi0p_CDFT_Outb = cuts_dvpi0p_CDFT_Outb_3sigma
            # self.cuts_dvpi0p_CD_Outb = cuts_dvpi0p_CD_Outb_3sigma
            # self.cuts_dvpi0p_FD_Outb = cuts_dvpi0p_FD_Outb_3sigma
        if width == "tight":
            self.Ge2Threshold = Ge2Threshold_tight
            # self.cuts_dvpi0p_CDFT_Inb = cuts_dvpi0p_CDFT_Inb_2sigma
            # self.cuts_dvpi0p_CD_Inb = cuts_dvpi0p_CD_Inb_2sigma
            # self.cuts_dvpi0p_FD_Inb = cuts_dvpi0p_FD_Inb_2sigma
            # self.cuts_dvpi0p_CDFT_Outb = cuts_dvpi0p_CDFT_Outb_2sigma
            # self.cuts_dvpi0p_CD_Outb = cuts_dvpi0p_CD_Outb_2sigma
            # self.cuts_dvpi0p_FD_Outb = cuts_dvpi0p_FD_Outb_2sigma
        if width == "loose":
            self.Ge2Threshold = Ge2Threshold_loose
            # self.cuts_dvpi0p_CDFT_Inb = cuts_dvpi0p_CDFT_Inb_4sigma
            # self.cuts_dvpi0p_CD_Inb = cuts_dvpi0p_CD_Inb_4sigma
            # self.cuts_dvpi0p_FD_Inb = cuts_dvpi0p_FD_Inb_4sigma
            # self.cuts_dvpi0p_CDFT_Outb = cuts_dvpi0p_CDFT_Outb_4sigma
            # self.cuts_dvpi0p_CD_Outb = cuts_dvpi0p_CD_Outb_4sigma
            # self.cuts_dvpi0p_FD_Outb = cuts_dvpi0p_FD_Outb_4sigma

    def readEPGG(self, entry_start = None, entry_stop = None, gen = "pi0norad", pol = "inbending", detRes = False, smearing = 1, nofid = False):
        '''save data into df_epg, df_epgg for parent class epg'''
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

        if gen == "pi0rad":
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
            self.df_MC = df_MC    #done with saving MC

        else:
            #two g's to one gg.
            gamGen = [df_gammaGen["GenGpx"], df_gammaGen["GenGpy"], df_gammaGen["GenGpz"]]
            df_gammaGen.loc[:, 'GenGp'] = mag(gamGen)

            gam1 = df_gammaGen[df_gammaGen.index.get_level_values('subentry')==0]
            gam1 = gam1.reset_index(drop=True)
            gam2 = df_gammaGen[df_gammaGen.index.get_level_values('subentry')==1]
            gam2 = gam2.reset_index(drop=True)

            gam1.loc[:,"GenGp2"] = gam2.loc[:,"GenGp"]
            gam1.loc[:,"GenGpx2"] = gam2.loc[:,"GenGpx"]
            gam1.loc[:,"GenGpy2"] = gam2.loc[:,"GenGpy"]
            gam1.loc[:,"GenGpz2"] = gam2.loc[:,"GenGpz"]
            df_gammaGen = gam1

            #sort GenG indices so that GenGp > GenGp2. This is because Gp > Gp2 at reconstruction level.
            df_gammaGencopy = copy(df_gammaGen)
            df_gammaGencopy.loc[:, "GenGp"] = np.where(df_gammaGen["GenGp"]>df_gammaGen["GenGp2"], df_gammaGen.loc[:, "GenGp"], df_gammaGen.loc[:, "GenGp2"])
            df_gammaGencopy.loc[:, "GenGpx"] = np.where(df_gammaGen["GenGp"]>df_gammaGen["GenGp2"], df_gammaGen.loc[:, "GenGpx"], df_gammaGen.loc[:, "GenGpx2"])
            df_gammaGencopy.loc[:, "GenGpy"] = np.where(df_gammaGen["GenGp"]>df_gammaGen["GenGp2"], df_gammaGen.loc[:, "GenGpy"], df_gammaGen.loc[:, "GenGpy2"])
            df_gammaGencopy.loc[:, "GenGpz"] = np.where(df_gammaGen["GenGp"]>df_gammaGen["GenGp2"], df_gammaGen.loc[:, "GenGpz"], df_gammaGen.loc[:, "GenGpz2"])
            df_gammaGencopy.loc[:, "GenGp2"] = np.where(df_gammaGen["GenGp"]>df_gammaGen["GenGp2"], df_gammaGen.loc[:, "GenGp2"], df_gammaGen.loc[:, "GenGp"])
            df_gammaGencopy.loc[:, "GenGpx2"] = np.where(df_gammaGen["GenGp"]>df_gammaGen["GenGp2"], df_gammaGen.loc[:, "GenGpx2"], df_gammaGen.loc[:, "GenGpx"])
            df_gammaGencopy.loc[:, "GenGpy2"] = np.where(df_gammaGen["GenGp"]>df_gammaGen["GenGp2"], df_gammaGen.loc[:, "GenGpy2"], df_gammaGen.loc[:, "GenGpy"])
            df_gammaGencopy.loc[:, "GenGpz2"] = np.where(df_gammaGen["GenGp"]>df_gammaGen["GenGp2"], df_gammaGen.loc[:, "GenGpz2"], df_gammaGen.loc[:, "GenGpz"])
            df_gammaGen = df_gammaGencopy

            gamGen = [df_gammaGen["GenGpx"], df_gammaGen["GenGpy"], df_gammaGen["GenGpz"]]
            # df_gammaGen.loc[:, 'GenGp'] = mag(gamGen)
            df_gammaGen.loc[:, 'GenGtheta'] = getTheta(gamGen)
            df_gammaGen.loc[:, 'GenGphi'] = getPhi(gamGen)

            gamGen2 = [df_gammaGen["GenGpx2"], df_gammaGen["GenGpy2"], df_gammaGen["GenGpz2"]]
            df_gammaGen.loc[:, 'GenGtheta2'] = getTheta(gamGen2)
            df_gammaGen.loc[:, 'GenGphi2'] = getPhi(gamGen2)

            df_MC = pd.merge(df_MC, df_gammaGen, how='inner', on='event')
        self.df_MC = df_MC    #done with saving z

        print("generator mode: ", gen)
        print("debug:: number of events", len(df_electronGen))
        print("debug:: number of all MC df", len(df_MC))

        # data frames and their keys to read X part
        df_electronRec = pd.DataFrame()
        df_protonRec = pd.DataFrame()
        df_gammaRec = pd.DataFrame()
        eleKeysRec = ["Epx", "Epy", "Epz", "Eedep", "Esector"]
        proKeysRec = ["Ppx", "Ppy", "Ppz", "Pstat", "Psector"]
        proKeysRec.extend(["PDc1Hitx", "PDc1Hity", "PDc1Hitz", "PCvt12Hitx", "PCvt12Hity", "PCvt12Hitz"])
        # proKeysRec.extend(["Pchi2pid", "Pchi2track", "PNDFtrack"])
        gamKeysRec = ["Gpx", "Gpy", "Gpz", "Gedep", "GcX", "GcY", "Gsector"]

        if detRes:
            eleKeysRec.extend(["Evx", "Evy", "Evz"])
            eleKeysRec.extend(["EDc1Hitx", "EDc1Hity", "EDc1Hitz", "EDc3Hitx", "EDc3Hity", "EDc3Hitz"])
            eleKeysRec.extend(["Eedep1", "Eedep2", "Eedep3"])
            eleKeysRec.extend(["EcalU1", "EcalV1", "EcalW1"])
            eleKeysRec.extend(["Enphe"])
            gamKeysRec.extend(["Gedep1", "Gedep2", "Gedep3"])
            gamKeysRec.extend(["GcalU1", "GcalV1", "GcalW1"])
            gamKeysRec.extend(["Gbeta"])
            proKeysRec.extend(["Pvz"])
            proKeysRec.extend(["PCvt1Hitx", "PCvt1Hity", "PCvt1Hitz", "PCvt3Hitx", "PCvt3Hity", "PCvt3Hitz", "PCvt5Hitx", "PCvt5Hity", "PCvt5Hitz", "PCvt7Hitx", "PCvt7Hity", "PCvt7Hitz"])
            proKeysRec.extend(["PDc1Hitx", "PDc1Hity", "PDc1Hitz", "PDc3Hitx", "PDc3Hity", "PDc3Hitz"])
            eleKeysRec.extend(["startTime"])
            proKeysRec.extend(["PFtof1aTime", "PFtof1bTime", "PFtof2Time", "PCtofTime"])
            proKeysRec.extend(["PFtof1aHitx", "PFtof1bHitx", "PFtof2Hitx", "PCtofHitx"])
            proKeysRec.extend(["PFtof1aHity", "PFtof1bHity", "PFtof2Hity", "PCtofHity"])
            proKeysRec.extend(["PFtof1aHitz", "PFtof1bHitz", "PFtof2Hitz", "PCtofHitz"])
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
        df_electronRec = df_electronRec.astype({"Epx": float, "Epy": float, "Epz": float})
        df_protonRec = df_protonRec.astype({"Ppx": float, "Ppy": float, "Ppz": float})
        df_gammaRec = df_gammaRec.astype({"Gpx": float, "Gpy": float, "Gpz": float, "Gedep": float, "GcX": float, "GcY": float})

        #apply photon fiducial cuts
        if nofid:
            df_gammaRec.loc[:, "GFid"] = 1
        else:
            #photon FD fiducial cuts by F.X. Girod
            df_gammaRec.loc[:, "GFid"] = 0

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
            df_gammaRec.loc[cond1, "GFid"] = 1
            #second condition else if the first
            # cond2_0 = df_gammaRec.GFid == 0 # not necessary, because cond2_1 rules out the first (S. Lee)
            cond2_1 = df_gammaRec.GcX < psplit
            cond2_2 = df_gammaRec.GcY < qleft * (df_gammaRec.GcX - rleft)
            cond2_3 = df_gammaRec.GcY > qright * (df_gammaRec.GcX - rright)
            cond2_4 = df_gammaRec.Gsector < 7
            cond2 = cond2_1 & cond2_2 & cond2_3 & cond2_4
            df_gammaRec.loc[cond2, "GFid"] = 1

            df_gammaRec.loc[df_gammaRec.Gsector > 7, "GFid"] = 1

            #FT fiducial cuts
            circleCenterX1 = -8.419
            circleCenterY1 = 9.889
            circleRadius1 = 1.6

            circleCenterX2 = -9.89
            circleCenterY2 = -5.327
            circleRadius2 = 1.6

            circleCenterX3 = -6.15
            circleCenterY3 = -13
            circleRadius3 = 2.3

            circleCenterX4 = 3.7
            circleCenterY4 = -6.5
            circleRadius4 = 2
            
            circle1 = (df_gammaRec.GcX - circleCenterX1)**2 + (df_gammaRec.GcY - circleCenterY1)**2 < circleRadius1**2
            circle2 = (df_gammaRec.GcX - circleCenterX2)**2 + (df_gammaRec.GcY - circleCenterY2)**2 < circleRadius2**2
            circle3 = (df_gammaRec.GcX - circleCenterX3)**2 + (df_gammaRec.GcY - circleCenterY3)**2 < circleRadius3**2
            circle4 = (df_gammaRec.GcX - circleCenterX4)**2 + (df_gammaRec.GcY - circleCenterY4)**2 < circleRadius4**2

            df_gammaRec.loc[(df_gammaRec.Gsector > 7) & circle1, "GFid"] = 0
            df_gammaRec.loc[(df_gammaRec.Gsector > 7) & circle2, "GFid"] = 0
            df_gammaRec.loc[(df_gammaRec.Gsector > 7) & circle3, "GFid"] = 0
            df_gammaRec.loc[(df_gammaRec.Gsector > 7) & circle4, "GFid"] = 0

        #set up a dummy index for merging
        df_electronRec.loc[:,'event'] = df_electronRec.index
        df_protonRec.loc[:,'event'] = df_protonRec.index.get_level_values('entry')
        df_gammaRec.loc[:,'event'] = df_gammaRec.index.get_level_values('entry')
        df_gammaRec.loc[:,'GIndex'] = df_gammaRec.index.get_level_values('subentry')

        #prepare for proton energy loss corrections correction
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
        else:
            df_protonRec.loc[:, "PCvt12theta"] = -100000
            df_protonRec.loc[:, "PCvt12phi"] = -100000

        df_protonRecFD = df_protonRec.loc[df_protonRec.Psector<7, :]
        df_protonRecCD = df_protonRec.loc[(df_protonRec.Psector>7) & (df_protonRec.Ptheta<75), :]
        df_protonRecOthers = df_protonRec.loc[(df_protonRec.Psector>7) & (df_protonRec.Ptheta>=75), :]

        correction = False

        #two band criterion
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
        else:
            df_protonRecCD.loc[:, "PCvt12theta"] = getTheta([df_protonRecCD.PCvt12Hitx, df_protonRecCD.PCvt12Hity, df_protonRecCD.PCvt12Hitz])
            df_protonRecCD.loc[:, "PCvt12phi"] = getPhi([df_protonRecCD.PCvt12Hitx, df_protonRecCD.PCvt12Hity, df_protonRecCD.PCvt12Hitz])

        #inbending proton energy loss correction
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
        #outbending proton energy loss correction
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
            print("energy loss correction applied for " + pol)

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

            #smearing proton after the energy loss correction
            print("smearing factor {} from nominal".format(smearing))
            #CD proton
            def cubic(args, x): #equivalent to poly1d
                a, b, c, d = args
                return a*x**3 +b*x**2 + c*x + d
            regulator = np.abs(2*(1/(1+np.exp(-(df_protonRec.loc[df_protonRec["Psector"]>7, "Pp"]-0.3)/0.01))-0.5))
            sigma1_CD = np.where(df_protonRec.loc[df_protonRec["Psector"]>7, "Pp"]<0.85, cubic([0.0926, 0.137, -0.230, 0.139], df_protonRec.loc[df_protonRec["Psector"]>7, "Pp"]), 0.1)
            sigma2_CD = np.where(df_protonRec.loc[df_protonRec["Psector"]>7, "Pp"]<1.34, cubic([-2.797, 9.351, -9.488, 3.503], df_protonRec.loc[df_protonRec["Psector"]>7, "Pp"]), 0.85)
            sigma3_CD = 0.8 + 2.2/(1+np.exp(5.518*(df_protonRec.loc[df_protonRec.Psector>7, "Pp"]-0.625)))
            df_protonRec.loc[df_protonRec["Psector"]>7, "Pp"] = df_protonRec.loc[df_protonRec["Psector"]>7, "Pp"]*np.random.normal(1, smearing*regulator*sigma1_CD, len(df_protonRec.loc[df_protonRec.Psector>7]))
            df_protonRec.loc[df_protonRec["Psector"]>7, "Ptheta"] = df_protonRec.loc[df_protonRec["Psector"]>7, "Ptheta"] + np.random.normal(0, smearing*sigma2_CD, len(df_protonRec.loc[df_protonRec.Psector>7]))
            df_protonRec.loc[df_protonRec["Psector"]>7, "Pphi"] = df_protonRec.loc[df_protonRec["Psector"]>7, "Pphi"] + np.random.normal(0, smearing*sigma3_CD, len(df_protonRec.loc[df_protonRec.Psector>7])) 
            #FD proton
            args_sigmas_FD_inb = [[-0.233, 1.216, -2.279, 1.812, -0.445], [ 0.277, -1.366, 2.318, -1.619,  0.466 ],[ 0.0728, -0.223, 0.0888,  0.225, -0.0889],[-0.204, 0.977, -1.766, 1.411, -0.342], [ 0.277, -1.059, 1.362, -0.641, 0.137], [-0.219, 1.132, -2.153, 1.763, -0.447]]
            args_sigmas_FD_outb = [[0.481,-1.548, 1.524,-0.415, 0.0277], [1.872, -8.054, 12.536, -8.358,  2.083], [-0.0656, 0.480, -1.191, 1.169, -0.315], [-1.559, 7.356, -12.639, 9.312,  -2.405], [ 0.189, -0.344, -0.253,  0.717, -0.238], [0.466, -1.560, 1.622, -0.485, 0.0322]]
            def quartic(x, sector, pol = "inbending"):
                if pol == "inbending":
                    a, b, c, d, e = args_sigmas_FD_inb[sector - 1]
                    return np.select( [x<0.55, (x>=0.55)& (x < 1.55), x>=1.55], [a*0.55**4+b*0.55**3+c*0.55**2+d*0.55+e, a*x**4 +b*x**3 + c*x**2 + d*x + e, a*1.55**4 + b*1.55**3 + c*1.55**2+d*1.55 +e])
                if pol == "outbending":
                    a, b, c, d, e = args_sigmas_FD_outb[sector - 1]
                    return np.select( [x<0.65, (x>=0.65)& (x < 1.55), x>=1.55], [a*0.65**4+b*0.65**3+c*0.65**2+d*0.65+e, a*x**4 +b*x**3 + c*x**2 + d*x + e, a*1.55**4 + b*1.55**3 + c*1.55**2+d*1.55 +e])
            def sigmaFDOutb(x):
                return np.select([x<.95, (x>=.95) & (x<1.2), (x>=1.2)&(x<1.575), (x>=1.575) & (x<1.9), (x>1.9)], [0.1, -0.045/(1.2-.95)*x+1.2*0.045/(1.2-.95) + 0.055, 0.055, -0.015/(1.9-1.575)*x+1.9*0.015/(1.9-1.575) + 0.04,0.04])

            for sector in range(1, 7):
                if pol == "inbending":
                    regulator = (1/(1+np.exp(-(df_protonRec.loc[df_protonRec["Psector"]==sector, "Pp"]-0.5)/0.05)))
                    sigmas_FD = quartic(df_protonRec.loc[df_protonRec.Psector == sector, "Pp"], sector, pol)
                elif pol == "outbending":
                    regulator = (1/(1+np.exp(-(df_protonRec.loc[df_protonRec["Psector"]==sector, "Pp"]-0.6)/0.05)))
                    sigmas_FD = sigmaFDOutb(df_protonRec.loc[df_protonRec["Psector"]==sector, "Pp"]) #quartic(df_protonRec.loc[df_protonRec.Psector == sector, "Pp"], sector, pol)
                df_protonRec.loc[df_protonRec["Psector"]==sector, "Pp"] = df_protonRec.loc[df_protonRec["Psector"]==sector, "Pp"]*np.random.normal(1, smearing*regulator*sigmas_FD, len(df_protonRec.loc[df_protonRec["Psector"]==sector, "Pp"]))

            #moduli proton phi
            df_protonRec.loc[:, "Pphi"] = np.where(df_protonRec.loc[:, "Pphi"]%360<180, df_protonRec.loc[:, "Pphi"]%360, df_protonRec.loc[:, "Pphi"]%360-360)

            df_protonRec.loc[:, "Ppx"] = df_protonRec.loc[:, "Pp"]*np.sin(np.radians(df_protonRec.loc[:, "Ptheta"]))*np.cos(np.radians(df_protonRec.loc[:, "Pphi"]))
            df_protonRec.loc[:, "Ppy"] = df_protonRec.loc[:, "Pp"]*np.sin(np.radians(df_protonRec.loc[:, "Ptheta"]))*np.sin(np.radians(df_protonRec.loc[:, "Pphi"]))
            df_protonRec.loc[:, "Ppz"] = df_protonRec.loc[:, "Pp"]*np.cos(np.radians(df_protonRec.loc[:, "Ptheta"]))

            pro = [df_protonRec['Ppx'], df_protonRec['Ppy'], df_protonRec['Ppz']]
            df_protonRec.loc[:, 'Pe'] = getEnergy(pro, M)

            #smearing photon
            gam = [df_gammaRec['Gpx'], df_gammaRec['Gpy'], df_gammaRec['Gpz']]
            df_gammaRec.loc[:, 'Gp'] = mag(gam)
            df_gammaRec.loc[:, 'Gtheta'] = getTheta(gam)
            df_gammaRec.loc[:, 'Gphi'] = getPhi(gam)
            #FT photon
            df_gammaRec.loc[df_gammaRec["Gsector"]>7, "Gp"] = df_gammaRec.loc[df_gammaRec["Gsector"]>7, "Gp"]*np.random.normal(1, smearing*(0.013 + 0.003/(1+np.exp(0.761*(df_gammaRec.loc[df_gammaRec["Gsector"]>7, "Gp"]-6)))), len(df_gammaRec.loc[df_gammaRec.Gsector>7]))
            #FD photon
            df_gammaRec.loc[df_gammaRec["Gsector"]<7, "Gp"] = df_gammaRec.loc[df_gammaRec["Gsector"]<7, "Gp"]*np.random.normal(1, smearing*(0.0395/(1+np.exp(5.308*(df_gammaRec.loc[df_gammaRec["Gsector"]<7, "Gp"]- 8.005)))), len(df_gammaRec.loc[df_gammaRec.Gsector<7]))

            df_gammaRec.loc[:, "Gpx"] = df_gammaRec.loc[:, "Gp"]*np.sin(np.radians(df_gammaRec.loc[:, "Gtheta"]))*np.cos(np.radians(df_gammaRec.loc[:, "Gphi"]))
            df_gammaRec.loc[:, "Gpy"] = df_gammaRec.loc[:, "Gp"]*np.sin(np.radians(df_gammaRec.loc[:, "Gtheta"]))*np.sin(np.radians(df_gammaRec.loc[:, "Gphi"]))
            df_gammaRec.loc[:, "Gpz"] = df_gammaRec.loc[:, "Gp"]*np.cos(np.radians(df_gammaRec.loc[:, "Gtheta"]))

        ele = [df_electronRec['Epx'], df_electronRec['Epy'], df_electronRec['Epz']]
        df_electronRec.loc[:, 'Ep'] = mag(ele)
        df_electronRec.loc[:,'ESamplFrac'] = df_electronRec.Eedep/ df_electronRec.Ep
        df_gammaRec.loc[:,'GSamplFrac'] = df_gammaRec.Gedep/ df_gammaRec.Gp

        df_gg = pd.merge(df_gammaRec, df_gammaRec,
                         how='outer', on='event', suffixes=("", "2"))
        df_gg = df_gg[df_gg["GIndex"] < df_gg["GIndex2"]]
        df_gg = df_gg.drop(['GIndex', 'GIndex2'], axis = 1)

        # proton fiducial cuts
        if nofid:
            df_protonRec.loc[:, "PFid"] = 1
        else:
            df_protonRec.loc[:, "PFid"] = 0

            df_protonRec.loc[df_protonRec.Psector<7, "PFid"] = 1 #FD fid done by previous pipeline

            cut_CD = df_protonRec.Psector > 7
            cut_right = cut_CD & (df_protonRec.Ptheta<64.23)
            cut_bottom = cut_CD & (df_protonRec.PCvt12theta>44.5)
            cut_sidel = cut_CD & (df_protonRec.PCvt12theta<-2.942 + 1.274*df_protonRec.Ptheta)
            cut_sider = cut_CD & (df_protonRec.PCvt12theta>-3.523 + 1.046*df_protonRec.Ptheta)

            cut_trapezoid = cut_CD & cut_right & cut_bottom & cut_sidel & cut_sider

            cut_gaps1 = ~((df_protonRec.PCvt12phi>-95) & (df_protonRec.PCvt12phi<-80))
            cut_gaps2 = ~((df_protonRec.PCvt12phi>25) & (df_protonRec.PCvt12phi<40))
            cut_gaps3 = ~((df_protonRec.PCvt12phi>143) & (df_protonRec.PCvt12phi<158))
            cut_gaps = cut_CD & cut_gaps1 & cut_gaps2 & cut_gaps3
            cut_total = cut_gaps & cut_trapezoid

            df_protonRec.loc[cut_total, "PFid"] = 1 #CD fid

        if detRes:
            df_gg = df_gg.loc[:, ~df_gg.columns.duplicated()]
            df_gg.loc[:, "Gedep2_tot"] = df_gg.Gedep12 + df_gg.Gedep22 + df_gg.Gedep32
        else:
            df_protonRec = df_protonRec.drop(["PDc1Hitx", "PDc1Hity", "PDc1Hitz", "PDc1theta", "PCvt12Hitx", "PCvt12Hity", "PCvt12Hitz", "PCvt12theta", "PCvt12phi"], axis = 1)
            df_gammaRec = df_gammaRec.drop(["GcX", "GcY"], axis = 1)
            df_gg = df_gg.drop(["GcX", "GcY", "GcX2", "GcY2"], axis = 1)

        df_ep = pd.merge(df_electronRec, df_protonRec, how='outer', on='event')

        df_epgg = pd.merge(df_ep, df_gg, how='outer', on='event')
        df_epgg = df_epgg[~np.isnan(df_epgg["Ppx"])]
        df_epgg = df_epgg[~np.isnan(df_epgg["Gpx"])]
        df_epgg = df_epgg[~np.isnan(df_epgg["Gpx2"])]

        self.df_epgg = df_epgg #temporarily saves df_epgg

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
        df_epgg.loc[:,'y'] = df_epgg['nu']/ebeam
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
        df_epgg.loc[:,'openingAngle'] = angle(gam, gam2)

        df_epgg.loc[:, "closeness"] = np.abs(df_epgg.loc[:, "Mpi0"] - .1349766)

        # encode unassigned bin as -1
        df_epgg.loc[:, "Q2bin"] = -1
        df_epgg.loc[:, "xBbin"] = -1
        df_epgg.loc[:, "tbin"] = -1
        # df_epgg.loc[:, "tbin2"] = -1
        df_epgg.loc[:, "phibin"] = -1
        # df_epgg.loc[:, "phibin2"] = -1
        df_epgg.loc[:, "Q2xBbin"] = -1
        df_epgg.loc[:, "Q2xBtbin"] = -1
        # df_epgg.loc[:, "Q2xBtbin2"] = -1
        df_epgg.loc[:, "Q2xBtphibin"] = -1
        Q2xBbin = 0

        # encode all binning
        for Q2bin in range(len(Q2bin_i)):
            #square Q2 binning
            df_epgg.loc[(df_epgg.Q2>=Q2bin_i[Q2bin]) & (df_epgg.Q2<Q2bin_f[Q2bin]), "Q2bin"] = Q2bin
            #adaptive xB binning
            for xBbin in range(len(xBbin_i[Q2bin])):
                if Q2bin < len(Q2bin_i) -1:
                    if xBbin == 0:
                        df_epgg.loc[(df_epgg.Q2>=Q2bin_i[Q2bin]) & (df_epgg.Q2<Q2bin_f[Q2bin]) & (df_epgg.Q2<=2*M*(10.604-2)*df_epgg.xB) & (df_epgg.xB<xBbin_f[Q2bin][xBbin]), "xBbin"] = xBbin #0
                        df_epgg.loc[(df_epgg.Q2>=Q2bin_i[Q2bin]) & (df_epgg.Q2<Q2bin_f[Q2bin]) & (df_epgg.Q2<=2*M*(10.604-2)*df_epgg.xB) & (df_epgg.xB<xBbin_f[Q2bin][xBbin]), "Q2xBbin"] = Q2xBbin #0
                    elif xBbin < len(xBbin_i[Q2bin])-1:
                        df_epgg.loc[(df_epgg.Q2>=Q2bin_i[Q2bin]) & (df_epgg.Q2<Q2bin_f[Q2bin]) & (df_epgg.xB>=xBbin_i[Q2bin][xBbin]) & (df_epgg.xB<xBbin_f[Q2bin][xBbin]), "xBbin"] = xBbin
                        df_epgg.loc[(df_epgg.Q2>=Q2bin_i[Q2bin]) & (df_epgg.Q2<Q2bin_f[Q2bin]) & (df_epgg.xB>=xBbin_i[Q2bin][xBbin]) & (df_epgg.xB<xBbin_f[Q2bin][xBbin]), "Q2xBbin"] = Q2xBbin
                    else:
                        df_epgg.loc[(df_epgg.Q2>=Q2bin_i[Q2bin]) & (df_epgg.Q2<Q2bin_f[Q2bin]) & (df_epgg.xB>=xBbin_i[Q2bin][xBbin]) & (df_epgg.Q2>=(4-M*M)*df_epgg.xB/(1-df_epgg.xB)), "xBbin"] = xBbin
                        df_epgg.loc[(df_epgg.Q2>=Q2bin_i[Q2bin]) & (df_epgg.Q2<Q2bin_f[Q2bin]) & (df_epgg.xB>=xBbin_i[Q2bin][xBbin]) & (df_epgg.Q2>=(4-M*M)*df_epgg.xB/(1-df_epgg.xB)), "Q2xBbin"] = Q2xBbin
                else:
                    df_epgg.loc[(df_epgg.Q2>=Q2bin_i[Q2bin]) & (df_epgg.Q2<Q2bin_f[Q2bin]) & (df_epgg.Q2<=2*M*(10.604-2)*df_epgg.xB)& (df_epgg.Q2>=(4-M*M)*df_epgg.xB/(1-df_epgg.xB)), "xBbin"] = xBbin
                    df_epgg.loc[(df_epgg.Q2>=Q2bin_i[Q2bin]) & (df_epgg.Q2<Q2bin_f[Q2bin]) & (df_epgg.Q2<=2*M*(10.604-2)*df_epgg.xB)& (df_epgg.Q2>=(4-M*M)*df_epgg.xB/(1-df_epgg.xB)), "Q2xBbin"] = Q2xBbin #0

                Q2xBbin = Q2xBbin + 1
        for tbin in range(len(tbin_i)):
            #square t binning
            df_epgg.loc[(df_epgg.t1>=tbin_i[tbin]) & (df_epgg.t1<tbin_f[tbin]), "tbin"] = tbin
            # df_epgg.loc[(df_epgg.t2>=tbin_i[tbin]) & (df_epgg.t2<tbin_f[tbin]), "tbin2"] = tbin
        for phibin in range(len(phibin_i)):
            #square phi binning
            df_epgg.loc[(df_epgg.phi1>=phibin_i[phibin]) & (df_epgg.phi1<phibin_f[phibin]), "phibin"] = phibin
            # df_epgg.loc[(df_epgg.phi2>=phibin_i[phibin]) & (df_epgg.phi2<phibin_f[phibin]), "phibin2"] = phibin

        df_epgg.loc[(df_epgg.Q2xBbin>=0)&(df_epgg.tbin>=0), "Q2xBtbin"] = len(tbin_i) * df_epgg.loc[(df_epgg.Q2xBbin>=0)&(df_epgg.tbin>=0), "Q2xBbin"] + df_epgg.loc[(df_epgg.Q2xBbin>=0)&(df_epgg.tbin>=0), "tbin"]
        # df_epgg.loc[(df_epgg.Q2bin>0)&(df_epgg.xBbin>0)&(df_epgg.tbin2>0), "Q2xBtbin2"] = df_epgg.Q2bin.astype(str) + df_epgg.xBbin.astype(str) + df_epgg.tbin2.astype(str)
        df_epgg.loc[(df_epgg.Q2xBbin>=0)&(df_epgg.tbin>=0), "Q2xBtphibin"] = len(phibin_i) * df_epgg.loc[(df_epgg.Q2xBbin>=0)&(df_epgg.tbin>=0), "Q2xBtbin"] + df_epgg.loc[(df_epgg.Q2xBbin>=0)&(df_epgg.tbin>=0), "phibin"]

        df_epgg = df_epgg.astype({"Q2bin": int, "xBbin": int, "tbin": int, "phibin": int, "Q2xBbin": int, "Q2xBtbin": int, "Q2xBtphibin": int})

        self.df_epgg = df_epgg

    def makeDVpi0P(self, pol = "inbending"):
        #make dvpi0 pairs
        df_dvpi0p = self.df_epgg

        #common cuts
        cut_xBupper = df_dvpi0p.loc[:, "xB"] < 1  # xB
        cut_xBlower = df_dvpi0p.loc[:, "xB"] > 0  # xB
        cut_Q2 = df_dvpi0p.loc[:, "Q2"] > 1  # Q2
        cut_W = df_dvpi0p.loc[:, "W"] > 2  # W
        cut_Ee = df_dvpi0p["Ee"] > 2  # Ee
        cut_Ge2 = df_dvpi0p["Ge2"] > self.Ge2Threshold  # Ge cut. Ge>3 for DVCS module.
        cut_Esector = (df_dvpi0p["Esector"]!=df_dvpi0p["Gsector"]) & (df_dvpi0p["Esector"]!=df_dvpi0p["Gsector2"])
        cut_Psector = ~( ((df_dvpi0p["Pstat"]//10)%10>0) & (df_dvpi0p["Psector"]==df_dvpi0p["Gsector"]) ) & ~( ((df_dvpi0p["Pstat"]//10)%10>0) & (df_dvpi0p["Psector"]==df_dvpi0p["Gsector2"]) )
        cut_Ppmax = df_dvpi0p.Pp < 1.6  # Pp
        cut_Pthetamin = df_dvpi0p.Ptheta > 0  # Ptheta
        # cut_Vz = np.abs(df_dvcs["Evz"] - df_dvcs["Pvz"]) < 2.5 + 2.5 / mag([df_dvcs["Ppx"], df_dvcs["Ppy"], df_dvcs["Ppz"]])
        cut_common = cut_xBupper & cut_xBlower & cut_Q2 & cut_W & cut_Ee & cut_Ge2 & cut_Esector & cut_Psector & cut_Ppmax & cut_Pthetamin

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
        self.df_dvpi0p = df_dvpi0p #done with saving x

    def save(self, raw = False, dvcs = False, pol = "inbending"):
        if raw and not dvcs:
            print("saving raw with common cuts")
            df_Rec = self.df_epgg
            #common cuts
            cut_xBupper = df_Rec.loc[:, "xB"] < 1  # xB
            cut_xBlower = df_Rec.loc[:, "xB"] > 0  # xB
            cut_Q2 = df_Rec.loc[:, "Q2"] > 1  # Q2
            cut_W = df_Rec.loc[:, "W"] > 2  # W
            cut_Ee = df_Rec["Ee"] > 2  # Ee
            cut_Ge2 = df_Rec["Ge2"] > 0.6  # Ge cut. Ge>3 for DVCS module.
            cut_Esector = (df_Rec["Esector"]!=df_Rec["Gsector"]) & (df_Rec["Esector"]!=df_Rec["Gsector2"]) 
            cut_Psector = ~( ((df_Rec["Pstat"]//10)%10>0) & (df_Rec["Psector"]==df_Rec["Gsector"]) ) & ~( ((df_Rec["Pstat"]//10)%10>0) & (df_Rec["Psector"]==df_Rec["Gsector2"]) )
            cut_Ppmax = df_Rec.Pp < 1.6  # Pp
            cut_Pthetamin = df_Rec.Ptheta > 0  # Ptheta
            # cut_Vz = np.abs(df_dvcs["Evz"] - df_dvcs["Pvz"]) < 2.5 + 2.5 / mag([df_dvcs["Ppx"], df_dvcs["Ppy"], df_dvcs["Ppz"]])
            cut_common = cut_xBupper & cut_xBlower & cut_Q2 & cut_W & cut_Ee & cut_Ge2 & cut_Esector & cut_Psector & cut_Ppmax & cut_Pthetamin
            df_Rec = df_Rec[cut_common]

            #CDFT
            cut_Pp1_CDFT = df_Rec.Pp > 0.3  # Pp
            cut_Psector_CDFT = df_Rec.Psector>7
            cut_Ptheta1_CDFT = df_dvpi0p.Ptheta<CD_Ptheta_ub
            cut_Ptheta2_CDFT = df_dvpi0p.Ptheta>CD_Ptheta_lb
            cut_Gsector_CDFT = df_Rec.Gsector>7
            cut_Gsector2_CDFT = df_Rec.Gsector2>7
            cut_GFid_CDFT = df_Rec.GFid==1
            cut_GFid2_CDFT = df_Rec.GFid2==1
            cut_PFid_CDFT = df_Rec.PFid==1
            #CD
            cut_Pp1_CD = df_Rec.Pp > 0.3  # Pp
            cut_Psector_CD = df_Rec.Psector>7
            cut_Ptheta1_CD = df_dvpi0p.Ptheta<CD_Ptheta_ub
            cut_Ptheta2_CD = df_dvpi0p.Ptheta>CD_Ptheta_lb
            cut_Gsector_CD = (df_Rec.Gsector<7) & (df_Rec.Gsector>0)
            cut_Gsector2_CD = (df_Rec.Gsector2<7) & (df_Rec.Gsector2>0)
            cut_GFid_CD = df_Rec.GFid==1
            cut_GFid2_CD = df_Rec.GFid2==1
            cut_PFid_CD = df_Rec.PFid==1
            #FD
            if pol == "inbending":
                cut_Pp1_FD = df_Rec.Pp > 0.42  # Pp
            elif pol == "outbending":
                cut_Pp1_FD = df_Rec.Pp > 0.5  # Pp
            cut_Psector_FD = df_Rec.Psector<7
            cut_Ptheta1_FD = df_dvpi0p.Ptheta<FD_Ptheta_ub
            cut_Ptheta2_FD = df_dvpi0p.Ptheta>FD_Ptheta_lb
            cut_Gsector_FD = (df_Rec.Gsector<7) & (df_Rec.Gsector>0)
            cut_Gsector2_FD = (df_Rec.Gsector2<7) & (df_Rec.Gsector2>0)
            cut_GFid_FD = df_Rec.GFid==1
            cut_GFid2_FD = df_Rec.GFid2==1
            cut_PFid_FD = df_Rec.PFid==1

            cut_CDFT = (cut_Pp1_CDFT & cut_Psector_CDFT & cut_Ptheta1_CDFT & cut_Ptheta2_CDFT & cut_Gsector_CDFT & cut_Gsector2_CDFT & cut_GFid_CDFT & cut_GFid2_CDFT & cut_PFid_CDFT)
            cut_CD = (cut_Pp1_CD & cut_Psector_CD & cut_Ptheta1_CD & cut_Ptheta2_CD & cut_Gsector_CD & cut_Gsector2_CD & cut_GFid_CD & cut_GFid2_CD & cut_PFid_CD)
            cut_FD = (cut_Pp1_FD & cut_Psector_FD & cut_Ptheta1_FD & cut_Ptheta2_FD & cut_Gsector_FD & cut_Gsector2_FD & cut_GFid_FD & cut_GFid2_FD & cut_PFid_FD)

            df_Rec.loc[cut_CDFT, "config"] = 3
            df_Rec.loc[cut_CD, "config"] = 2
            df_Rec.loc[cut_FD, "config"] = 1

            df_Rec = df_Rec[df_Rec.config>0]

            df_Rec = df_Rec.sort_values(by=['closeness', 'Psector', 'Gsector'], ascending = [True, True, True])
            df_Rec = df_Rec.loc[~df_Rec.event.duplicated(), :]
            df_Rec = df_Rec.sort_values(by='event')

        elif (raw and dvcs):
            print("saving raw and dvcs overlap")
            df_Rec = self.df_epgg
            #common cuts
            cut_xBupper = df_Rec["xB"] < 1  # xB
            cut_xBlower = df_Rec["xB"] > 0  # xB
            cut_Q2 = df_Rec["Q2"] > 1  # Q2
            cut_W = df_Rec["W"] > 2  # W
            cut_Ee = df_Rec["Ee"] > 2  # Ee
            cut_Ge = df_Rec["Ge"] > 2  # Ge
            cut_Esector = 1#(df_Rec["Esector"]!=df_Rec["Gsector"]) & (df_Rec["Esector"]!=df_Rec["Gsector2"]) 
            cut_Psector = 1#~( ((df_Rec["Pstat"]//10)%10>0) & (df_Rec["Psector"]==df_Rec["Gsector"])) & ~( ((df_Rec["Pstat"]//10)%10>0) & df_Rec["Psector"]!=df_Rec["Gsector2"])
            cut_Ppmax = df_Rec.Pp < 1.6  # Pp
            cut_Pthetamin = df_Rec.Ptheta > 0  # Ptheta
            # cut_Vz = np.abs(df_Rec["Evz"] - df_Rec["Pvz"]) < 2.5 + 2.5 / mag([df_Rec["Ppx"], df_Rec["Ppy"], df_Rec["Ppz"]])
            cut_common = cut_xBupper & cut_xBlower & cut_Q2 & cut_W & cut_Ee & cut_Ge & cut_Esector & cut_Psector & cut_Ppmax & cut_Pthetamin

            df_Rec = df_Rec[cut_common]

            #CDFT
            cut_Pp1_CDFT = df_Rec.Pp > 0.3  # Pp
            cut_Psector_CDFT = df_Rec.Psector>7
            cut_Ptheta1_CDFT = df_dvpi0p.Ptheta<CD_Ptheta_ub
            cut_Ptheta2_CDFT = df_dvpi0p.Ptheta>CD_Ptheta_lb
            cut_Gsector_CDFT = df_Rec.Gsector>7
            cut_GFid_CDFT = df_Rec.GFid==1
            cut_PFid_CDFT = df_Rec.PFid==1
            if pol == "inbending":
                cut_mmep1_CDFT = df_Rec["MM2_ep"] < 0.6  # mmep
                cut_mmep2_CDFT = df_Rec["MM2_ep"] > -0.6  # mmep
            elif pol == "outbending":
                cut_mmep1_CDFT = df_Rec["MM2_ep"] < 0.6  # mmep
                cut_mmep2_CDFT = df_Rec["MM2_ep"] > -0.6  # mmep
            
            #CD
            cut_Pp1_CD = df_Rec.Pp > 0.3  # Pp
            cut_Psector_CD = df_Rec.Psector>7
            cut_Ptheta1_CD = df_dvpi0p.Ptheta<CD_Ptheta_ub
            cut_Ptheta2_CD = df_dvpi0p.Ptheta>CD_Ptheta_lb
            cut_Gsector_CD = (df_Rec.Gsector<7) & (df_Rec.Gsector>0)
            cut_GFid_CD = df_Rec.GFid==1
            cut_PFid_CD = df_Rec.PFid==1
            if pol == "inbending":
                cut_mmep1_CD = df_Rec["MM2_ep"] < 0.6  # mmep
                cut_mmep2_CD = df_Rec["MM2_ep"] > -0.6  # mmep
            elif pol == "outbending":
                cut_mmep1_CD = df_Rec["MM2_ep"] < 0.6  # mmep
                cut_mmep2_CD = df_Rec["MM2_ep"] > -0.6  # mmep

            #FD
            cut_Psector_FD = df_Rec.Psector<7
            cut_Ptheta1_FD = df_dvpi0p.Ptheta<FD_Ptheta_ub
            cut_Ptheta2_FD = df_dvpi0p.Ptheta>FD_Ptheta_lb
            cut_Gsector_FD = (df_Rec.Gsector<7) & (df_Rec.Gsector>0)
            cut_GFid_FD = df_Rec.GFid==1
            cut_PFid_FD = df_Rec.PFid==1
            if pol == "inbending":
                cut_Pp1_FD = df_Rec.Pp > 0.42  # Pp
                cut_mmep1_FD = df_Rec["MM2_ep"] < 0.6 # mmep
                cut_mmep2_FD = df_Rec["MM2_ep"] > -0.6  # mmep
            elif pol == "outbending":
                cut_Pp1_FD = df_Rec.Pp > 0.5  # Pp
                cut_mmep1_FD = df_Rec["MM2_ep"] < 0.6  # mmep
                cut_mmep2_FD = df_Rec["MM2_ep"] > -0.6  # mmep                

            cut_CDFT = (cut_Pp1_CDFT & cut_Psector_CDFT & cut_Ptheta1_CDFT & cut_Ptheta2_CDFT & cut_Gsector_CDFT & cut_GFid_CDFT & cut_PFid_CDFT & cut_mmep1_CDFT & cut_mmep2_CDFT)
            cut_CD = (cut_Pp1_CD & cut_Psector_CD & cut_Ptheta1_CD & cut_Ptheta2_CD & cut_Gsector_CD & cut_GFid_CD & cut_PFid_CD & cut_mmep1_CD & cut_mmep2_CD)
            cut_FD = (cut_Pp1_FD & cut_Psector_FD & cut_Ptheta1_FD & cut_Ptheta2_FD & cut_Gsector_FD & cut_GFid_FD & cut_PFid_FD & cut_mmep1_FD & cut_mmep2_FD)

            df_Rec.loc[cut_CDFT, "config"] = 3
            df_Rec.loc[cut_CD, "config"] = 2
            df_Rec.loc[cut_FD, "config"] = 1

            df_Rec = df_Rec[df_Rec.config>0]

            df_Rec = df_Rec.sort_values(by=['closeness', 'Psector', 'Gsector'], ascending = [True, True, True])
            df_Rec = df_Rec.loc[~df_Rec.event.duplicated(), :]
            df_Rec = df_Rec.sort_values(by='event')

        else:
            df_Rec = self.df_dvpi0p
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
    parser.add_argument("-w","--width", help="width of selection cuts", default = "default")
    parser.add_argument("-D","--dvcs", help="save dvcs overlap", action = "store_true")
    parser.add_argument("-sm","--smearing", help="save dvcs overlap", default = "1")
    parser.add_argument("-nf","--nofid", help="no additional fiducial cuts", action = "store_true")
    
    args = parser.parse_args()

    if args.entry_start:
        args.entry_start = int(args.entry_start)
    if args.entry_stop:
        args.entry_stop = int(args.entry_stop)
    smearingFactor = float(args.smearing)

    converter = root2pickle(args.fname, entry_start = args.entry_start, entry_stop = args.entry_stop, pol = args.polarity, gen = args.generator, raw = args.raw, detRes = args.detRes, width = args.width, dvcs = args.dvcs, smearing = smearingFactor, nofid = args.nofid)
    df = converter.df

    df.to_pickle(args.out)
