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
import sys


class root2pickle():
    #class to read root to make epg pairs, inherited from epg
    def __init__(self, fname, entry_stop = None, pol = "inbending"):
        self.fname = fname
        self.readEPGG(entry_stop, pol = pol)
        self.saveDVpi0vars()
        self.saveRaw()

    def readFile(self):
        #read root using uproot
        self.file = uproot.open(self.fname)
        self.tree = self.file["T"]

    def closeFile(self):
        #close file for saving memory
        self.file = None
        self.tree = None

    def readEPGG(self, entry_stop = None, pol = "inbending"):
        #save data into df_epg, df_epgg for parent class epg
        self.readFile()

        # data frames and their keys to read Z part
        df_electronGen = pd.DataFrame()
        df_protonGen = pd.DataFrame()
        df_gammaGen = pd.DataFrame()
        # eleKeysGen = ["GenEpx", "GenEpy", "GenEpz", "GenEvx", "GenEvy", "GenEvz"]
        eleKeysGen = ["GenEpx", "GenEpy", "GenEpz"]
        proKeysGen = ["GenPpx", "GenPpy", "GenPpz"]
        gamKeysGen = ["GenGpx", "GenGpy", "GenGpz"]
        # read keys
        for key in eleKeysGen:
            df_electronGen[key] = self.tree[key].array(library="pd", entry_stop=entry_stop)
        for key in proKeysGen:
            df_protonGen[key] = self.tree[key].array(library="pd", entry_stop=entry_stop)
        for key in gamKeysGen:
            df_gammaGen[key] = self.tree[key].array(library="pd", entry_stop=entry_stop)

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
        # df_electronGen = df_electronGen.loc[:, ["event", "GenEpx", "GenEpy", "GenEpz", "GenEvz"]]
        df_electronGen = df_electronGen.loc[:, ["event", "GenEpx", "GenEpy", "GenEpz"]]

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


        #spherical coordinates
        eleGen = [df_electronGen["GenEpx"], df_electronGen["GenEpy"], df_electronGen["GenEpz"]]
        df_electronGen.loc[:, 'GenEp'] = mag(eleGen)
        df_electronGen.loc[:, 'GenEtheta'] = getTheta(eleGen)
        df_electronGen.loc[:, 'GenEphi'] = getPhi(eleGen)

        proGen = [df_protonGen["GenPpx"], df_protonGen["GenPpy"], df_protonGen["GenPpz"]]
        df_protonGen.loc[:, 'GenPp'] = mag(proGen)
        df_protonGen.loc[:, 'GenPtheta'] = getTheta(proGen)
        df_protonGen.loc[:, 'GenPphi'] = getPhi(proGen)

        gamGen = [df_gammaGen["GenGpx"], df_gammaGen["GenGpy"], df_gammaGen["GenGpz"]]
        # df_gammaGen.loc[:, 'GenGp'] = mag(gamGen)
        df_gammaGen.loc[:, 'GenGtheta'] = getTheta(gamGen)
        df_gammaGen.loc[:, 'GenGphi'] = getPhi(gamGen)

        gamGen2 = [df_gammaGen["GenGpx2"], df_gammaGen["GenGpy2"], df_gammaGen["GenGpz2"]]
        debug = df_gammaGen.loc[:, 'GenGp2'] == mag(gamGen2)
        df_gammaGen.loc[:, 'GenGtheta2'] = getTheta(gamGen2)
        df_gammaGen.loc[:, 'GenGphi2'] = getPhi(gamGen2)

        df_z = pd.merge(df_electronGen, df_protonGen, how='inner', on='event')
        df_z = pd.merge(df_z, df_gammaGen, how='inner', on='event')

        #added from sangbaek: more variables

        ele = [df_z['GenEpx'], df_z['GenEpy'], df_z['GenEpz']]
        pro = [df_z['GenPpx'], df_z['GenPpy'], df_z['GenPpz']]
        
        gam = [df_z['GenGpx'], df_z['GenGpy'], df_z['GenGpz']]
        gam2 = [df_z['GenGpx2'], df_z['GenGpy2'], df_z['GenGpz2']]



        df_z.loc[:, 'GenEp'] = mag(ele)
        df_z.loc[:, 'GenEe'] = getEnergy(ele, me)
        df_z.loc[:, 'GenEtheta'] = getTheta(ele)
        df_z.loc[:, 'GenEphi'] = getPhi(ele)

        df_z.loc[:, 'GenPp'] = mag(pro)
        df_z.loc[:, 'GenPe'] = getEnergy(pro, M)
        df_z.loc[:, 'GenPtheta'] = getTheta(pro)
        df_z.loc[:, 'GenPphi'] = getPhi(pro)


        df_z.loc[:, 'GenGp'] = mag(gam)
        df_z.loc[:, 'GenGe'] = getEnergy(gam, 0)
        df_z.loc[:, 'GenGtheta'] = getTheta(gam)
        df_z.loc[:, 'GenGphi'] = getPhi(gam)

        df_z.loc[:, 'GenGp2'] = mag(gam2)
        df_z.loc[:,'GenGe2'] = getEnergy(gam2, 0)
        df_z.loc[:, 'GenGtheta2'] = getTheta(gam2)
        df_z.loc[:, 'GenGphi2'] = getPhi(gam2)

        pi0 = vecAdd(gam, gam2)

        Ppt = mag([df_z['GenPpx'], df_z['GenPpy'], 0])
        VGS = [-df_z['GenEpx'], -df_z['GenEpy'], pbeam - df_z['GenEpz']]
        v3l = cross(beam, ele)
        v3h = cross(pro, VGS)
        v3g = cross(VGS, gam)
        VmissG = [-df_z["GenEpx"] - df_z["GenPpx"], -df_z["GenEpy"] - df_z["GenPpy"],
                    pbeam - df_z["GenEpz"] - df_z["GenPpz"]]
        VmissP = [-(df_z["GenEpx"] + df_z["GenGpx"]), -(df_z["GenEpy"] + df_z["GenGpy"]),
                    -(-pbeam + df_z["GenEpz"] + df_z["GenGpz"])]
        Vmiss = [-(df_z["GenEpx"] + df_z["GenPpx"] + df_z["GenGpx"]), -(df_z["GenEpy"] + df_z["GenPpy"] + df_z["GenGpy"]),
                    -(-pbeam + df_z["GenEpz"] + df_z["GenPpz"] + df_z["GenGpz"])]
        costheta = cosTheta(VGS, gam)



        df_z.loc[:, 'GenMpx'], df_z.loc[:, 'GenMpy'], df_z.loc[:, 'GenMpz'] = Vmiss
        # binning kinematics
        df_z.loc[:,'GenQ2'] = -((ebeam - df_z['GenEe'])**2 - mag2(VGS))
        df_z.loc[:,'Gennu'] = (ebeam - df_z['GenEe'])
        df_z.loc[:,'Geny'] = df_z['Gennu']/ebeam
        df_z.loc[:,'GenxB'] = df_z['GenQ2'] / 2.0 / M / df_z['Gennu']
        df_z.loc[:,'Gent1'] = 2 * M * (df_z['GenPe'] - M)
        df_z.loc[:,'Gent2'] = (M * df_z['GenQ2'] + 2 * M * df_z['Gennu'] * (df_z['Gennu'] - np.sqrt(df_z['Gennu'] * df_z['Gennu'] + df_z['GenQ2']) * costheta))\
        / (M + df_z['Gennu'] - np.sqrt(df_z['Gennu'] * df_z['Gennu'] + df_z['GenQ2']) * costheta)
        df_z.loc[:,'GenW'] = np.sqrt(np.maximum(0, (ebeam + M - df_z['GenEe'])**2 - mag2(VGS)))
        # trento angles
        df_z.loc[:,'Genphi1'] = angle(v3l, v3h)
        df_z.loc[:,'Genphi1'] = np.where(dot(v3l, pro) > 0, 360.0 -
                                    df_z['Genphi1'], df_z['Genphi1'])
        df_z.loc[:,'Genphi2'] = angle(v3l, v3g)
        df_z.loc[:,'Genphi2'] = np.where(dot(v3l, gam) <
                                    0, 360.0 - df_z['Genphi2'], df_z['Genphi2'])
        # exclusivity variables
        df_z.loc[:,'GenMM2_epg'] = (-M - ebeam + df_z["GenEe"] +
                                df_z["GenPe"] + df_z["GenGe"])**2 - mag2(Vmiss)
        df_z.loc[:,'GenME_epg'] = (M + ebeam - df_z["GenEe"] - df_z["GenPe"] - df_z["GenGe"])
        df_z.loc[:,'GenMM2_ep'] = (-M - ebeam + df_z["GenEe"] + df_z["GenPe"])**2 - mag2(VmissG)
        df_z.loc[:,'GenMM2_eg'] = (-M - ebeam + df_z["GenEe"] + df_z["GenGe"])**2 - mag2(VmissP)
        df_z.loc[:,'GenMPt'] = np.sqrt((df_z["GenEpx"] + df_z["GenPpx"] + df_z["GenGpx"])**2 +
                                (df_z["GenEpy"] + df_z["GenPpy"] + df_z["GenGpy"])**2)
        df_z.loc[:,'GenconeAngle'] = angle(ele, gam)
        df_z.loc[:,'GenreconGam'] = angle(gam, VmissG)
        df_z.loc[:,'Gencoplanarity'] = angle(v3h, v3g)

        self.df_z = df_z    #done with saving z

        # data frames and their keys to read X part
        df_electronRec = pd.DataFrame()
        df_protonRec = pd.DataFrame()
        df_gammaRec = pd.DataFrame()
        # eleKeysRec = ["Epx", "Epy", "Epz", "Evx", "Evy", "Evz", "Esector"]
        # proKeysRec = ["Ppx", "Ppy", "Ppz", "Pvz", "Psector"]
        eleKeysRec = ["Epx", "Epy", "Epz", "Evz", "Esector"]
        proKeysRec = ["Ppx", "Ppy", "Ppz", "Pvz", "Psector"]
        gamKeysRec = ["Gpx", "Gpy", "Gpz", "Gsector"]
        # read them
        for key in eleKeysRec:
            df_electronRec[key] = self.tree[key].array(library="pd", entry_stop=entry_stop)
        for key in proKeysRec:
            df_protonRec[key] = self.tree[key].array(library="pd", entry_stop=entry_stop)
        for key in gamKeysRec:
            df_gammaRec[key] = self.tree[key].array(library="pd", entry_stop=entry_stop)

        self.closeFile()

        #convert data type to standard double
        # df_electronRec = df_electronRec.astype({"Epx": float, "Epy": float, "Epz": float, "Evx": float, "Evy": float, "Evz": float})
        # df_protonRec = df_protonRec.astype({"Ppx": float, "Ppy": float, "Ppz": float, "Pvz": float})
        df_electronRec = df_electronRec.astype({"Epx": float, "Epy": float, "Epz": float, "Evz": float})
        df_protonRec = df_protonRec.astype({"Ppx": float, "Ppy": float, "Ppz": float, "Pvz": float})
        df_gammaRec = df_gammaRec.astype({"Gpx": float, "Gpy": float, "Gpz": float})

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

        df_protonRec.loc[:, "Ppx"] = df_protonRec.loc[:, "Pp"]*np.sin(np.radians(df_protonRec.loc[:, "Ptheta"]))*np.cos(np.radians(df_protonRec.loc[:, "Pphi"]))
        df_protonRec.loc[:, "Ppy"] = df_protonRec.loc[:, "Pp"]*np.sin(np.radians(df_protonRec.loc[:, "Ptheta"]))*np.sin(np.radians(df_protonRec.loc[:, "Pphi"]))
        df_protonRec.loc[:, "Ppz"] = df_protonRec.loc[:, "Pp"]*np.cos(np.radians(df_protonRec.loc[:, "Ptheta"]))
        pro = [df_protonRec['Ppx'], df_protonRec['Ppy'], df_protonRec['Ppz']]

        df_protonRec.loc[:, 'Pe'] = getEnergy(pro, M)

        # df_gammaRec = df_gammaRec[df_gammaRec["Gsector"]<7]

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
        df_epgg.loc[:,'phi2'] = np.where(dot(VGS, cross(v3l, v3g)) <
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
        self.df_epgg = df_epgg

    
    def saveRaw(self):
        df_x = self.df_epgg
        df_z = self.df_z
        df = pd.merge(df_x, df_z, how = 'inner', on='event')
        self.df = df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get args",formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-f","--fname", help="a single root file to convert into pickles", default="/Users/sangbaek/Dropbox (MIT)/data/project/merged_9628_files.root")
    parser.add_argument("-o","--out", help="a single pickle file name as an output", default="goodbyeRoot.pkl")
    parser.add_argument("-s","--entry_stop", help="entry_stop to stop reading the root file", default = None)
    parser.add_argument("-p","--polarity", help="polarity", default = "inbending")
    
    args = parser.parse_args()

    converter = root2pickle(args.fname, entry_stop = args.entry_stop, pol = args.polarity)
    df = converter.df

    df.to_pickle(args.out)