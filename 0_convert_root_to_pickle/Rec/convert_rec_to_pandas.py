#gen

import uproot
import pandas as pd
import numpy as np
import argparse
from copy import copy
from utils import const, physics, filestruct
import os,sys

fs = filestruct.fs()

PhysicsConstants = const.PhysicsConstants()


def convert_rec_to_pandas(args):
    #(self, fname, entry_start = None, entry_stop = None, pol = "inbending", gen = "norad", raw = False, detRes = False, width = "mid", dvcs = False, smearing = 1, nofid = False):
    #    def readEPGG(self, entry_start = None, entry_stop = None, gen = "pi0norad", pol = "inbending", detRes = False, smearing = 1, nofid = False):
    detRes =0
    nofid = args.nofid
    smearing = args.smearing
    pol = args.polarity

    print("PROCESSING WITH OPTIONS- nofid: {} smearing: {} pol: {} correction: {}".format(nofid, smearing, pol,args.correction))

    rec_file = uproot.open(args.fname)
    rec_tree = rec_file["T"]

    if args.entry_stop is None:
        args.entry_stop = rec_tree.num_entries
    
    # if entry stop less than chunk size, set chunk size to entry stop
    if args.entry_stop < args.chunk_size:
        args.chunk_size = args.entry_stop

    starting_entry = args.entry_start
    iteration = 0

    #Make an output directory that has the same location as the input file, and is named the same as the input file.
    filename_base = args.fname.split("/")[-1].split(".")[0]
    output_dir = "/".join(args.fname.split("/")[:-1])+"/"
    print('Output to directory: '+output_dir+filename_base)

    if not os.path.exists(output_dir+filename_base):
        os.makedirs(output_dir+filename_base)

    while starting_entry < args.entry_stop:
        print("Creating DF with starting entry: {}".format(starting_entry))

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
            df_electronGen[key] = rec_tree[key].array(library="pd", entry_start = starting_entry, entry_stop=starting_entry+args.chunk_size)
        for key in proKeysGen:
            df_protonGen[key] = rec_tree[key].array(library="pd", entry_start = starting_entry, entry_stop=starting_entry+args.chunk_size)
        for key in gamKeysGen:
            df_gammaGen[key] = rec_tree[key].array(library="pd", entry_start = starting_entry, entry_stop=starting_entry+args.chunk_size)

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
        df_electronGen.loc[:, 'GenEp'] = physics.mag(eleGen)
        df_electronGen.loc[:, 'GenEtheta'] = physics.getTheta(eleGen)
        df_electronGen.loc[:, 'GenEphi'] = physics.getPhi(eleGen)

        proGen = [df_protonGen["GenPpx"], df_protonGen["GenPpy"], df_protonGen["GenPpz"]]
        df_protonGen.loc[:, 'GenPp'] = physics.mag(proGen)
        df_protonGen.loc[:, 'GenPtheta'] = physics.getTheta(proGen)
        df_protonGen.loc[:, 'GenPphi'] = physics.getPhi(proGen)


        df_MC = pd.merge(df_electronGen, df_protonGen, how='inner', on='event')

        if args.rad:
            df_pi0Gen = pd.DataFrame()
            for key in pi0KeysGen:
                df_pi0Gen[key] = rec_tree[key].array(library="pd", entry_start = starting_entry, entry_stop=starting_entry+args.chunk_size)
            df_pi0Gen = df_pi0Gen.astype({"GenPipx": float, "GenPipy": float, "GenPipz": float})
            df_pi0Gen.loc[:,'event'] = df_pi0Gen.index
            #two g's to one gg.
            pi0Gen = [df_pi0Gen["GenPipx"], df_pi0Gen["GenPipy"], df_pi0Gen["GenPipz"]]
            df_pi0Gen.loc[:, 'GenPip'] = physics.mag(pi0Gen)
            df_pi0Gen.loc[:, 'GenPitheta'] = physics.getTheta(pi0Gen)
            df_pi0Gen.loc[:, 'GenPiphi'] = physics.getPhi(pi0Gen)

            df_gammaGen = df_gammaGen[df_gammaGen.index.get_level_values('subentry')==0]
            gamGen = [df_gammaGen["GenGpx"], df_gammaGen["GenGpy"], df_gammaGen["GenGpz"]]
            df_gammaGen.loc[:, 'GenGp'] = physics.mag(gamGen)
            df_gammaGen.loc[:, 'GenGtheta'] = physics.getTheta(gamGen)
            df_gammaGen.loc[:, 'GenGphi'] = physics.getPhi(gamGen)

            df_MC = pd.merge(df_MC, df_gammaGen, how='inner', on='event')
            df = pd.merge(df_MC, df_pi0Gen, how='inner', on='event')

        else:
            #two g's to one gg.
            gamGen = [df_gammaGen["GenGpx"], df_gammaGen["GenGpy"], df_gammaGen["GenGpz"]]
            df_gammaGen.loc[:, 'GenGp'] = physics.mag(gamGen)

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
            df_gammaGen.loc[:, 'GenGtheta'] = physics.getTheta(gamGen)
            df_gammaGen.loc[:, 'GenGphi'] = physics.getPhi(gamGen)

            gamGen2 = [df_gammaGen["GenGpx2"], df_gammaGen["GenGpy2"], df_gammaGen["GenGpz2"]]
            df_gammaGen.loc[:, 'GenGtheta2'] = physics.getTheta(gamGen2)
            df_gammaGen.loc[:, 'GenGphi2'] = physics.getPhi(gamGen2)

            df = pd.merge(df_MC, df_gammaGen, how='inner', on='event')

        # Calculate kinematics
        ele = [df['GenEpx'], df['GenEpy'], df['GenEpz']]
        df.loc[:, 'GenEp'] = physics.mag(ele)
        df.loc[:, 'GenEe'] = physics.getEnergy(ele, PhysicsConstants.electron_mass)
        df.loc[:, 'GenEtheta'] = physics.getTheta(ele)
        df.loc[:, 'GenEphi'] = physics.getPhi(ele)

        pro = [df['GenPpx'], df['GenPpy'], df['GenPpz']]
        df.loc[:, 'GenPp'] = physics.mag(pro)
        df.loc[:, 'GenPe'] = physics.getEnergy(pro, PhysicsConstants.proton_mass)
        df.loc[:, 'GenPtheta'] = physics.getTheta(pro)
        df.loc[:, 'GenPphi'] = physics.getPhi(pro)

        if args.rad:
            gam = [df['GenGpx'], df['GenGpy'], df['GenGpz']]
            df.loc[:, 'GenGp'] = physics.mag(gam)
            df.loc[:, 'GenGe'] = physics.getEnergy(gam, 0)
            df.loc[:, 'GenGtheta'] = physics.getTheta(gam)
            df.loc[:, 'GenGphi'] = physics.getPhi(gam)

            pi0 = [df['GenPipx'], df['GenPipy'], df['GenPipz']]
            df.loc[:, 'GenPip'] = physics.mag(pi0)
            df.loc[:, 'GenPie'] = physics.getEnergy(pi0, 0)
            df.loc[:, 'GenPitheta'] = physics.getTheta(pi0)
            df.loc[:, 'GenPiphi'] = physics.getPhi(pi0)

        else:
            gam = [df['GenGpx'], df['GenGpy'], df['GenGpz']]
            df.loc[:, 'GenGp'] = physics.mag(gam)
            df.loc[:, 'GenGe'] = physics.getEnergy(gam, 0)
            df.loc[:, 'GenGtheta'] = physics.getTheta(gam)
            df.loc[:, 'GenGphi'] = physics.getPhi(gam)
            
            gam2 = [df['GenGpx2'], df['GenGpy2'], df['GenGpz2']]
            df.loc[:, 'GenGp2'] = physics.mag(gam2)
            df.loc[:,'GenGe2'] = physics.getEnergy(gam2, 0)
            df.loc[:, 'GenGtheta2'] = physics.getTheta(gam2)
            df.loc[:, 'GenGphi2'] = physics.getPhi(gam2)

            pi0 = physics.vecAdd(gam, gam2)
            #add the components of pi0 to the dataframe
            df.loc[:, 'GenPipx'] = pi0[0]
            df.loc[:, 'GenPipy'] = pi0[1]
            df.loc[:, 'GenPipz'] = pi0[2]
            df.loc[:, 'GenPip'] = physics.mag(pi0)
            df.loc[:, 'GenPie'] = physics.getEnergy(pi0, 0)
            df.loc[:, 'GenPitheta'] = physics.getTheta(pi0)
            df.loc[:, 'GenPiphi'] = physics.getPhi(pi0)


        VGS = [-df['GenEpx'], -df['GenEpy'], PhysicsConstants.electron_beam_momentum_magnitude - df['GenEpz']]
        v3l = physics.cross(PhysicsConstants.electron_beam_3_vector, ele)
        v3h = physics.cross(pro, VGS)
        v3g = physics.cross(VGS, gam)
        v3pi0 = physics.cross(VGS, pi0)

        VmissEP = [-df["GenEpx"] - df["GenPpx"], -df["GenEpy"] - df["GenPpy"],
                PhysicsConstants.electron_beam_momentum_magnitude - df["GenEpz"] - df["GenPpz"]]


        VmissEGG = [-(df["GenEpx"] + df["GenPipx"]), -(df["GenEpy"] +  df["GenPipy"]),
                    -(-PhysicsConstants.electron_beam_momentum_magnitude + df["GenEpz"] + df["GenPipz"])]
        
        
        VmissEPGG = [-(df["GenEpx"] + df["GenPpx"] + df["GenPipx"]), -(df["GenEpy"] + df["GenPpy"] + df["GenPipy"]),
                    -(-PhysicsConstants.electron_beam_momentum_magnitude + df["GenEpz"] + df["GenPpz"] + df["GenPipz"])]
        
        costheta = physics.cosTheta(VGS, pi0)
        
        df.loc[:, 'GenMpx'], df.loc[:, 'GenMpy'], df.loc[:, 'GenMpz'] = VmissEPGG

        # binning kinematics
        df.loc[:,'GenQ2'] = -((PhysicsConstants.electron_beam_energy - df['GenEe'])**2 - physics.mag2(VGS))
        df.loc[:,'Gennu'] = (PhysicsConstants.electron_beam_energy - df['GenEe'])
        df.loc[:,'Geny'] = df['Gennu']/PhysicsConstants.electron_beam_energy
        df.loc[:,'GenxB'] = df['GenQ2'] / 2.0 / PhysicsConstants.proton_mass / df['Gennu']
        df.loc[:,'Gent1'] = 2 * PhysicsConstants.proton_mass * (df['GenPe'] - PhysicsConstants.proton_mass)
        df.loc[:,'Gent2'] = (PhysicsConstants.proton_mass * df['GenQ2'] + 2 * PhysicsConstants.proton_mass * df['Gennu'] * (df['Gennu'] - np.sqrt(df['Gennu'] * df['Gennu'] + df['GenQ2']) * costheta))\
        / (PhysicsConstants.proton_mass + df['Gennu'] - np.sqrt(df['Gennu'] * df['Gennu'] + df['GenQ2']) * costheta)
        
        df.loc[:,'GenW'] = np.sqrt(np.maximum(0, (PhysicsConstants.electron_beam_energy + PhysicsConstants.proton_mass - df['GenEe'])**2 - physics.mag2(VGS)))
    
        # trento angles
        df.loc[:,'Genphi1'] = physics.angle(v3l, v3h)
        df.loc[:,'Genphi1'] = np.where(physics.dot(v3l, pro) > 0, 360.0 -
                                    df['Genphi1'], df['Genphi1'])
        df.loc[:,'Genphi2'] = physics.angle(v3l, v3pi0)
        df.loc[:,'Genphi2'] = np.where(physics.dot(v3l, pi0) <
                                    0, 360.0 - df['Genphi2'], df['Genphi2'])
        
        # exclusivity variables

        df.loc[:,'GenMPt'] = np.sqrt((df["GenEpx"] + df["GenPpx"] + df["GenPipx"])**2 +
                                (df["GenEpy"] + df["GenPpy"] + df["GenPipy"])**2)
        
        df.loc[:,'GenMM2_epgg'] = (-PhysicsConstants.proton_mass - PhysicsConstants.electron_beam_energy + df["GenEe"] +
                        df["GenPe"] + df["GenPie"])**2 - physics.mag2(VmissEPGG)
        
        df.loc[:,'GenMM2_ep'] = (-PhysicsConstants.proton_mass - PhysicsConstants.electron_beam_energy + df["GenEe"] + 
                                 df["GenPe"])**2 - physics.mag2(VmissEP)

        df.loc[:,'GenMM2_egg'] = (-PhysicsConstants.proton_mass-PhysicsConstants.electron_beam_energy + df["GenEe"] + 
                                  df["GenPie"])**2 - physics.mag2(VmissEGG)
        
        df.loc[:,'GenME_epgg'] = (PhysicsConstants.proton_mass + PhysicsConstants.electron_beam_energy - 
                                  df["GenEe"] - df["GenPe"] - df["GenPie"])


        df.loc[:,'GenconeAngle'] = physics.angle(ele, pi0)
        df.loc[:,'GenreconGam'] = physics.angle(gam, VmissEP)

        df.loc[:,'Gencoplanarity'] = physics.angle(v3h, v3pi0)
        df.loc[:,'GenconeAngle1'] = physics.angle(ele, gam) # higher energy photon for aao_norad, radiated photon for aao_rad
        
        if not args.rad:
            df.loc[:,'GenconeAngle2'] = physics.angle(ele, gam2)
            df.loc[:,'GenopeningAngle'] = physics.angle(gam, gam2)
        
        df_MC = df

        # DONE WITH MC BANKS


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
            df_electronRec[key] = rec_tree[key].array(library="pd", entry_start = starting_entry, entry_stop=starting_entry+args.chunk_size)
        for key in proKeysRec:
            df_protonRec[key] = rec_tree[key].array(library="pd", entry_start = starting_entry, entry_stop=starting_entry+args.chunk_size)
        for key in gamKeysRec:
            df_gammaRec[key] = rec_tree[key].array(library="pd", entry_start = starting_entry, entry_stop=starting_entry+args.chunk_size)

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

        #prepare for proton energy loss correction correction
        pro = [df_protonRec['Ppx'], df_protonRec['Ppy'], df_protonRec['Ppz']]
        df_protonRec.loc[:, 'Pp'] = physics.mag(pro)
        df_protonRec.loc[:, 'Ptheta'] = physics.getTheta(pro)
        df_protonRec.loc[:, 'Pphi'] = physics.getPhi(pro)

        df_protonRec.loc[:, "PDc1theta"] = -100000


        if detRes:
            df_protonRec.loc[:, "PDc3theta"] = -100000

            df_electronRec.loc[:, "EDc1theta"] = physics.getTheta([df_electronRec.EDc1Hitx, df_electronRec.EDc1Hity, df_electronRec.EDc1Hitz])
            df_electronRec.loc[:, "EDc3theta"] = physics.getTheta([df_electronRec.EDc3Hitx, df_electronRec.EDc3Hity, df_electronRec.EDc3Hitz])
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

        #two band criterion
        def corr(x, t):
            x0, x1, x2, x3 = x
            return x0 + x1*np.power(t-np.ones(len(t))*0.3, x3)

        df_protonRecFD = df_protonRecFD.loc[df_protonRec.Pp > 0.3, :]
        df_protonRecFD.loc[:, "PDc1theta"] = physics.getTheta([df_protonRecFD.PDc1Hitx, df_protonRecFD.PDc1Hity, df_protonRecFD.PDc1Hitz])
        if detRes:
            df_protonRecFD.loc[:, "PDc3theta"] = physics.getTheta([df_protonRecFD.PDc3Hitx, df_protonRecFD.PDc3Hity, df_protonRecFD.PDc3Hitz])
        best_params = [-53.14680163254601, 79.61307254040804, 0.3, 0.05739232362022314]
        df_protonRecFD_1 = df_protonRecFD.loc[df_protonRecFD.PDc1theta < corr(best_params, df_protonRecFD.Pp), :]
        df_protonRecFD_2 = df_protonRecFD.loc[df_protonRecFD.PDc1theta >= corr(best_params, df_protonRecFD.Pp), :]

        if detRes:
            df_protonRecCD.loc[:, "PCvt1r"] = physics.mag([df_protonRecCD.PCvt1Hitx, df_protonRecCD.PCvt1Hity, df_protonRecCD.PCvt1Hitz])
            df_protonRecCD.loc[:, "PCvt1theta"] = physics.getTheta([df_protonRecCD.PCvt1Hitx, df_protonRecCD.PCvt1Hity, df_protonRecCD.PCvt1Hitz])
            df_protonRecCD.loc[:, "PCvt1phi"] = physics.getPhi([df_protonRecCD.PCvt1Hitx, df_protonRecCD.PCvt1Hity, df_protonRecCD.PCvt1Hitz])
            df_protonRecCD.loc[:, "PCvt3r"] = physics.mag([df_protonRecCD.PCvt3Hitx, df_protonRecCD.PCvt3Hity, df_protonRecCD.PCvt3Hitz])
            df_protonRecCD.loc[:, "PCvt3theta"] = physics.getTheta([df_protonRecCD.PCvt3Hitx, df_protonRecCD.PCvt3Hity, df_protonRecCD.PCvt3Hitz])
            df_protonRecCD.loc[:, "PCvt3phi"] = physics.getPhi([df_protonRecCD.PCvt3Hitx, df_protonRecCD.PCvt3Hity, df_protonRecCD.PCvt3Hitz])
            df_protonRecCD.loc[:, "PCvt5r"] = physics.mag([df_protonRecCD.PCvt5Hitx, df_protonRecCD.PCvt5Hity, df_protonRecCD.PCvt5Hitz])
            df_protonRecCD.loc[:, "PCvt5theta"] = physics.getTheta([df_protonRecCD.PCvt5Hitx, df_protonRecCD.PCvt5Hity, df_protonRecCD.PCvt5Hitz])
            df_protonRecCD.loc[:, "PCvt5phi"] = physics.getPhi([df_protonRecCD.PCvt5Hitx, df_protonRecCD.PCvt5Hity, df_protonRecCD.PCvt5Hitz])
            df_protonRecCD.loc[:, "PCvt7r"] = physics.mag([df_protonRecCD.PCvt7Hitx, df_protonRecCD.PCvt7Hity, df_protonRecCD.PCvt7Hitz])
            df_protonRecCD.loc[:, "PCvt7theta"] = physics.getTheta([df_protonRecCD.PCvt7Hitx, df_protonRecCD.PCvt7Hity, df_protonRecCD.PCvt7Hitz])
            df_protonRecCD.loc[:, "PCvt7phi"] = physics.getPhi([df_protonRecCD.PCvt7Hitx, df_protonRecCD.PCvt7Hity, df_protonRecCD.PCvt7Hitz])
            df_protonRecCD.loc[:, "PCvt12r"] = physics.mag([df_protonRecCD.PCvt12Hitx, df_protonRecCD.PCvt12Hity, df_protonRecCD.PCvt12Hitz])
            df_protonRecCD.loc[:, "PCvt12theta"] = physics.getTheta([df_protonRecCD.PCvt12Hitx, df_protonRecCD.PCvt12Hity, df_protonRecCD.PCvt12Hitz])
            df_protonRecCD.loc[:, "PCvt12phi"] = physics.getPhi([df_protonRecCD.PCvt12Hitx, df_protonRecCD.PCvt12Hity, df_protonRecCD.PCvt12Hitz])
        else:
            df_protonRecCD.loc[:, "PCvt12theta"] = physics.getTheta([df_protonRecCD.PCvt12Hitx, df_protonRecCD.PCvt12Hity, df_protonRecCD.PCvt12Hitz])
            df_protonRecCD.loc[:, "PCvt12phi"] = physics.getPhi([df_protonRecCD.PCvt12Hitx, df_protonRecCD.PCvt12Hity, df_protonRecCD.PCvt12Hitz])


        #inbending proton energy loss correction
        if args.correction:
            if args.polarity == "inbending":
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
            elif args.polarity == "outbending":
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
            correction = False
        

        gam = [df_gammaRec['Gpx'], df_gammaRec['Gpy'], df_gammaRec['Gpz']]
        df_gammaRec.loc[:, 'Gp'] = physics.mag(gam)
        df_gammaRec.loc[:, 'Gtheta'] = physics.getTheta(gam)
        df_gammaRec.loc[:, 'Gphi'] = physics.getPhi(gam)
        
        if correction:
            print("energy loss correction applied for " + args.polarity)

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
            df_protonRec.loc[:, 'Pe'] = physics.getEnergy(pro, PhysicsConstants.proton_mass)

            #smearing photon

            #FT photon
            df_gammaRec.loc[df_gammaRec["Gsector"]>7, "Gp"] = df_gammaRec.loc[df_gammaRec["Gsector"]>7, "Gp"]*np.random.normal(1, smearing*(0.013 + 0.003/(1+np.exp(0.761*(df_gammaRec.loc[df_gammaRec["Gsector"]>7, "Gp"]-6)))), len(df_gammaRec.loc[df_gammaRec.Gsector>7]))
            #FD photon
            df_gammaRec.loc[df_gammaRec["Gsector"]<7, "Gp"] = df_gammaRec.loc[df_gammaRec["Gsector"]<7, "Gp"]*np.random.normal(1, smearing*(0.0395/(1+np.exp(5.308*(df_gammaRec.loc[df_gammaRec["Gsector"]<7, "Gp"]- 8.005)))), len(df_gammaRec.loc[df_gammaRec.Gsector<7]))

            df_gammaRec.loc[:, "Gpx"] = df_gammaRec.loc[:, "Gp"]*np.sin(np.radians(df_gammaRec.loc[:, "Gtheta"]))*np.cos(np.radians(df_gammaRec.loc[:, "Gphi"]))
            df_gammaRec.loc[:, "Gpy"] = df_gammaRec.loc[:, "Gp"]*np.sin(np.radians(df_gammaRec.loc[:, "Gtheta"]))*np.sin(np.radians(df_gammaRec.loc[:, "Gphi"]))
            df_gammaRec.loc[:, "Gpz"] = df_gammaRec.loc[:, "Gp"]*np.cos(np.radians(df_gammaRec.loc[:, "Gtheta"]))

        
        ele = [df_electronRec['Epx'], df_electronRec['Epy'], df_electronRec['Epz']]
        df_electronRec.loc[:, 'Ep'] = physics.mag(ele)
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

        # useful objects
        ele = [df_epgg['Epx'], df_epgg['Epy'], df_epgg['Epz']]
        df_epgg.loc[:, 'Ep'] = physics.mag(ele)
        df_epgg.loc[:, 'Ee'] = physics.getEnergy(ele, PhysicsConstants.electron_mass)
        df_epgg.loc[:, 'Etheta'] = physics.getTheta(ele)
        df_epgg.loc[:, 'Ephi'] = physics.getPhi(ele)

        pro = [df_epgg['Ppx'], df_epgg['Ppy'], df_epgg['Ppz']]

        if not args.correction:
            df_epgg.loc[:, 'Pp'] = physics.mag(pro)
            df_epgg.loc[:, 'Pe'] = physics.getEnergy(pro, PhysicsConstants.proton_mass)
            df_epgg.loc[:, 'Ptheta'] = physics.getTheta(pro)
            df_epgg.loc[:, 'Pphi'] = physics.getPhi(pro)


        gam = [df_epgg['Gpx'], df_epgg['Gpy'], df_epgg['Gpz']]
        df_epgg.loc[:, 'Gp'] = physics.mag(gam)
        df_epgg.loc[:, 'Ge'] = physics.getEnergy(gam, 0)
        df_epgg.loc[:, 'Gtheta'] = physics.getTheta(gam)
        df_epgg.loc[:, 'Gphi'] = physics.getPhi(gam)

        gam2 = [df_epgg['Gpx2'], df_epgg['Gpy2'], df_epgg['Gpz2']]
        df_epgg.loc[:, 'Gp2'] = physics.mag(gam2)
        df_epgg.loc[:,'Ge2'] = physics.getEnergy(gam2, 0)
        df_epgg.loc[:, 'Gtheta2'] = physics.getTheta(gam2)
        df_epgg.loc[:, 'Gphi2'] = physics.getPhi(gam2)

        pi0 = physics.vecAdd(gam, gam2)
        VGS = [-df_epgg['Epx'], -df_epgg['Epy'], PhysicsConstants.electron_beam_momentum_magnitude - df_epgg['Epz']]
        v3l = physics.cross(PhysicsConstants.electron_beam_3_vector, ele)
        v3h = physics.cross(pro, VGS)
        v3g = physics.cross(VGS, gam)
        v3pi0 = physics.cross(VGS, pi0)

        VmissPi0 = [-df_epgg["Epx"] - df_epgg["Ppx"], -df_epgg["Epy"] -
                    df_epgg["Ppy"], PhysicsConstants.electron_beam_momentum_magnitude - df_epgg["Epz"] - df_epgg["Ppz"]]
        VmissP = [-df_epgg["Epx"] - df_epgg["Gpx"] - df_epgg["Gpx2"], -df_epgg["Epy"] -
                    df_epgg["Gpy"] - df_epgg["Gpy2"], PhysicsConstants.electron_beam_momentum_magnitude - df_epgg["Epz"] - df_epgg["Gpz"] - df_epgg["Gpz2"]]
        Vmiss = [-df_epgg["Epx"] - df_epgg["Ppx"] - df_epgg["Gpx"] - df_epgg["Gpx2"],
                    -df_epgg["Epy"] - df_epgg["Ppy"] - df_epgg["Gpy"] - df_epgg["Gpy2"],
                    PhysicsConstants.electron_beam_momentum_magnitude - df_epgg["Epz"] - df_epgg["Ppz"] - df_epgg["Gpz"] - df_epgg["Gpz2"]]
        costheta = physics.cosTheta(VGS, pi0)

        df_epgg.loc[:, 'Mpx'], df_epgg.loc[:, 'Mpy'], df_epgg.loc[:, 'Mpz'] = Vmiss

        # binning kinematics
        df_epgg.loc[:,'Q2'] = -((PhysicsConstants.electron_beam_energy - df_epgg['Ee'])**2 - physics.mag2(VGS))
        df_epgg.loc[:,'nu'] = (PhysicsConstants.electron_beam_energy - df_epgg['Ee'])
        df_epgg.loc[:,'xB'] = df_epgg['Q2'] / 2.0 / PhysicsConstants.proton_mass / df_epgg['nu']
        df_epgg.loc[:,'y'] = df_epgg['nu']/PhysicsConstants.electron_beam_energy
        df_epgg.loc[:,'t1'] = 2 * PhysicsConstants.proton_mass * (df_epgg['Pe'] - PhysicsConstants.proton_mass)
        df_epgg.loc[:,'t2'] = (PhysicsConstants.proton_mass * df_epgg['Q2'] + 2 * PhysicsConstants.proton_mass * df_epgg['nu'] * (df_epgg['nu'] - np.sqrt(df_epgg['nu'] * df_epgg['nu'] + df_epgg['Q2']) * costheta))\
        / (PhysicsConstants.proton_mass + df_epgg['nu'] - np.sqrt(df_epgg['nu'] * df_epgg['nu'] + df_epgg['Q2']) * costheta)
        df_epgg.loc[:,'W'] = np.sqrt(np.maximum(0, (PhysicsConstants.electron_beam_energy + PhysicsConstants.proton_mass - df_epgg['Ee'])**2 - physics.mag2(VGS)))
        df_epgg.loc[:,'MPt'] = np.sqrt((df_epgg["Epx"] + df_epgg["Ppx"] + df_epgg["Gpx"] + df_epgg["Gpx2"])**2 +
                                (df_epgg["Epy"] + df_epgg["Ppy"] + df_epgg["Gpy"] + df_epgg["Gpy2"])**2)
        # trento angles
        df_epgg.loc[:,'phi1'] = physics.angle(v3l, v3h)
        df_epgg.loc[:,'phi1'] = np.where(physics.dot(v3l, pro) > 0, 360.0 -
                                df_epgg['phi1'], df_epgg['phi1'])
        df_epgg.loc[:,'phi2'] = physics.angle(v3l, v3g)
        df_epgg.loc[:,'phi2'] = np.where(physics.dot(v3l, pi0) <
                                0, 360.0 - df_epgg['phi2'], df_epgg['phi2'])

        # exclusivity variables
        df_epgg.loc[:,'MM2_ep'] = (-PhysicsConstants.proton_mass - PhysicsConstants.electron_beam_energy + df_epgg["Ee"] +
                            df_epgg["Pe"])**2 - physics.mag2(VmissPi0)
        df_epgg.loc[:,'MM2_egg'] = (-PhysicsConstants.proton_mass - PhysicsConstants.electron_beam_energy + df_epgg["Ee"] +
                            df_epgg["Ge"] + df_epgg["Ge2"])**2 - physics.mag2(VmissP)
        df_epgg.loc[:,'MM2_epgg'] = (-PhysicsConstants.proton_mass - PhysicsConstants.electron_beam_energy + df_epgg["Ee"] + df_epgg["Pe"] +
                            df_epgg["Ge"] + df_epgg["Ge2"])**2 - physics.mag2(Vmiss)
        df_epgg.loc[:,'ME_epgg'] = (PhysicsConstants.proton_mass + PhysicsConstants.electron_beam_energy - df_epgg["Ee"] - df_epgg["Pe"] - df_epgg["Ge"] - df_epgg["Ge2"])
        df_epgg.loc[:,'Mpi0'] = physics.pi0InvMass(gam, gam2)
        df_epgg.loc[:,'reconPi'] = physics.angle(VmissPi0, pi0)
        df_epgg.loc[:,"Pie"] = df_epgg['Ge'] + df_epgg['Ge2']
        df_epgg.loc[:,'coplanarity'] = physics.angle(v3h, v3pi0)
        df_epgg.loc[:,'coneAngle1'] = physics.angle(ele, gam)
        df_epgg.loc[:,'coneAngle2'] = physics.angle(ele, gam2)
        df_epgg.loc[:,'openingAngle'] = physics.angle(gam, gam2)

        df_epgg.loc[:, "closeness"] = np.abs(df_epgg.loc[:, "Mpi0"] - PhysicsConstants.neutral_pion_mass)

        df_out = pd.merge(df_epgg, df_MC, how = 'inner', on='event')
    
        return df_out





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get args",formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-f","--fname", help="a single root file to convert into pickles", default="infile.root")
    parser.add_argument("-t","--test", help="use to enable testing flag", action='store_true',default=False)
    parser.add_argument("-n","--correction", help="apply momentum correction", action='store_true',default=True)
    parser.add_argument("-r","--rad", help="use radiatve generator, otherwise use norad generator", action='store_true',default=False)
    parser.add_argument("-o","--out", help="a single pickle file name as an output", default="outfile.pkl")
    parser.add_argument("-c","--chunk_size", type=int, metavar='N', help="block size of each pandas file", default = 10_000_000)
    parser.add_argument("-s","--entry_start", type=int, metavar='N', help="entry_start to start reading the root file", default = 0)
    parser.add_argument("-p","--entry_stop", type=int, metavar='N',help="entry_stop to stop reading the root file", default = None)
    parser.add_argument("-l","--polarity", help="polarity", default = "inbending")
    parser.add_argument("-g","--generator", help="choose dvcs or pi0", default = "pi0norad")
    parser.add_argument("-w","--raw", help="save raw only", default = False, action = "store_true")
    parser.add_argument("-d","--detRes", help="include detector response", action = "store_true")
    parser.add_argument("-i","--width", help="width of selection cuts", default = "default")
    parser.add_argument("-sm","--smearing", help="change smearing factor", default = 1)
    parser.add_argument("-nf","--nofid", help="no additional fiducial cuts", default = "0")
    
    args = parser.parse_args()

    input_dir = fs.inb_norad_rec_root_dir
    output_dir = fs.inb_norad_rec_epgg_dir

    args.polarity = "inbending"

    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    for file in files:
        fname_base = file.split(".")[0]
        print("converting {} to pandas".format(file))
        output_file_name = os.path.join(output_dir, fname_base+".pkl")
        print("will be saved as {}".format(output_file_name))
        args.fname = os.path.join(input_dir, file)
        df = convert_rec_to_pandas(args)
        df.to_pickle(output_file_name)

    # test_file_norad_inb = fs.data_path+ "rec_inbend_norad/" +"norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon.root"
    # test_file_norad_outb = fs.data_path+ "rec_outbend_norad/" +"norad_10000_20230703_1814_Fall_2018_Outbending_100_50nA_recon.root"
    # test_file_rad_inb = fs.data_path+ "rec_inbend_rad/" +"rad_10000_20230126_1645_Fall_2018_Inbending_50nA_recon.root"
    # test_file_rad_outb = fs.data_path+ "rec_outbend_rad/" +"rad_10000_20230126_1645_Fall_2018_Outbending_100_50nA_recon.root"


    # test_outfile_norad_inb = "/mnt/c/Users/rober/Dropbox/Bobby/Linux/work/CLAS12/mit-clas12-analysis/theana/paragon/analysis/threnody/0_convert_root_to_pickle/Rec/test/rec_test_norad_inb.pkl"
    # test_outfile_norad_outb = "/mnt/c/Users/rober/Dropbox/Bobby/Linux/work/CLAS12/mit-clas12-analysis/theana/paragon/analysis/threnody/0_convert_root_to_pickle/Rec/test/rec_test_norad_outb.pkl"
    # test_outfile_rad_inb = "/mnt/c/Users/rober/Dropbox/Bobby/Linux/work/CLAS12/mit-clas12-analysis/theana/paragon/analysis/threnody/0_convert_root_to_pickle/Rec/test/rec_test_rad_inb.pkl"
    # test_outfile_rad_outb = "/mnt/c/Users/rober/Dropbox/Bobby/Linux/work/CLAS12/mit-clas12-analysis/theana/paragon/analysis/threnody/0_convert_root_to_pickle/Rec/test/rec_test_rad_outb.pkl"


    
    # if args.test:
    #     if args.rad:
    #         if args.polarity == "inbending":
    #             test_file = test_file_rad_inb
    #             outdir = "/mnt/d/GLOBUS/CLAS12/Thesis/1_potential_dvpip_events/inb/rec/"
    #         elif args.polarity =="outbending":
    #             test_file = test_file_rad_outb
    #             outdir = "/mnt/d/GLOBUS/CLAS12/Thesis/1_potential_dvpip_events/outb/rec/"
                
    #             #args.out = test_outfile_rad_outb
    #     else:
    #         if args.polarity == "inbending":
    #             test_file = test_file_norad_inb
    #             outdir = "/mnt/d/GLOBUS/CLAS12/Thesis/1_potential_dvpip_events/inb/rec/"

    #            # args.out = test_outfile_norad_inb
    #         elif args.polarity =="outbending":
    #             test_file = test_file_norad_outb
    #             outdir = "/mnt/d/GLOBUS/CLAS12/Thesis/1_potential_dvpip_events/outb/rec/"

    #            # args.out = test_outfile_norad_outb
    #     print("test enabled, using {}".format(test_file))
    #     args.fname = test_file

    # fname_base = args.fname.split(".")[0]

    # print("converting {} to pandas".format(args.fname))


    # polarities = ["inbending", "outbending"]

    # options = ["_nofid_nocorr_nosmear","_fid_nocorr_nosmear","_fid_corr_nosmear","_fid_corr_smear"]
    # corr_config = [False,False,True,True]
    # smear_config = [0,0,0,1]
    # fid_config = [1,0,0,0]

    # nominal_inbend = fs.data_path+ "rec_inbend_norad/" +"norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon.root"
    # nominal_outbend = fs.data_path+ "rec_outbend_norad/" +"norad_10000_20230703_1814_Fall_2018_Outbending_100_50nA_recon.root"

    # for polarity_option in polarities:
    #     args.polarity =polarity_option
    #     if args.polarity == "inbending":
    #         args.fname = nominal_inbend
    #         outdir = "/mnt/d/GLOBUS/CLAS12/Thesis/1_potential_dvpip_events/inb/rec/"
    #     else:
    #         args.fname = nominal_outbend
    #         outdir = "/mnt/d/GLOBUS/CLAS12/Thesis/1_potential_dvpip_events/outb/rec/"

    #     fname_base = args.fname.split(".")[0].split("/")[-1]
    #     print(fname_base)
        
    #     for option in options:
    #         name_suffix = option
    #         args.correction = corr_config[options.index(option)]
    #         args.smearing = smear_config[options.index(option)]
    #         args.nofid = fid_config[options.index(option)]
    #         print("converting {} to pandas".format(args.fname))
    #         print("correction: {}, smearing: {}, fiducial: {}".format(args.correction, args.smearing, args.nofid))
    #         df = convert_rec_to_pandas(args)
    #         df.to_pickle(outdir+fname_base+name_suffix+".pkl")





