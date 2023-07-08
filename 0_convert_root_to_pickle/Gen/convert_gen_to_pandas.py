#gen

import uproot
import pandas as pd
import numpy as np
import argparse
from copy import copy
from utils import const, physics
import os

PhysicsConstants = const.PhysicsConstants()

def convert_gen_rad_to_pandas(args):

        df_electronGen = pd.DataFrame()
        df_protonGen = pd.DataFrame()
        df_pionGen = pd.DataFrame()
        df_gammaGen = pd.DataFrame()
        eleKeysGen = ["GenEpx", "GenEpy", "GenEpz"]
        proKeysGen = ["GenPpx", "GenPpy", "GenPpz"]
        piKeysGen = ["GenPipx", "GenPipy", "GenPipz"]
        gamKeysGen = ["GenGpx", "GenGpy", "GenGpz"]
        # read keys
        for key in eleKeysGen:
            print(entry_start, entry_stop)
            df_electronGen[key] = tree[key].array(library="pd",  entry_start = starting_entry, entry_stop=starting_entry+chunk_size)
            print("at least made it to here")
        for key in proKeysGen:
            df_protonGen[key] = tree[key].array(library="pd",  entry_start = starting_entry, entry_stop=starting_entry+chunk_size)
        for key in piKeysGen:
            df_pionGen[key] = tree[key].array(library="pd",  entry_start = starting_entry, entry_stop=starting_entry+chunk_size)
        for key in gamKeysGen:
            df_gammaGen[key] = tree[key].array(library="pd",  entry_start = starting_entry, entry_stop=starting_entry+chunk_size)

        #convert data type to standard double
        df_electronGen = df_electronGen.astype({"GenEpx": float, "GenEpy": float, "GenEpz": float})
        df_protonGen = df_protonGen.astype({"GenPpx": float, "GenPpy": float, "GenPpz": float})
        df_pionGen = df_pionGen.astype({"GenPipx": float, "GenPipy": float, "GenPipz": float})
        df_gammaGen = df_gammaGen.astype({"GenGpx": float, "GenGpy": float, "GenGpz": float})

        #set up a dummy index for merging
        df_electronGen.loc[:,'event'] = df_electronGen.index
        df_protonGen.loc[:,'event'] = df_protonGen.index
        df_pionGen.loc[:,'event'] = df_pionGen.index
        df_gammaGen.loc[:,'event'] = df_gammaGen.index.get_level_values('entry')

        #sort columns for readability
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

        piGen = [df_pionGen["GenPipx"], df_pionGen["GenPipy"], df_pionGen["GenPipz"]]
        df_pionGen.loc[:, 'GenPip'] = mag(piGen)
        df_pionGen.loc[:, 'GenPitheta'] = getTheta(piGen)
        df_pionGen.loc[:, 'GenPiphi'] = getPhi(piGen)

        gamGen = [df_gammaGen["GenGpx"], df_gammaGen["GenGpy"], df_gammaGen["GenGpz"]]
        df_gammaGen.loc[:, 'GenGp'] = mag(gamGen)
        df_gammaGen.loc[:, 'GenGtheta'] = getTheta(gamGen)
        df_gammaGen.loc[:, 'GenGphi'] = getPhi(gamGen)

        gamGen2 = [df_gammaGen["GenGpx2"], df_gammaGen["GenGpy2"], df_gammaGen["GenGpz2"]]
        debug = df_gammaGen.loc[:, 'GenGp2'] == mag(gamGen2)
        df_gammaGen.loc[:, 'GenGtheta2'] = getTheta(gamGen2)
        df_gammaGen.loc[:, 'GenGphi2'] = getPhi(gamGen2)

        df = pd.merge(df_electronGen, df_protonGen, how='inner', on='event')
        df = pd.merge(df, df_pionGen, how='inner', on='event')
        df = pd.merge(df, df_gammaGen, how='inner', on='event')

        #Calculate kinematics
        
        ele = [df['GenEpx'], df['GenEpy'], df['GenEpz']]
        pro = [df['GenPpx'], df['GenPpy'], df['GenPpz']]
        pi0 = [df['GenPipx'], df['GenPipy'], df['GenPipz']]
        gam = [df['GenGpx'], df['GenGpy'], df['GenGpz']]

        df.loc[:, 'GenEe'] = getEnergy(ele, me)
        df.loc[:, 'GenPe'] = getEnergy(pro, M)
        df.loc[:, 'GenGe'] = getEnergy(gam, 0)
        df.loc[:, 'GenPie'] = getEnergy(pi0, m_pi0)



        Ppt = mag([df['GenPpx'], df['GenPpy'], 0])
        VGS = [-df['GenEpx'], -df['GenEpy'], pbeam - df['GenEpz']]
        v3l = cross(beam, ele)
        v3h = cross(pro, VGS)
        v3g = cross(VGS, pi0)
        VmissG = [-df["GenEpx"] - df["GenPpx"], -df["GenEpy"] - df["GenPpy"],
                    pbeam - df["GenEpz"] - df["GenPpz"]]
        VmissP = [-(df["GenEpx"] + df["GenGpx"]), -(df["GenEpy"] + df["GenGpy"]),
                    -(-pbeam + df["GenEpz"] + df["GenGpz"])]
        Vmiss = [-(df["GenEpx"] + df["GenPpx"] + df["GenGpx"]), -(df["GenEpy"] + df["GenPpy"] + df["GenGpy"]),
                    -(-pbeam + df["GenEpz"] + df["GenPpz"] + df["GenGpz"])]
        costheta = cosTheta(VGS, pi0)
        df.loc[:, 'GenMpx'], df.loc[:, 'GenMpy'], df.loc[:, 'GenMpz'] = Vmiss
        # binning kinematics
        df.loc[:,'GenQ2'] = -((ebeam - df['GenEe'])**2 - mag2(VGS))
        df.loc[:,'Gennu'] = (ebeam - df['GenEe'])
        df.loc[:,'Geny'] = df['Gennu']/ebeam
        df.loc[:,'GenxB'] = df['GenQ2'] / 2.0 / M / df['Gennu']
        df.loc[:,'Gent1'] = 2 * M * (df['GenPe'] - M)
        df.loc[:,'Gent2'] = (M * df['GenQ2'] + 2 * M * df['Gennu'] * (df['Gennu'] - np.sqrt(df['Gennu'] * df['Gennu'] + df['GenQ2']) * costheta))\
        / (M + df['Gennu'] - np.sqrt(df['Gennu'] * df['Gennu'] + df['GenQ2']) * costheta)
        df.loc[:,'GenW'] = np.sqrt(np.maximum(0, (ebeam + M - df['GenEe'])**2 - mag2(VGS)))
        # trento angles
        df.loc[:,'Genphi1'] = angle(v3l, v3h)
        df.loc[:,'Genphi1'] = np.where(dot(v3l, pro) > 0, 360.0 -
                                    df['Genphi1'], df['Genphi1'])
        df.loc[:,'Genphi2'] = angle(v3l, v3g)
        df.loc[:,'Genphi2'] = np.where(dot(v3l, gam) <
                                    0, 360.0 - df['Genphi2'], df['Genphi2'])
        # exclusivity variables
        df.loc[:,'GenMM2_epg'] = (-M - ebeam + df["GenEe"] +
                                df["GenPe"] + df["GenGe"])**2 - mag2(Vmiss)
        df.loc[:,'GenME_epg'] = (M + ebeam - df["GenEe"] - df["GenPe"] - df["GenGe"])
        df.loc[:,'GenMM2_ep'] = (-M - ebeam + df["GenEe"] + df["GenPe"])**2 - mag2(VmissG)
        df.loc[:,'GenMM2_eg'] = (-M - ebeam + df["GenEe"] + df["GenGe"])**2 - mag2(VmissP)
        df.loc[:,'GenMPt'] = np.sqrt((df["GenEpx"] + df["GenPpx"] + df["GenGpx"])**2 +
                                (df["GenEpy"] + df["GenPpy"] + df["GenGpy"])**2)
        df.loc[:,'GenconeAngle'] = angle(ele, gam)
        df.loc[:,'GenreconGam'] = angle(gam, VmissG)
        df.loc[:,'Gencoplanarity'] = angle(v3h, v3g)




def convert_gen_to_pandas(args):

    gen_file = uproot.open(args.fname)
    gen_tree = gen_file["T"]

    if args.entry_stop is None:
        args.entry_stop = gen_tree.num_entries
    
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

        df_electronGen = pd.DataFrame()
        df_protonGen = pd.DataFrame()
        df_gammaGen = pd.DataFrame()
        # eleKeysGen = ["GenEpx", "GenEpy", "GenEpz", "GenEvx", "GenEvy", "GenEvz"]
        eleKeysGen = ["GenEpx", "GenEpy", "GenEpz"]
        proKeysGen = ["GenPpx", "GenPpy", "GenPpz"]
        gamKeysGen = ["GenGpx", "GenGpy", "GenGpz"]

        for key in eleKeysGen:
            df_electronGen[key] = gen_tree[key].array(library="pd", entry_start = starting_entry, entry_stop=starting_entry+args.chunk_size)
        for key in proKeysGen:
            df_protonGen[key] = gen_tree[key].array(library="pd",  entry_start = starting_entry, entry_stop=starting_entry+args.chunk_size)
        for key in gamKeysGen:
            df_gammaGen[key] = gen_tree[key].array(library="pd",  entry_start = starting_entry, entry_stop=starting_entry+args.chunk_size)



        df_electronGen = df_electronGen.astype({"GenEpx": float, "GenEpy": float, "GenEpz": float})
        df_protonGen = df_protonGen.astype({"GenPpx": float, "GenPpy": float, "GenPpz": float})
        df_gammaGen = df_gammaGen.astype({"GenGpx": float, "GenGpy": float, "GenGpz": float})

        #set up a dummy index for merging
        df_electronGen.loc[:,'event'] = df_electronGen.index
        df_protonGen.loc[:,'event'] = df_protonGen.index
        df_gammaGen.loc[:,'event'] = df_gammaGen.index.get_level_values('entry')

        #sort columns for readability
        df_electronGen = df_electronGen.loc[:, ["event", "GenEpx", "GenEpy", "GenEpz"]]

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


        #spherical coordinates
        eleGen = [df_electronGen["GenEpx"], df_electronGen["GenEpy"], df_electronGen["GenEpz"]]
        df_electronGen.loc[:, 'GenEp'] = physics.mag(eleGen)
        df_electronGen.loc[:, 'GenEtheta'] = physics.getTheta(eleGen)
        df_electronGen.loc[:, 'GenEphi'] = physics.getPhi(eleGen)

        proGen = [df_protonGen["GenPpx"], df_protonGen["GenPpy"], df_protonGen["GenPpz"]]
        df_protonGen.loc[:, 'GenPp'] = physics.mag(proGen)
        df_protonGen.loc[:, 'GenPtheta'] = physics.getTheta(proGen)
        df_protonGen.loc[:, 'GenPphi'] = physics.getPhi(proGen)

        gamGen = [df_gammaGen["GenGpx"], df_gammaGen["GenGpy"], df_gammaGen["GenGpz"]]
        df_gammaGen.loc[:, 'GenGp'] = physics.mag(gamGen)
        df_gammaGen.loc[:, 'GenGtheta'] = physics.getTheta(gamGen)
        df_gammaGen.loc[:, 'GenGphi'] = physics.getPhi(gamGen)

        gamGen2 = [df_gammaGen["GenGpx2"], df_gammaGen["GenGpy2"], df_gammaGen["GenGpz2"]]
        debug = df_gammaGen.loc[:, 'GenGp2'] = physics.mag(gamGen2)
        df_gammaGen.loc[:, 'GenGtheta2'] = physics.getTheta(gamGen2)
        df_gammaGen.loc[:, 'GenGphi2'] = physics.getPhi(gamGen2)

        df = pd.merge(df_electronGen, df_protonGen, how='inner', on='event')
        df = pd.merge(df, df_gammaGen, how='inner', on='event')

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
        gam = [df['GenGpx'], df['GenGpy'], df['GenGpz']]
        df.loc[:, 'GenGp'] = physics.mag(gam)
        df.loc[:, 'GenGe'] = physics.getEnergy(gam, 0)
        df.loc[:, 'GenGtheta'] = physics.getTheta(gam)
        df.loc[:, 'GenGphi'] = physics.getPhi(gam)

        Ppt = physics.mag([df['GenPpx'], df['GenPpy'], 0])
        VGS = [-df['GenEpx'], -df['GenEpy'],PhysicsConstants.electron_beam_momentum_magnitude - df['GenEpz']]
        v3l = physics.cross(PhysicsConstants.electron_beam_3_vector, ele)
        v3h = physics.cross(pro, VGS)
        v3g = physics.cross(VGS, gam)
        VmissG = [-df["GenEpx"] - df["GenPpx"], -df["GenEpy"] - df["GenPpy"],
                PhysicsConstants.electron_beam_momentum_magnitude - df["GenEpz"] - df["GenPpz"]]
        VmissP = [-(df["GenEpx"] + df["GenGpx"]), -(df["GenEpy"] + df["GenGpy"]),
                    -(-PhysicsConstants.electron_beam_momentum_magnitude + df["GenEpz"] + df["GenGpz"])]
        Vmiss = [-(df["GenEpx"] + df["GenPpx"] + df["GenGpx"]), -(df["GenEpy"] + df["GenPpy"] + df["GenGpy"]),
                    -(-PhysicsConstants.electron_beam_momentum_magnitude + df["GenEpz"] + df["GenPpz"] + df["GenGpz"])]
        costheta = physics.cosTheta(VGS, gam)
        df.loc[:, 'GenMpx'], df.loc[:, 'GenMpy'], df.loc[:, 'GenMpz'] = Vmiss
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
        df.loc[:,'Genphi2'] = physics.angle(v3l, v3g)
        df.loc[:,'Genphi2'] = np.where(physics.dot(v3l, gam) <
                                    0, 360.0 - df['Genphi2'], df['Genphi2'])
        # exclusivity variables
        df.loc[:,'GenMM2_epg'] = (-PhysicsConstants.proton_mass - PhysicsConstants.electron_beam_energy + df["GenEe"] +
                                df["GenPe"] + df["GenGe"])**2 - physics.mag2(Vmiss)
        df.loc[:,'GenME_epg'] = (PhysicsConstants.proton_mass + PhysicsConstants.electron_beam_energy - df["GenEe"] - df["GenPe"] - df["GenGe"])
        df.loc[:,'GenMM2_ep'] = (-PhysicsConstants.proton_mass - PhysicsConstants.electron_beam_energy + df["GenEe"] + df["GenPe"])**2 - physics.mag2(VmissG)
        df.loc[:,'GenMM2_eg'] = (-PhysicsConstants.proton_mass - PhysicsConstants.electron_beam_energy + df["GenEe"] + df["GenGe"])**2 - physics.mag2(VmissP)
        df.loc[:,'GenMPt'] = np.sqrt((df["GenEpx"] + df["GenPpx"] + df["GenGpx"])**2 +
                                (df["GenEpy"] + df["GenPpy"] + df["GenGpy"])**2)
        df.loc[:,'GenconeAngle'] = physics.angle(ele, gam)
        df.loc[:,'GenreconGam'] = physics.angle(gam, VmissG)
        df.loc[:,'Gencoplanarity'] = physics.angle(v3h, v3g)



        #save to pickle and put in the output folder we made earlier
        df.to_pickle(output_dir+filename_base+"/"+filename_base+"_genOnly_"+str(iteration)+".pkl")

        print(df)
        starting_entry+=args.chunk_size
        iteration += 1
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get args",formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-f","--fname", help="a single root file to convert into pickles", default="infile.root")
    parser.add_argument("-t","--test", help="use to enable testing flag", action='store_true',default=False)
    parser.add_argument("-r","--rad", help="use radiatve generator, otherwise use norad generator", action='store_true',default=False)
    parser.add_argument("-o","--out", help="a single pickle file name as an output", default="outfile.pkl")
    parser.add_argument("-c","--chunk_size", type=int, metavar='N', help="block size of each pandas file", default = 10_000_000)
    parser.add_argument("-s","--entry_start", type=int, metavar='N', help="entry_start to start reading the root file", default = 0)
    parser.add_argument("-p","--entry_stop", type=int, metavar='N',help="entry_stop to stop reading the root file", default = None)
    
    args = parser.parse_args()

    test_file_norad = "/mnt/c/Users/rober/Dropbox/Bobby/Linux/work/CLAS12/mit-clas12-analysis/theana/paragon/analysis/threnody/0_convert_root_to_pickle/Gen/test/gen_test_norad.root"
    test_file_rad = "/mnt/c/Users/rober/Dropbox/Bobby/Linux/work/CLAS12/mit-clas12-analysis/theana/paragon/analysis/threnody/0_convert_root_to_pickle/Gen/test/gen_test_rad.root"

    test_outfile_norad = "/mnt/c/Users/rober/Dropbox/Bobby/Linux/work/CLAS12/mit-clas12-analysis/theana/paragon/analysis/threnody/0_convert_root_to_pickle/Gen/test/gen_test_norad.pkl"
    test_outfile_rad = "/mnt/c/Users/rober/Dropbox/Bobby/Linux/work/CLAS12/mit-clas12-analysis/theana/paragon/analysis/threnody/0_convert_root_to_pickle/Gen/test/gen_test_rad.pkl"

    if args.test:
        if args.rad:
            test_file = test_file_rad
            args.out = test_outfile_rad
        else:
            test_file = test_file_norad
            args.out = test_outfile_norad
        print("test enabled, using {}".format(test_file))
        args.fname = test_file

    fname_base = args.fname.split(".")[0]

    print("converting {} to pandas".format(args.fname))
    convert_gen_to_pandas(args)