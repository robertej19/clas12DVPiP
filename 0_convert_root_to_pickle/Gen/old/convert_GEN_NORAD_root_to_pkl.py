import uproot
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from copy import copy
import sys
import utils.make_histos
#import make_histos
from icecream import ic
import sys

M = 0.938272081 # target mass
me = 0.5109989461 * 0.001 # electron mass
ebeam = 10.604 # beam energy
pbeam = np.sqrt(ebeam * ebeam - me * me) # beam electron momentum
beam = [0, 0, pbeam] # beam vector
target = [0, 0, 0] # target vector

def dot(vec1, vec2):
    # dot product of two 3d vectors
    return vec1[0]*vec2[0]+vec1[1]*vec2[1]+vec1[2]*vec2[2]

def mag(vec1):
    # L2 norm of vector
    return np.sqrt(dot(vec1, vec1))

def mag2(vec1):
    # square of L2 norm
    return  dot(vec1, vec1)

def cosTheta(vec1, vec2):
    # cosine angle between two 3d vectors
    return dot(vec1,vec2)/np.sqrt(mag2(vec1) * mag2(vec2))

def angle(vec1, vec2):
    # angle between two 3d vectors
    return 180/np.pi*np.arccos(np.minimum(1, cosTheta(vec1, vec2)))

def cross(vec1, vec2):
    # cross product of two 3d vectors
    return [vec1[1]*vec2[2]-vec1[2]*vec2[1], vec1[2]*vec2[0]-vec1[0]*vec2[2], vec1[0]*vec2[1]-vec1[1]*vec2[0]]

def vecAdd(gam1, gam2):
    # add two 3d vectors
    return [gam1[0]+gam2[0], gam1[1]+gam2[1], gam1[2]+gam2[2]]

def pi0Energy(gam1, gam2):
    # reconstructed pi0 energy of two 3d photon momenta
    return mag(gam1)+mag(gam2)

def pi0InvMass(gam1, gam2):
    # pi0 invariant mass of two 3d photon momenta
    pi0mass2 = pi0Energy(gam1, gam2)**2-mag2(vecAdd(gam1, gam2))
    pi0mass2 = np.where(pi0mass2 >= 0, pi0mass2, 10**6)
    pi0mass = np.sqrt(pi0mass2)
    pi0mass = np.where(pi0mass > 100, -1000, pi0mass)
    return pi0mass

def getPhi(vec1):
    # azimuthal angle of one 3d vector
    return 180/np.pi*np.arctan2(vec1[1], vec1[0])

def getTheta(vec1):
    # polar angle of one 3d vector
    return 180/np.pi*np.arctan2(np.sqrt(vec1[0]*vec1[0]+vec1[1]*vec1[1]), vec1[2])

def getEnergy(vec1, mass):
    # for taken 3d momenta p and mass m, return energy = sqrt(p**2 + m**2)
    return np.sqrt(mag2(vec1)+mass**2)

def readFile(fname):
    #read root using uproot
    ffile = uproot.open(fname)
    tree = ffile["T"]
    return tree

# entry_start (None or int) – The first entry to include. If None, start at zero. If negative, count from the end, like a Python slice.
# entry_stop (None or int) – The first entry to exclude (i.e. one greater than the last entry to include). If None, stop at num_entries. If negative, count from the end, like a Python slice.

def readEPGG(filename, entry_stop = None):
    tree = readFile(filename)
    
    # data frames and their keys to read Z part
    df_electronGen = pd.DataFrame()
    df_protonGen = pd.DataFrame()
    df_pionGen = pd.DataFrame()
    df_gammaGen = pd.DataFrame()
    eleKeysGen = ["GenEpx", "GenEpy", "GenEpz"]
    proKeysGen = ["GenPpx", "GenPpy", "GenPpz"]
    gamKeysGen = ["GenGpx", "GenGpy", "GenGpz"]
    # read keys
    for key in eleKeysGen:
        df_electronGen[key] = tree[key].array(library="pd", entry_stop=entry_stop)
    for key in proKeysGen:
        df_protonGen[key] = tree[key].array(library="pd", entry_stop=entry_stop)
    for key in gamKeysGen:
        df_gammaGen[key] = tree[key].array(library="pd", entry_stop=entry_stop)



        # array = tree[key].array(library="pd", entry_stop=entry_stop)
        # print(f'Shape of {key} array: {array.shape}')
        # df_gammaGen[key] = array
        
    #print all dataframes:
    print(df_electronGen)
    print(df_protonGen)
    print(df_gammaGen)

    #convert data type to standard double
    df_electronGen = df_electronGen.astype({"GenEpx": float, "GenEpy": float, "GenEpz": float})
    df_protonGen = df_protonGen.astype({"GenPpx": float, "GenPpy": float, "GenPpz": float})
    df_gammaGen = df_gammaGen.astype({"GenGpx": float, "GenGpy": float, "GenGpz": float})

    sys.exit()

    #set up a dummy index for merging
    df_electronGen.loc[:,'event'] = df_electronGen.index
    df_protonGen.loc[:,'event'] = df_protonGen.index
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

    gamGen = [df_gammaGen["GenGpx"], df_gammaGen["GenGpy"], df_gammaGen["GenGpz"]]
    df_gammaGen.loc[:, 'GenGp'] = mag(gamGen)
    df_gammaGen.loc[:, 'GenGtheta'] = getTheta(gamGen)
    df_gammaGen.loc[:, 'GenGphi'] = getPhi(gamGen)

    gamGen2 = [df_gammaGen["GenGpx2"], df_gammaGen["GenGpy2"], df_gammaGen["GenGpz2"]]
    debug = df_gammaGen.loc[:, 'GenGp2'] == mag(gamGen2)
    df_gammaGen.loc[:, 'GenGtheta2'] = getTheta(gamGen2)
    df_gammaGen.loc[:, 'GenGphi2'] = getPhi(gamGen2)

    df = pd.merge(df_electronGen, df_protonGen, how='inner', on='event')
    df = pd.merge(df, df_gammaGen, how='inner', on='event')

    ele = [df['GenEpx'], df['GenEpy'], df['GenEpz']]
    df.loc[:, 'GenEp'] = mag(ele)
    df.loc[:, 'GenEe'] = getEnergy(ele, me)
    df.loc[:, 'GenEtheta'] = getTheta(ele)
    df.loc[:, 'GenEphi'] = getPhi(ele)
    pro = [df['GenPpx'], df['GenPpy'], df['GenPpz']]
    df.loc[:, 'GenPp'] = mag(pro)
    df.loc[:, 'GenPe'] = getEnergy(pro, M)
    df.loc[:, 'GenPtheta'] = getTheta(pro)
    df.loc[:, 'GenPphi'] = getPhi(pro)
    gam = [df['GenGpx'], df['GenGpy'], df['GenGpz']]
    df.loc[:, 'GenGp'] = mag(gam)
    df.loc[:, 'GenGe'] = getEnergy(gam, 0)
    df.loc[:, 'GenGtheta'] = getTheta(gam)
    df.loc[:, 'GenGphi'] = getPhi(gam)

    Ppt = mag([df['GenPpx'], df['GenPpy'], 0])
    VGS = [-df['GenEpx'], -df['GenEpy'], pbeam - df['GenEpz']]
    v3l = cross(beam, ele)
    v3h = cross(pro, VGS)
    v3g = cross(VGS, gam)
    VmissG = [-df["GenEpx"] - df["GenPpx"], -df["GenEpy"] - df["GenPpy"],
                pbeam - df["GenEpz"] - df["GenPpz"]]
    VmissP = [-(df["GenEpx"] + df["GenGpx"]), -(df["GenEpy"] + df["GenGpy"]),
                -(-pbeam + df["GenEpz"] + df["GenGpz"])]
    Vmiss = [-(df["GenEpx"] + df["GenPpx"] + df["GenGpx"]), -(df["GenEpy"] + df["GenPpy"] + df["GenGpy"]),
                -(-pbeam + df["GenEpz"] + df["GenPpz"] + df["GenGpz"])]
    costheta = cosTheta(VGS, gam)
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

    return df



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get args",formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-f","--fname", help="a single root file to convert into pickles", default="infile.root")
    parser.add_argument("-t","--test", help="use to enable testing flag", action='store_true',default=False)
    parser.add_argument("-o","--out", help="a single pickle file name as an output", default="outfile.pkl")
    parser.add_argument("-s","--entry_stop", help="entry_stop to stop reading the root file", default = None)
    
    args = parser.parse_args()

    if args.test:
        test_file = "tests/sample_noradgen_1.root"
        print("test enabled, using {}")
        args.fname = test_file

    fname_base = args.fname.split(".")[0]

    df_gen = readEPGG(args.fname)

    print(df_gen.shape)
    print(df_gen.head(20))


    print(df_gen.columns)

    df_gen.to_pickle(args.out)


    # df_gen.to_pickle(fname_base+"_genONLY.pkl")

    # #df_e = df.query("particleID == 11")
        
    # df_e = df_gen
    # ic(df_e.columns)
    # x_data = df_e['GenEpx']
    # y_data = df_e['GenEpy']
    # var_names = ['xb','q2']
    # ranges = [[-2,2,100],[-2,2,120]]

    # make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
    #     saveplot=False,pics_dir="none",plot_title="none",
    #     filename="ExamplePlot",units=["",""])



