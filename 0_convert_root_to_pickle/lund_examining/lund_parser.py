  
#!/usr/bin/env python3

# # # # import os

from utils import const, physics
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
from itertools import combinations

PhysicsConstants = const.PhysicsConstants()
print(PhysicsConstants.proton_mass) # prints 0.938272081


def read_multiple(args,lund_batch):
    print("READING MULTIPLE")
    # Initialize an empty list to store dataframes
    chunk_size = 1000
    index = 0

    if args.rad:
        file_dir = args.rad_dir
        filter_lund_func = filter_rad_lund
        print("RAD")
    else:
        filter_lund_func = filter_norad_lund
        file_dir = args.dir
        print("NORAD")

    #convert file_dir to only have _ instead of /
    output_file_base = file_dir.replace("/","_")


    #make an output dir named lund_to_pickle
    if not os.path.exists("lund_to_pickle"):
        os.makedirs("lund_to_pickle")

    file_list = os.listdir(file_dir)
    # if there are more than 1000 files, process in batches of 1000
    if len(file_list) > chunk_size:
        #break into parts
        file_list_parts = [file_list[i:i + chunk_size] for i in range(0, len(file_list), chunk_size)]
        print("There will be {} parts".format(len(file_list_parts)))
    else:
        file_list_parts = [file_list]
        print("there are only {} files".format(len(file_list)))

    for part in file_list_parts:
        print("On part {}".format(part))
        # Loop over all files in the directory
        dfs = []
        for filename in part:
            print("on file {}".format(filename))
            # Check if the file is of the correct type (e.g. text file)
            if filename.endswith(".lund"):
                # Full file path
                filepath = os.path.join(file_dir, filename)
                print(filepath)
                # Apply filter to each file and store the resulting dataframe
                df_1 = filter_lund_func(filepath)
                # Append the dataframe to the list
                dfs.append(df_1)

        # Combine all dataframes into a single one
        df = pd.concat(dfs, ignore_index=True)

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
        df.loc[:,'GenconeAngle'] = physics.angle(ele, pi0)
        df.loc[:,'GenreconGam'] = physics.angle(gam, VmissEP)
        df.loc[:,'Gencoplanarity'] = physics.angle(v3h, v3pi0)

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


        # save as output_file_base + index name
        df.to_pickle("/mnt/d/GLOBUS/CLAS12/Thesis/1_potential_dvpip_events/gen/{}/{}_{}.pkl".format(lund_batch,output_file_base,index))

        #increment
        index += 1

    #return the last dataframe
    return df


def filter_rad_lund(filename):
    filter_infile_name = filename
    data = []

    with open(filter_infile_name,"r") as lst:
        txtlst = lst.readlines()
    
    for ind,line in enumerate(txtlst):
        if ind %5000 == 0:
            print("On event {}".format(ind/5))

        if ind % 5 == 0:
            a = line
            b = txtlst[ind+1]
            c = txtlst[ind+2]
            d = txtlst[ind+3]
            e = txtlst[ind+4]
            for sub_line in (a,b,c,d,e):
                #print(sub_line)
                cols = sub_line.split()
                # gather particle 4 momentum:
                if cols[3]=='11':
                    e_4mom = (float(cols[9]),float(cols[6]),float(cols[7]),float(cols[8]))

                if cols[3]=='2212':
                    pro_4mom = (float(cols[9]),float(cols[6]),float(cols[7]),float(cols[8]))

                if cols[3]=='111':
                    pi_4mom = (float(cols[9]),float(cols[6]),float(cols[7]),float(cols[8]))

                # get the radiated photon momenta
                if cols[3]=='22':
                    photon_4mom = (float(cols[9]),float(cols[6]),float(cols[7]),float(cols[8]))

            virtual_gamma = physics.vec_subtract(PhysicsConstants.electron_beam_4_vector,e_4mom)
            Q2 = -1*physics.calc_inv_mass_squared(virtual_gamma)
            W2 = physics.calc_inv_mass_squared(physics.vec_subtract(physics.vec_add(PhysicsConstants.electron_beam_4_vector,PhysicsConstants.target_4_vector),e_4mom))
            nu = virtual_gamma[0]
            xB = Q2/(2*PhysicsConstants.proton_mass*nu)
            t = -1*physics.calc_inv_mass_squared(physics.vec_subtract(PhysicsConstants.target_4_vector,pro_4mom))

            event = {
                'GenEe': e_4mom[0], 'GenEpx': e_4mom[1], 'GenEpy': e_4mom[2], 'GenEpz': e_4mom[3], 
                'GenPe': pro_4mom[0], 'GenPpx': pro_4mom[1], 'GenPpy': pro_4mom[2], 'GenPpz': pro_4mom[3],
                'GenPie': pi_4mom[0], 'GenPipx': pi_4mom[1], 'GenPipy': pi_4mom[2], 'GenPipz': pi_4mom[3],
                'GenGe': photon_4mom[0], 'GenGpx': photon_4mom[1], 'GenGpy': photon_4mom[2], 'GenGpz': photon_4mom[3], 
                'GenQ2': Q2, 'GenW2': W2, 'Gennu': nu, 'GenxB': xB, 'Gent': t
            }

            data.append(event)

    return pd.DataFrame(data)


def filter_norad_lund(filename):
    filter_infile_name = filename
    data = []
    photon_momentums = []

    with open(filter_infile_name,"r") as lst:
        txtlst = lst.readlines()
    
    for ind,line in enumerate(txtlst):
        if ind %5000 == 0:
            print("On event {}".format(ind/5))

        if ind % 5 == 0:
            a = line
            b = txtlst[ind+1]
            c = txtlst[ind+2]
            d = txtlst[ind+3]
            e = txtlst[ind+4]
            for sub_line in (a,b,c,d,e):
                #print(sub_line)
                cols = sub_line.split()
                # gather particle 4 momentum:
                if cols[3]=='11':
                    e_4mom = (float(cols[9]),float(cols[6]),float(cols[7]),float(cols[8]))
                
                if cols[3]=='2212':
                    pro_4mom = (float(cols[9]),float(cols[6]),float(cols[7]),float(cols[8]))

                # get the two photon momentums
                if cols[3]=='22':
                    photon_momentums.append((float(cols[9]),float(cols[6]),float(cols[7]),float(cols[8])))
                    if len(photon_momentums) == 2:
                        if photon_momentums[0][0] > photon_momentums[1][0]:
                            higher_energy_photon_4mom = photon_momentums[0]
                            lower_energy_photon_4mom = photon_momentums[1]
                        else:
                            higher_energy_photon_4mom = photon_momentums[1]
                            lower_energy_photon_4mom = photon_momentums[0]
                        photon_momentums = []

            virtual_gamma = physics.vec_subtract(PhysicsConstants.electron_beam_4_vector,e_4mom)
            Q2 = -1*physics.calc_inv_mass_squared(virtual_gamma)
            W2 = physics.calc_inv_mass_squared(physics.vec_subtract(physics.vec_add(PhysicsConstants.electron_beam_4_vector,PhysicsConstants.target_4_vector),e_4mom))
            nu = virtual_gamma[0]
            xB = Q2/(2*PhysicsConstants.proton_mass*nu)
            t = -1*physics.calc_inv_mass_squared(physics.vec_subtract(PhysicsConstants.target_4_vector,pro_4mom))

            event = {'GenEe': e_4mom[0], 'GenEpx': e_4mom[1], 'GenEpy': e_4mom[2], 'GenEpz': e_4mom[3], 
                     'GenPe': pro_4mom[0], 'GenPpx': pro_4mom[1], 'GenPpy': pro_4mom[2], 'GenPpz': pro_4mom[3],
                     'GenGe': higher_energy_photon_4mom[0], 'GenGpx': higher_energy_photon_4mom[1], 
                     'GenGpy': higher_energy_photon_4mom[2], 'GenGpz': higher_energy_photon_4mom[3], 
                     'GenGe2': lower_energy_photon_4mom[0], 'GenGpx2': lower_energy_photon_4mom[1], 
                     'GenGpy2': lower_energy_photon_4mom[2], 'GenGpz2': lower_energy_photon_4mom[3], 
                     'GenQ2': Q2, 'GenW2': W2, 'Gennu': nu, 'GenxB': xB, 'Gent': t}
            data.append(event)


    return pd.DataFrame(data)




# Start the entry point of the script
if __name__ == "__main__":
    #Now use argparser to take in the file name and include a default
    # Create an ArgumentParser object

    test_dir = "/mnt/c/Users/rober/Dropbox/Bobby/Linux/work/CLAS12/mit-clas12-analysis/theana/paragon/analysis/threnody/0_convert_root_to_pickle/lund_examining/norad_test/"
    test_file = "/mnt/c/Users/rober/Dropbox/Bobby/Linux/work/CLAS12/mit-clas12-analysis/theana/paragon/analysis/threnody/0_convert_root_to_pickle/lund_examining/norad_test/test_aao_norad_2023.lund"
    test_pickle = "/mnt/c/Users/rober/Dropbox/Bobby/Linux/work/CLAS12/mit-clas12-analysis/theana/paragon/analysis/threnody/0_convert_root_to_pickle/lund_examining/norad_test/all_norad.pkl"
    test_dir_rad = "/mnt/c/Users/rober/Dropbox/Bobby/Linux/work/CLAS12/mit-clas12-analysis/theana/paragon/analysis/threnody/0_convert_root_to_pickle/lund_examining/rad_test/"
    big_test_rad_dir = "/mnt/d/GLOBUS/CLAS12/lund_rad_examining/lund_files/"
    parser = argparse.ArgumentParser(description="Process an input file and save the resulting DataFrame as a pickle file.")
    # Add an argument for the input file

    parser.add_argument('-f', '--file', default=test_file, help='The input file to process')
    parser.add_argument('-d', '--dir', default=test_dir, help='The input file to process')
    parser.add_argument('-x', '--rad_dir', default=big_test_rad_dir, help='The input file to process')
    parser.add_argument('-p', '--pickle', default=test_pickle, help='The input file to process')
    #set a store true flag

    parser.add_argument('-m', '--multiple', action='store_true', help='Store the resulting DataFrame as a pickle file')
    parser.add_argument('-r', '--rad', action='store_true', help='Store the resulting DataFrame as a pickle file')
    parser.add_argument('-l', '--lund_process', action='store_true', help='Store the resulting DataFrame as a pickle file')

    # Parse the command-line arguments
    args = parser.parse_args()

    

    # Append to the name based on the flags
    if args.rad:
        pickle_basename = test_dir_rad+"rad_output.pkl"
    else:
        pickle_basename = test_dir+"norad_output.pkl"

    lund_batch = args.dir.split("/")[-2]
    # print(lund_batch)
    if args.lund_process:
        if args.multiple:
            df = read_multiple(args,lund_batch)
        else:
            df = filter_norad_lund(args.file)
    






