  
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


def read_multiple(args):
    print("READING MULTIPLE")
    # Initialize an empty list to store dataframes
    dfs = []


    if args.rad:
        file_dir = args.rad_dir
        filter_lund_func = filter_rad_lund
        spherical_func = get_rad_spherical
        print("RAD")
    else:
        filter_lund_func = filter_norad_lund
        file_dir = args.dir
        spherical_func = get_norad_spherical
        print("NORAD")

    # Loop over all files in the directory
    for filename in os.listdir(file_dir):
        print("on file {}".format(filename))
        # Check if the file is of the correct type (e.g. text file)
        if filename.endswith(".lund"):
            # Full file path
            filepath = os.path.join(file_dir, filename)
            print(filepath)
            # Apply filter to each file and store the resulting dataframe
            df = filter_lund_func(filepath)
            # Append the dataframe to the list
            dfs.append(df)

    # Combine all dataframes into a single one
    result = pd.concat(dfs, ignore_index=True)

    df_out = spherical_func(result)

    
    return df_out


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

def get_norad_spherical(df):
    # Electron spherical coordinates
    total_mom_e, theta_e, phi_e = physics.cartesian_to_spherical(df['GenEpx'],df['GenEpy'],df['GenEpz'])

    # Proton spherical coordinates
    total_mom_pro, theta_pro, phi_pro = physics.cartesian_to_spherical(df['GenPpx'],df['GenPpy'],df['GenPpz'])

    # Higher energy photon spherical coordinates
    total_mom_higher_photon, theta_higher_photon, phi_higher_photon = physics.cartesian_to_spherical(df['GenGpx'],df['GenGpy'],df['GenGpz'])

    # Lower energy photon spherical coordinates
    total_mom_lower_photon, theta_lower_photon, phi_lower_photon = physics.cartesian_to_spherical(df['GenGpx2'],df['GenGpy2'],df['GenGpz2'])

    # Add the spherical coordinates to the DataFrame
    df['GenEp'] = total_mom_e
    df['GenEtheta'] = theta_e
    df['GenEphi'] = phi_e
    df['GenPp'] = total_mom_pro
    df['GenPtheta'] = theta_pro
    df['GenPphi'] = phi_pro
    df['GenGp'] = total_mom_higher_photon
    df['GenGtheta'] = theta_higher_photon
    df['GenGphi'] = phi_higher_photon
    df['GenGp2'] = total_mom_lower_photon
    df['GenGtheta2'] = theta_lower_photon
    df['GenGphi2'] = phi_lower_photon

    return df

def get_rad_spherical(df):
    # Electron spherical coordinates
    total_mom_e, theta_e, phi_e = physics.cartesian_to_spherical(df['GenEpx'],df['GenGpy'],df['GenGpz'])

    # Proton spherical coordinates
    total_mom_pro, theta_pro, phi_pro = physics.cartesian_to_spherical(df['GenPpx'],df['GenPpz'],df['GenPpz'])

    # photon spherical coordinates
    total_mom_photon, theta_photon, phi_photon = physics.cartesian_to_spherical(df['GenGpx'],df['GenGpy'],df['GenGpz'])

    # pion spherical coordinates
    total_mom_pion, theta_pion, phi_pion = physics.cartesian_to_spherical(df['GenPipx'],df['GenPipy'],df['GenPipz'])

    # Add the spherical coordinates to the DataFrame
    df['GenEp'] = total_mom_e
    df['GenEtheta'] = theta_e
    df['GenEphi'] = phi_e
    df['GenPp'] = total_mom_pro
    df['GenPtheta'] = theta_pro
    df['GenPphi'] = phi_pro
    df['GenGp'] = total_mom_photon
    df['GenGtheta'] = theta_photon
    df['GenGphi'] = phi_photon
    df['GenPip'] = total_mom_pion
    df['GenPitheta'] = theta_pion
    df['GenPiphi'] = phi_pion


    return df


def plot_2d_hist(df):    # List of all the columns you want to create combinations for
    columns = df.columns

    dir = "figs/"
    # Find all combinations of 2 variables
    comb = combinations(columns, 2)

    # Iterate over each combination and create a histogram
    for i in list(comb):
        #print what iteration we are on
        print(i)
        plt.figure()
        plt.hist2d(df[i[0]], df[i[1]], bins=30, cmap='viridis')  # adjust bins and cmap as needed
        plt.xlabel(i[0])
        plt.ylabel(i[1])
        plt.colorbar(label='Counts')
        plt.title(f'Histogram of {i[0]} vs {i[1]}')
        plt.savefig(dir+f'{i[0]}_vs_{i[1]}_histogram.png')  # save figure
        plt.close()



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

    if args.lund_process:
        if args.multiple:
            df = read_multiple(args)
        else:
            df = filter_norad_lund(args.file)
        # Save the resulting DataFrame as a pickle file, using the original filename but with a .pkl extension
        #df.to_pickle(pickle_basename)

        ele = [df['GenEpx'], df['GenEpy'], df['GenEpz']]
        df.loc[:, 'GenEp'] = physics.mag(ele)
        df.loc[:, 'GenEe_alt'] = physics.getEnergy(ele, PhysicsConstants.electron_mass)
        df.loc[:, 'GenEtheta'] = physics.getTheta(ele)
        df.loc[:, 'GenEphi'] = physics.getPhi(ele)

        pro = [df['GenPpx'], df['GenPpy'], df['GenPpz']]
        df.loc[:, 'GenPp'] = physics.mag(pro)
        df.loc[:, 'GenPe_alt'] = physics.getEnergy(pro, PhysicsConstants.proton_mass)
        df.loc[:, 'GenPtheta'] = physics.getTheta(pro)
        df.loc[:, 'GenPphi'] = physics.getPhi(pro)

        gam = [df['GenGpx'], df['GenGpy'], df['GenGpz']]
        df.loc[:, 'GenGp'] = physics.mag(gam)
        df.loc[:, 'GenGe_alt'] = physics.getEnergy(gam, 0)
        df.loc[:, 'GenGtheta'] = physics.getTheta(gam)
        df.loc[:, 'GenGphi'] = physics.getPhi(gam)
        
        gam2 = [df['GenGpx2'], df['GenGpy2'], df['GenGpz2']]
        df.loc[:, 'GenGp2'] = physics.mag(gam2)
        df.loc[:,'GenGe2_alt'] = physics.getEnergy(gam2, 0)
        df.loc[:, 'GenGtheta2'] = physics.getTheta(gam2)
        df.loc[:, 'GenGphi2'] = physics.getPhi(gam2)

        pi0 = physics.vecAdd(gam, gam2)
        VGS = [-df['GenEpx'], -df['GenEpy'], PhysicsConstants.electron_beam_momentum_magnitude - df['GenEpz']]
        v3l = physics.cross(PhysicsConstants.electron_beam_3_vector, ele)
        v3h = physics.cross(pro, VGS)
        v3g = physics.cross(VGS, gam)
        v3pi0 = physics.cross(VGS, pi0)

        VmissG = [-df["GenEpx"] - df["GenPpx"], -df["GenEpy"] - df["GenPpy"],
                PhysicsConstants.electron_beam_momentum_magnitude - df["GenEpz"] - df["GenPpz"]]

        VmissP = [-(df["GenEpx"] + df["GenGpx"]+ df["GenGpx2"]), -(df["GenEpy"] + df["GenGpy"]+ df["GenGpy2"]),
                    -(-PhysicsConstants.electron_beam_momentum_magnitude + df["GenEpz"] + df["GenGpz"]+ df["GenGpz2"])]
        
        Vmiss = [-(df["GenEpx"] + df["GenPpx"] + df["GenGpx"]+ df["GenGpx2"]), -(df["GenEpy"] + df["GenPpy"] + df["GenGpy"]+ df["GenGpy2"]),
                    -(-PhysicsConstants.electron_beam_momentum_magnitude + df["GenEpz"] + df["GenPpz"] + df["GenGpz"]+ df["GenGpz2"])]
        
        Vmiss2 = [-(df["GenEpx"] + df["GenPpx"] + df["GenGpx"]+ df["GenGpx2"]), -(df["GenEpy"] + df["GenPpy"] + df["GenGpy"]+ df["GenGpy2"]),
                    -(-PhysicsConstants.electron_beam_momentum_magnitude + df["GenEpz"] + df["GenPpz"] + df["GenGpz"]+ df["GenGpz2"])]
        
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
        df.loc[:,'GenMM2_epgg'] = (-PhysicsConstants.proton_mass - PhysicsConstants.electron_beam_energy + df["GenEe"] +
                        df["GenPe"] + df["GenGe"]+ df["GenGe2"])**2 - physics.mag2(Vmiss)
        
        df.loc[:,'GenME_epg'] = (PhysicsConstants.proton_mass + PhysicsConstants.electron_beam_energy - df["GenEe"] - df["GenPe"] - df["GenGe"])
        df.loc[:,'GenMM2_ep'] = (-PhysicsConstants.proton_mass - PhysicsConstants.electron_beam_energy + df["GenEe"] + df["GenPe"])**2 - physics.mag2(VmissG)
        df.loc[:,'GenMM2_eg'] = (-PhysicsConstants.proton_mass - PhysicsConstants.electron_beam_energy + df["GenEe"] + df["GenGe"]+ df["GenGe2"])**2 - physics.mag2(VmissP)
        df.loc[:,'GenMPt'] = np.sqrt((df["GenEpx"] + df["GenPpx"] + df["GenGpx"])**2 +
                                (df["GenEpy"] + df["GenPpy"] + df["GenGpy"])**2)
        df.loc[:,'GenconeAngle'] = physics.angle(ele, gam)
        df.loc[:,'GenreconGam'] = physics.angle(gam, VmissG)
        df.loc[:,'Gencoplanarity'] = physics.angle(v3h, v3g)


        df.to_pickle("panick3.pkl")

        import matplotlib.pyplot as plt
        import matplotlib as mpl

        #plt.hist(np.sqrt(df['GenMM2_eg']), bins=50)
        # plt.hist((df['GenEe_alt']-df['GenEe']), bins=50)

        # plt.show()

        # plt.hist((df['GenPe_alt']-df['GenPe']), bins=50)

        # plt.show()
        # plt.hist((df['GenGe_alt']-df['GenGe']), bins=50)

        # plt.show()
        # plt.hist((df['GenGe2_alt']-df['GenGe2']), bins=50)

        # plt.show()

        # make 2d histogram of GenPe_alt vs GenPe:
        # plt.hist2d(df['GenPe_alt'], df['GenPe'], bins=50, norm=mpl.colors.LogNorm())
        # plt.xlabel('GenPe_alt')
        # plt.show()

        pe_diff = df['GenPe_alt']-df['GenPe']
        plt.plot(pe_diff, df['GenPe'], 'o')
        plt.xlabel('GenPe_alt - GenPe')
        plt.ylabel('GenPe')
        plt.show()
    else:
        #df = pd.read_pickle(pickle_basename)
        df = pd.read_pickle("panick2.pkl")
        print(df.columns.values)
        print(df['e_vx'].nunique())

        df['W'] = np.sqrt(df['W2'])
        df = df[df['W'] >= 2]
        print(df)
        print("Min:", df["Q2"].min())
        print("Max:", df["Q2"].max())

        import matplotlib as mpl

        plt.hist2d(df['xB'],df['Q2'], bins=[200, 200], norm=mpl.colors.LogNorm())

        #plt.hist2d(x_data, y_data, bins =[x_bins, y_bins],
        #range=[[xmin,xmax],[ymin,ymax]],norm=mpl.colors.LogNorm())# cmap = plt.cm.nipy_spectral) 


        plt.title('Histogram of column_name')
        plt.xlabel('column_name')
        plt.ylabel('Frequency')
        #make log distribution
        #plt.yscale('log')
        plt.show()
        # df = df[df['Q2'] >= 1]
        # print(df)
        # df = df[df['W'] >= 2]
        # print(df)







#plot_2d_hist(df)

# # Calculate Etheta
# df['Etheta'] = np.degrees(np.arccos(df['Epz'] / df['Ep']))

# # Calculate 4-momenta
# df['Ee'] = np.sqrt(df['Epx']**2 + df['Epy']**2 + df['Epz']**2 + PhysicsConstants.electron_mass**2)
# df['E_4mom'] = df.apply(lambda row: (row['Ee'], row['Epx'], row['Epy'], row['Epz']), axis=1)

# # Calculate Q^2, W^2, xB
# df['Q2'] = df['E_4mom'].apply(lambda p: -physics.calc_inv_mass_squared(physics.vec_subtract(PhysicsConstants.electron_beam_4_vector, p)))
# df['W2'] = df['E_4mom'].apply(lambda p: physics.calc_inv_mass_squared(physics.vec_subtract(physics.vec_add(PhysicsConstants.electron_beam_4_vector, PhysicsConstants.target_4_vector), p)))
# df['nu'] = df['E_4mom'].apply(lambda p: physics.vec_subtract(PhysicsConstants.electron_beam_4_vector, p)[0])
# df['xB'] = df['Q2'] / (2 * PhysicsConstants.proton_mass * df['nu'])
# df['W'] = np.sqrt(df['W2'])
