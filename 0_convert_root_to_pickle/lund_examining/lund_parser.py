  
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

def cartesian_to_spherical(x,y,z):
    # Spherical coordinates
    total_mom = np.sqrt(x**2 + y**2 + z**2) # Radius
    theta = np.arccos(z/total_mom) # polar angle - range [0, pi]
    phi = np.arctan2(y, x) # azimuthal angle - range [-pi, pi]

    return total_mom, theta, phi



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

    plt.hist(df_out['W2'], bins=50, color='blue', edgecolor='black')
    plt.title('Histogram of column_name')
    plt.xlabel('column_name')
    plt.ylabel('Frequency')
    plt.show()
    
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
                'e_E': e_4mom[0], 'e_vx': e_4mom[1], 'e_vy': e_4mom[2], 'e_vz': e_4mom[3], 
                'pro_E': pro_4mom[0], 'pro_vx': pro_4mom[1], 'pro_vy': pro_4mom[2], 'pro_vz': pro_4mom[3],
                'pi_E': pi_4mom[0], 'pi_vx': pi_4mom[1], 'pi_vy': pi_4mom[2], 'pi_vz': pi_4mom[3],
                'photon_E': photon_4mom[0], 'photon_vx': photon_4mom[1], 'photon_vy': photon_4mom[2], 'photon_vz': photon_4mom[3], 
                'Q2': Q2, 'W2': W2, 'nu': nu, 'xB': xB, 't': t
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

            event = {'e_E': e_4mom[0], 'e_vx': e_4mom[1], 'e_vy': e_4mom[2], 'e_vz': e_4mom[3], 
                     'pro_E': pro_4mom[0], 'pro_vx': pro_4mom[1], 'pro_vy': pro_4mom[2], 'pro_vz': pro_4mom[3],
                     'higher_photon_E': higher_energy_photon_4mom[0], 'higher_photon_vx': higher_energy_photon_4mom[1], 
                     'higher_photon_vy': higher_energy_photon_4mom[2], 'higher_photon_vz': higher_energy_photon_4mom[3], 
                     'lower_photon_E': lower_energy_photon_4mom[0], 'lower_photon_vx': lower_energy_photon_4mom[1], 
                     'lower_photon_vy': lower_energy_photon_4mom[2], 'lower_photon_vz': lower_energy_photon_4mom[3], 
                     'Q2': Q2, 'W2': W2, 'nu': nu, 'xB': xB, 't': t}
            data.append(event)

    return pd.DataFrame(data)

def get_norad_spherical(df):
    # Electron spherical coordinates
    total_mom_e, theta_e, phi_e = cartesian_to_spherical(df['e_vx'],df['e_vy'],df['e_vz'])

    # Proton spherical coordinates
    total_mom_pro, theta_pro, phi_pro = cartesian_to_spherical(df['pro_vx'],df['pro_vy'],df['pro_vz'])

    # Higher energy photon spherical coordinates
    total_mom_higher_photon, theta_higher_photon, phi_higher_photon = cartesian_to_spherical(df['higher_photon_vx'],df['higher_photon_vy'],df['higher_photon_vz'])

    # Lower energy photon spherical coordinates
    total_mom_lower_photon, theta_lower_photon, phi_lower_photon = cartesian_to_spherical(df['lower_photon_vx'],df['lower_photon_vy'],df['lower_photon_vz'])

    # Add the spherical coordinates to the DataFrame
    df['e_mom'] = total_mom_e
    df['e_theta'] = theta_e
    df['e_phi'] = phi_e
    df['pro_mom'] = total_mom_pro
    df['pro_theta'] = theta_pro
    df['pro_phi'] = phi_pro
    df['higher_photon_mom'] = total_mom_higher_photon
    df['higher_photon_theta'] = theta_higher_photon
    df['higher_photon_phi'] = phi_higher_photon
    df['lower_photon_mom'] = total_mom_lower_photon
    df['lower_photon_theta'] = theta_lower_photon
    df['lower_photon_phi'] = phi_lower_photon

    return df

def get_rad_spherical(df):
    # Electron spherical coordinates
    total_mom_e, theta_e, phi_e = cartesian_to_spherical(df['e_vx'],df['e_vy'],df['e_vz'])

    # Proton spherical coordinates
    total_mom_pro, theta_pro, phi_pro = cartesian_to_spherical(df['pro_vx'],df['pro_vy'],df['pro_vz'])

    # photon spherical coordinates
    total_mom_photon, theta_photon, phi_photon = cartesian_to_spherical(df['photon_vx'],df['photon_vy'],df['photon_vz'])

    # pion spherical coordinates
    total_mom_pion, theta_pion, phi_pion = cartesian_to_spherical(df['pi_vx'],df['pi_vy'],df['pi_vz'])

    # Add the spherical coordinates to the DataFrame
    df['e_mom'] = total_mom_e
    df['e_theta'] = theta_e
    df['e_phi'] = phi_e
    df['pro_mom'] = total_mom_pro
    df['pro_theta'] = theta_pro
    df['pro_phi'] = phi_pro
    df['photon_mom'] = total_mom_photon
    df['photon_theta'] = theta_photon
    df['photon_phi'] = phi_photon
    df['pion_mom'] = total_mom_pion
    df['pion_theta'] = theta_pion
    df['pion_phi'] = phi_pion


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
    parser = argparse.ArgumentParser(description="Process an input file and save the resulting DataFrame as a pickle file.")
    # Add an argument for the input file

    parser.add_argument('-f', '--file', default=test_file, help='The input file to process')
    parser.add_argument('-d', '--dir', default=test_dir, help='The input file to process')
    parser.add_argument('-x', '--rad_dir', default=test_dir_rad, help='The input file to process')
    parser.add_argument('-p', '--pickle', default=test_pickle, help='The input file to process')
    #set a store true flag


    parser.add_argument('-m', '--multiple', action='store_true', help='Store the resulting DataFrame as a pickle file')
    parser.add_argument('-r', '--rad', action='store_true', help='Store the resulting DataFrame as a pickle file')
    parser.add_argument('-l', '--lund_process', action='store_true', help='Store the resulting DataFrame as a pickle file')

    # Parse the command-line arguments
    args = parser.parse_args()

    if args.lund_process:
        if args.multiple:
            df = read_multiple(args)
        else:
            df = filter_norad_lund(args.file)
        # Save the resulting DataFrame as a pickle file, using the original filename but with a .pkl extension
        df.to_pickle(os.path.splitext(args.file)[0] + '.pkl')
    else:
        df = pd.read_pickle(args.pickle)


        # df['W'] = np.sqrt(df['W2'])
        # print(df)
        # print("Min:", df["Q2"].min())
        # print("Max:", df["Q2"].max())

        # plt.hist(df['Q2'], bins=50, color='blue', edgecolor='black')
        # plt.title('Histogram of column_name')
        # plt.xlabel('column_name')
        # plt.ylabel('Frequency')
        # plt.show()
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
