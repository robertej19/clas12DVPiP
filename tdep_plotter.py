import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os, sys
from PIL import Image
import numpy as np
import re
# import line2D for custom legend
from matplotlib.lines import Line2D

from utils import filestruct, const, make_histos
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline



PhysicsConstants = const.PhysicsConstants()


fs = filestruct.fs()
df = pd.read_pickle("t_dep_of_xsec.pkl")

def get_image_params(image_name):
    # extract params from image name using regular expression
    pattern = r"figure_([0-9\.]+)_([0-9\.]+)\.png"
    print(image_name)
    match = re.match(pattern, image_name)
    if match:
        x = float(match.group(1))  # Extracts the first number and converts it to float
        q = float(match.group(2))  # Extracts the second number and converts it to float

    if match:
        return x,q
    else:
        raise ValueError(f"Image name {image_name} is not in the expected format.")


def get_images_dict(dir_path="."):
    # get list of all image files in directory
    image_files = [f for f in os.listdir(dir_path) if f.endswith(".png")]
    print(image_files)

    # filter for images with the correct t value
    #t_images = [f for f in image_files if get_image_params(f)[2] == t]

    # create a dict of images with keys (x_B, Q2)
    images_dict = {(get_image_params(img)[0], get_image_params(img)[1]): img for img in image_files}

    return images_dict

def create_image_grid(image_dict, xBbins, Q2bins, dir_path="."):
    # load images
    images = {k: Image.open(os.path.join(dir_path, v)) for k, v in image_dict.items()}

    # assume all images are the same size
    img_width, img_height = next(iter(images.values())).size

    # create new image
    combined = Image.new("RGB", (img_width * len(xBbins), img_height * len(Q2bins)),"white")

    # place images
    for i, xB in enumerate(xBbins):
        for j, Q2 in enumerate(reversed(Q2bins)):
            if (xB, Q2) in images:
                combined.paste(images[(xB, Q2)], (i * img_width, j * img_height))

    return combined

def main(xBbins, Q2bins, in_dir_path=".",out_dir_path="."):
    image_dict = get_images_dict(in_dir_path)

    print(image_dict)
    combined = create_image_grid(image_dict, xBbins, Q2bins, in_dir_path)

    combined.save(os.path.join(out_dir_path, f"combined_t.png"))    
    


def get_GK():
    
    single = False
    if single:
        gk_location = "/mnt/d/GLOBUS/CLAS12/Thesis/the_GK_model_results/cross_section_pi0_10600_big.txt"
        df_GK = pd.read_csv(gk_location, sep='\t', header=0)
    else:
        gk_dir = "/mnt/c/Users/rober/Dropbox/Bobby/Linux/work/CLAS12/mit-clas12-analysis/theana/paragon/analysis/threnody/T_GK_model/july_2023_running/outputs/"
        csv_files = [f for f in os.listdir(gk_dir) if f.endswith('.txt')]

        # Initialize an empty dataframe
        all_data = pd.DataFrame()

        column_names = ['Q2', 'xB', 'mt', 'sigma_T', 'sigma_L', 'sigma_LT', 'sigma_TT', 'W', 'y', 'epsilon', 'gammaa', 'tmin']

        # Iterate over all csv files and append to all_data dataframe
        for file in csv_files:
            data = pd.read_csv(os.path.join(gk_dir, file),sep='\t',names=column_names)
            all_data = all_data.append(data, ignore_index=True)

        # Remove duplicate rows
        df_GK = all_data.drop_duplicates()

        print(df_GK)
        
        df_GK.to_csv("df_gk_saved.csv",index=False)
        #df_GK_calc = pd.read_csv('GK_Model/cross_section_pi0_10600_big.txt', sep='\t', header=0)
        #df_GK_calc = pd.read_csv('cross_section_pi0_575_new_big_1.txt', sep='\t', header=0)
        # Data Structure:
        #     Q2	xB	mt	sigma_T	sigma_L	sigma_LT	sigma_TT	W	y	epsilon	gammaa	tmin
        #  1.75 	 0.225 	 -0.020 	 nan 	 nan 	 -nan 	 nan 	 2.6282355 	 0.0806671 	 0.9961151 	 0.3190776 	 -0.0574737

    #for col in df_GK_calc.columns:
    #    print(col)

    df_GK['sigma_T'] = pd.to_numeric(df_GK["sigma_T"], errors='coerce')
    df_GK['sigma_L'] = pd.to_numeric(df_GK["sigma_L"], errors='coerce')
    df_GK['sigma_LT'] = pd.to_numeric(df_GK["sigma_LT"], errors='coerce')
    df_GK['sigma_TT'] = pd.to_numeric(df_GK["sigma_TT"], errors='coerce')
    df_GK['W'] = pd.to_numeric(df_GK["W"], errors='coerce')
    df_GK = df_GK.query('W > 2')
    print(df_GK)
    print(df_GK.columns.values)
    #multiply mt by -1
    df_GK['mt'] = df_GK['mt']*-1
    #print unique values of 'Q2' 'xB' 'mt' and 'tmin'
    print(df_GK['Q2'].unique())
    print(df_GK['xB'].unique())
    print(df_GK['mt'].unique())
    print(df_GK['tmin'].unique())


    return df_GK

def filter_dataframe(df, xB_true, Q2_true):
    # Check if there are more than one unique values for either "Q2" or "xB"
    if len(df["Q2"].unique()) > 1 or len(df["xB"].unique()) > 1:
        # Calculate absolute differences from the true values
        df["diff_Q2"] = np.abs(df["Q2"] - Q2_true)
        df["diff_xB"] = np.abs(df["xB"] - xB_true)
        
        # Get indices of rows with the smallest difference for "Q2" and "xB"
        idx_Q2_min_diff = df["diff_Q2"].idxmin()
        idx_xB_min_diff = df["diff_xB"].idxmin()

        #get value of Q2 and xB for these indices
        Q2_min_diff = df["Q2"][idx_Q2_min_diff]
        xB_min_diff = df["xB"][idx_xB_min_diff]

        #Get all rows that have these values of Q2 and xB
        df_Q2_min_diff = df[df["Q2"] == Q2_min_diff]
        df_xB_min_diff = df[df["xB"] == xB_min_diff]
        
        #print them
        print("Q2_min_diff")
        print(df_Q2_min_diff)
        print("xB_min_diff")
        print(df_xB_min_diff)

        #Only keep rows that have both values of Q2 and xB
        df = pd.merge(df_Q2_min_diff, df_xB_min_diff)

        # # If indices are the same, keep that row
        # if idx_Q2_min_diff == idx_xB_min_diff:
        #     df = df.loc[[idx_Q2_min_diff]]
        # else:
        #     # If indices are different, keep both rows
        #     df = df.loc[[idx_Q2_min_diff, idx_xB_min_diff]]
        
        # # Remove temporary difference columns
        df = df.drop(columns=["diff_Q2", "diff_xB"])

        print(df)
    return df



df_GK = get_GK()




combine = True


df = pd.read_pickle("t_dep_of_xsec.pkl")

#remove rows where 'A' is negative
df = df[df['A'] > 0]

# Define the data and group it
groups = df.groupby(['xmin', 'qmin'])

#

#drop the first 4 entries in groups

# Create a plot for each group
for i, (name, group) in enumerate(groups):
    #if i < 5:  # Skip the first four entries
    #    continue
    # if group['qmin'].values[0] != 2 or group['xmin'].values[0] != 0.3:
    #      continue
    # print(name)


    # get proper GK data
    mask = (df_GK['Q2'] >= group['qmin'].values[0]) & (df_GK['Q2'] <= group['qmax'].values[0]) & \
       (df_GK['xB'] >= group['xmin'].values[0]) & (df_GK['xB'] <= group['xmax'].values[0])

    filtered_df = df_GK[mask]

    print(filtered_df)

    # Define the values and errors to plot
    values_to_plot = ['A', 'B', 'C', 'c6_tel', 'c6_tt', 'c6_lt']
    errors_to_plot = ['A_uncert', 'B_uncert', 'C_uncert', 'c6_tel_err', 'c6_tt_err', 'c6_lt_err']
    formats = ['o', 'o', 'o', '^', '^', '^']
    x_axes = ['tave', 'tave', 'tave', 'tave_c6', 'tave_c6', 'tave_c6']
    colors = ['black', 'blue', 'red', 'black', 'blue', 'red']

    plt.rcParams["font.size"] = "30"
    plt.figure(figsize=(20,14))
    plt.xlabel('-t [GeV$^2$]')
    plt.ylabel('d$\sigma$/dt [nb/$GeV^2$]')
    if group[['c6_tel', 'c6_tt', 'c6_lt']].notna().any().any():
        # Set the background color to light gray
        plt.gca().set_facecolor('lightgray')
        
    legend_added = {'A': False, 'B': False, 'C': False, 'CLAS6': False, 'GK': False}

    # Create each line on the plot
    for val, err, x_axis, color, fmt in zip(values_to_plot, errors_to_plot, x_axes, colors, formats):
        label = None
        if val in ['A', 'B', 'C'] and not legend_added[val]:
            label = val
            legend_added[val] = True
        elif fmt == '^' and not legend_added['CLAS6']:
            label = 'CLAS6'
            legend_added['CLAS6'] = True
        plt.errorbar(group[x_axis], group[val], yerr=group[err], color=color, label=label, fmt=fmt, markersize=25, elinewidth=4, capsize=10)

    # Plot GK data
    # print the number of unique Q2 and xB values:
    print("HERE IS X AXIS AND VAL")
    #print group name values
    print(name)
    if len(filtered_df['Q2'].unique()) > 0 or len(filtered_df['xB'].unique()) > 0:

        if len(filtered_df['Q2'].unique()) > 1 or len(filtered_df['xB'].unique()) > 1:
            print(filtered_df['Q2'].unique())
            print(filtered_df['xB'].unique())
            print(group['xave'].mean())
            print(group['qave'].mean())

            filtered_df = filter_dataframe(filtered_df, group['xave'].mean(), group['qave'].mean())

        print(filtered_df)
        values_to_plot = [filtered_df['sigma_T']+filtered_df['epsilon']*filtered_df['sigma_L'], filtered_df['sigma_TT'], filtered_df['sigma_LT']]
        x_axes = [filtered_df['mt'], filtered_df['mt'], filtered_df['mt']]
        colors = ['black', 'blue', 'red']

        # for now, spline until get more real datapoints
        # Create a cubic interpolation function


        for val, x_axis, color in zip(values_to_plot, x_axes, colors):
            #f = interp1d(x_axis,val,kind='linear')#, kind='cubic')
            #remove any nan values
            mask2 = np.isnan(val) | np.isnan(x_axis)

            # Use this mask to select only the entries where neither is np.nan
            val_filtered = val[~mask2]
            x_axis_filtered = x_axis[~mask2]

            #remove any nan values from val
            #print x_axis_filtered and val_filtered
            print("HERE IS X AXIS AND VAL")
            #print group name values
            print(name)
            print(x_axis_filtered)
            print(val_filtered)
            # if there are no values, skip
            if len(x_axis_filtered) == 0:
                continue

            # if there is only one value, skip
            if len(x_axis_filtered) == 1:
                continue

            # if there are only 2 values, can't spline
            if len(x_axis_filtered) == 2:
                continue

            X_Y_Spline = make_interp_spline(x_axis_filtered, val_filtered)
            #get max and min of x_axis
            x_max = np.max(x_axis_filtered)
            x_min = np.min(x_axis_filtered)
            xnew = np.linspace(x_min, x_max,100, endpoint=True)
            #ynew = f(xnew)  # use interpolation function returned by `interp1d`
            ynew = X_Y_Spline(xnew)
            print(x_min,x_max)
            print(ynew)

            label = None
            if not legend_added['GK']:
                label = 'GK model'
                legend_added['GK'] = True
            plt.plot(xnew, ynew, color=color, label=label,linewidth=5)

    plt.title(r'$\langle x_B \rangle$'+'={:.2f}'.format(group['xave'].mean())+r', $\langle Q^2 \rangle$'+'={:.2f}'.format(group['qave'].mean())+' GeV$^2$')
    legend_elements = [Line2D([0], [0], marker='o', color='black', label='$\sigma_T+\epsilon \sigma_L$', markersize=15,linestyle='None'),
                   Line2D([0], [0], marker='o', color='blue', label=r'$\sigma_{TT}$', markersize=15,linestyle='None'),
                   Line2D([0], [0], marker='o', color='red', label=r'$\sigma_{LT}$', markersize=15,linestyle='None'),
                   #no line on legend
                   Line2D([0], [0], marker='^', color='black', label='CLAS6', markersize=15,linestyle='None'),
                   Line2D([0], [0], color='black', label='GK model', markersize=15)]

    # Create the figure and plot.

    # Add your custom legend.
    plt.legend(handles=legend_elements, loc='upper right')
    #plt.show()

    #set x axis range to (0.1,2.0)
    plt.xlim(0.1,2.0)

    #sys.exit()

    plt.savefig("tdep_test/"+f'figure_{name[0]}_{name[1]}.png')
    plt.close()

if combine:
    main(fs.xBbins, fs.Q2bins, in_dir_path="tdep_test/",out_dir_path="tdep_combined_plot/")





    #labels = ['$\sigma_T+\epsilon \sigma_L$', r'$\sigma_{TT}$', r'$\sigma_{LT}$', 'CLAS6 $\sigma_T+\epsilon \sigma_L$', r'CLAS6 $\sigma_{TT}$', r'CLAS6 $\sigma_{LT}$']
