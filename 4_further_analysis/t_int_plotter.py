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
    
    # cut the bottom 5% of every image
    cut_height = int(0.1075* img_height)
    cut_height_end = int(0.895* img_height)

    cropped_height = cut_height_end - cut_height
    cut_width = int(0.070* img_width)
    cut_end_width = int(0.901* img_width)

    cropped_width = cut_end_width - cut_width

    # create new image
    combined = Image.new("RGB", (cropped_width * len(xBbins), cropped_height * len(Q2bins)), "white")

    # place images
    for i, xB in enumerate(xBbins):
        for j, Q2 in enumerate(reversed(Q2bins)):
            if (xB, Q2) in images:
                cropped_img = images[(xB, Q2)].crop((cut_width, cut_height, cut_end_width, cut_height_end))
                # show the cropped_img
                #cropped_img.show()
                #sys.exit()
                combined.paste(cropped_img, (i * cropped_width, j * cropped_height))

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
        # if len(df)=0, just take the longer df instead
        if len(df) == 0:
            if len(df_Q2_min_diff) > len(df_xB_min_diff):
                df = df_Q2_min_diff
            else:
                df = df_xB_min_diff
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



individual = True
combine = False


if individual:
    df = pd.read_pickle("t_dep_of_xsec_unfolded.pkl")

    #keep only t values less than 0.3
    df = df[df['tave'] < 0.3]

    # Compute actual error values
    # 
    df['a_err_hi'] = df['A_uncert_top']-df['A']
    df['a_err_lo'] = df['A']-df['A_uncert_bot']
    df['b_err_hi'] = df['B_uncert_top']-df['B']
    df['b_err_lo'] = df['B']-df['B_uncert_bot']
    df['c_err_hi'] = df['C_uncert_top']-df['C']
    df['c_err_lo'] = df['C']-df['C_uncert_bot']

    df = df[~((np.abs(df['B']) > 500))]

    #remove any rows where qmin >4.4 and abs(B)>50 at the same time
    df = df[~((df['qmin'] > 4.4) & (np.abs(df['B']) > 50))]
    df = df[~((df['qmin'] > 3.6) & (np.abs(df['B']) > 100))]

    #df = df[df['A'] > 0]

    #get df where qmin = 3.5 and xmin = 0.48
    #df = df[(df['qmin'] == 3.5) & (df['xmin'] == 0.48)]
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
        
        # Define the values and errors to plot
        values_to_plot = ['A', 'B', 'C', 'c6_tel', 'c6_tt', 'c6_lt']
        #errors_to_plot = [[group['a_err_lo'],group['a_err_hi']],
        #                    [group['b_err_lo'],group['b_err_hi']],
        #                    [group['c_err_lo'],group['c_err_hi']],
        #                    group['c6_tel_err'], group['c6_tt_err'], group['c6_lt_err']]
        errors_to_plot = [[group['a_err_lo'],group['a_err_hi']],
                            [group['b_err_lo'],group['b_err_hi']],
                            [group['c_err_lo'],group['c_err_hi']],
                            0,0,0]

        formats = ['.', '.', '.', '^', '^', '^']
        x_axes = ['tave', 'tave', 'tave', 'tave_c6', 'tave_c6', 'tave_c6']
        colors = ['black', 'blue', 'red', 'black', 'blue', 'red']
        markerfacecolors = ['black', 'blue', 'red', 'green', 'green', 'green']
        x_shifts = [0, 0, 0, 0.025, 0.025, 0.025]
        #error_colors = ['black', 'blue', 'red', 'green', 'green', 'green']
        #cap_colors = ['black', 'blue', 'red', 'black', 'blue', 'red']
        error_colors = ['black', 'blue', 'red', 'none', 'none', 'none']
        cap_colors = ['black', 'blue', 'red', 'none', 'none', 'none']


        legend_added = {'A': False, 'B': False, 'C': False, 'CLAS6': False, 'GK': False}

        plt.rcParams["font.size"] = "30"
        plt.figure(figsize=(20,14))
        plt.xlabel('-t [GeV$^2$]')
        plt.ylabel('d$\sigma$/dt [nb/$GeV^2$]')
        if group[['c6_tel', 'c6_tt', 'c6_lt']].notna().any().any():
            # Set the background color to light gray
            pass
            #plt.gca().set_facecolor('lightgray')
            #legend_added['CLAS6'] = True
            

        # Create each line on the plot
        for count, (val, err, x_axis, color,mfcolor, fmt,shift,error_color,cap_color) in enumerate(zip(values_to_plot, errors_to_plot, x_axes, colors, 
                                                                                markerfacecolors,formats,x_shifts,error_colors,cap_colors)):
            label = None
            if count<3: #optional, for turning off plotting of CLAS6
                count+=1
                if val in ['A', 'B', 'C'] and not legend_added[val]:
                    label = val
                    legend_added[val] = True
                elif fmt == '^' and not legend_added['CLAS6']:
                    label = 'CLAS6'
                
                (_, caps, _) = plt.errorbar(group[x_axis]-shift, group[val], yerr=err, color=color,  
                             markerfacecolor=mfcolor, label=label, fmt=fmt, 
                             markersize=15, elinewidth=4, capsize=10,
                             ecolor=error_color)
                print(group[val],err)
                for cap in caps:
                    cap.set_color(cap_color)


        plt.title(r'$\langle x_B \rangle$'+'={:.2f}'.format(group['xave'].mean())+r', $\langle Q^2 \rangle$'+'={:.2f}'.format(group['qave'].mean())+' GeV$^2$',
                  y=0.94,x=0.45,bbox=dict(facecolor='white', alpha=1, edgecolor='none'))

        legend_elements = [Line2D([0], [0], marker='o', color='black', label='$\sigma_T+\epsilon \sigma_L$', markersize=15,linestyle='None'),
                    Line2D([0], [0], marker='o', color='blue', label=r'$\sigma_{TT}$', markersize=15,linestyle='None'),
                    Line2D([0], [0], marker='o', color='red', label=r'$\sigma_{LT}$', markersize=15,linestyle='None'),
                    #no line on legend
                    Line2D([0], [0], marker='^', color='black', label='CLAS6', markersize=15,linestyle='None'),
                    Line2D([0], [0], color='black', label='GK model', markersize=15)]

        # Create the figure and plot.

        #get the values of legend_added
        legend_added_list = list(legend_added.values())
        print(legend_added_list)

        # Filter the legend elements based on the boolean values in legend_added_list
        filtered_legend_elements = [element for element, is_added in zip(legend_elements, legend_added_list) if is_added]

        # Add your custom legend.
        plt.legend(handles=filtered_legend_elements, loc='upper right')
        #plt.show()

        #set x axis range to (0.1,2.0)
        plt.xlim(0.1,2.0)

        #plt.show()
        #sys.exit()

        #plt.show()
        plt.savefig("tdep_integrated/"+f'figure_{name[0]}_{name[1]}.png')
        plt.close()

if combine:
    main(fs.xBbins, fs.Q2bins, in_dir_path="tdep_integrated/",out_dir_path="tdep_integrated_combined")





    #labels = ['$\sigma_T+\epsilon \sigma_L$', r'$\sigma_{TT}$', r'$\sigma_{LT}$', 'CLAS6 $\sigma_T+\epsilon \sigma_L$', r'CLAS6 $\sigma_{TT}$', r'CLAS6 $\sigma_{LT}$']
