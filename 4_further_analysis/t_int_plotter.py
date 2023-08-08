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

    combined.save(os.path.join(out_dir_path, f"combined_t_int.png"))    
    



individual = True
combine = True


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the data
if individual:
    df = pd.read_pickle("t_int_of_xsec_unfolded.pkl")

    # keep only where xmin and qmin are in the bins we want - (0.38, 3.0)
    #df = df[(df['xmin'] >= 0.38) & (df['qmin'] >= 3.0)]
    #df = df[(df['xmin'] <= 0.438) & (df['qmin'] <= 3.20)]

    # only keep when int_value greater than 0
    df = df[df['int_value'] > 0]
    
    # Define the fit function
    def fit_func(t, A, B):
        return A * np.exp(-B * t)

    # Create an empty dataframe to store results
    results = pd.DataFrame(columns=['xave', 'qave', 'A', 'B', 'A_error', 'B_error'])

    # Group by 'xmin' and 'qmin'
    groups = df.groupby(['xmin', 'qmin'])

    for i, (name, group) in enumerate(groups):
        if len(group) <= 3:
            continue

        # Plotting
      
        # Fitting
        print(group['tave'][1:],         group['int_value'][1:])#, sigma=group['int_err_top'][1:])
        popt, pcov = curve_fit(fit_func, group['tave'][1:],         group['int_value'][1:], sigma=group['int_err_top'][1:], absolute_sigma=True)
        popt_top, pcov_top = curve_fit(fit_func, group['tave'][1:], group['int_value'][1:]+group['int_err_top'][1:], sigma=group['int_err_top'][1:], absolute_sigma=True)
        popt_bot, pcov_bot = curve_fit(fit_func, group['tave'][1:], group['int_value'][1:]-group['int_err_bot'][1:], sigma=group['int_err_bot'][1:], absolute_sigma=True)
        

        A, B = popt
        A_top, B_top = popt_top
        A_bot, B_bot = popt_bot
        A_error, B_error = np.sqrt(np.diag(pcov))

        if B<0:
            continue
        
        plt.rcParams["font.size"] = "30"
        plt.figure(figsize=(20,14))
        plt.xlabel('-t [GeV$^2$]')
        #plt.ylabel('d$\sigma$/dt [nb/$GeV^2$]')

        plt.errorbar(group['tave'], group['int_value'], yerr=[group['int_err_top'],group['int_err_bot']], fmt='k.', markersize=20, 
                    elinewidth=4, capsize=10, capthick=4)#, label='Integrated')
        

        #get minimum and maximum t values
        t_min = group['tave'].min()
        t_max = group['tave'].max()
        t_vals = np.linspace(t_min,t_max, 1000)  # or use the t range you prefer
        #make line thickness 3
        # Calculate the upper and lower bound for the fit
        y_upper = fit_func(t_vals, A_top, B_top)
        y_lower = fit_func(t_vals, A_bot, B_bot)
        
        plt.fill_between(t_vals, y_lower, y_upper, color='red', alpha=0.2, label='Fit error band')
        
        plt.plot(t_vals, fit_func(t_vals, *popt), 'r-', label='Fit: B={:.2f} $\\pm$ {:.2f}'.format(B, B_error), linewidth=3)
        plt.title(r'$\langle x_B \rangle$'+'={:.2f}'.format(group['xave'].mean())+r', $\langle Q^2 \rangle$'+'={:.2f}'.format(group['qave'].mean())+' GeV$^2$',
                  y=0.94,x=0.4,bbox=dict(facecolor='white', alpha=1, edgecolor='none'))

        # Configuration and show plot
        plt.legend()
        plt.yscale('log')
        plt.xlim(0.1,1.7)
        plt.ylim(.1,800)
        print(name)
        #plt.show()
        plt.savefig("tdep_integrated/"+f'figure_{name[0]}_{name[1]}.png')
        plt.close()

        A_err_total = A_top-A_bot
        B_err_total = B_top-B_bot
        # Append results to the dataframe
        results = results.append({
            'xave': group['xave'].mean(),  # Assuming you want the average xave and qave for each group
            'qave': group['qave'].mean(),
            'A': A,
            'B': B,
            'A_error': A_err_total,
            'B_error': B_err_total
        }, ignore_index=True)

    print(results)
    results.to_pickle("t_int_of_xsec_with_fit.pkl")

if combine:
    main(fs.xBbins, fs.Q2bins, in_dir_path="tdep_integrated/",out_dir_path="tdep_integrated_combined")





    #labels = ['$\sigma_T+\epsilon \sigma_L$', r'$\sigma_{TT}$', r'$\sigma_{LT}$', 'CLAS6 $\sigma_T+\epsilon \sigma_L$', r'CLAS6 $\sigma_{TT}$', r'CLAS6 $\sigma_{LT}$']
