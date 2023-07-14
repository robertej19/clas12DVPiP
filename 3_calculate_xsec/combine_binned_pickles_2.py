import pandas as pd
import numpy as np
import os, sys
from icecream import ic

from utils import filestruct

#pd.set_option('mode.chained_assignment', None)

fs = filestruct.fs()


# binned_outb_exp = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/outb/exp/final_f18_outb_exp_binned_with_area.pkl"
# binned_inb_exp = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/exp/final_f18_inb_exp_binned_with_area.pkl"

binned_exp = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/exp/singles_t2/"
exp_out = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/exp/"
binned_rec = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/rec_rad/singles_t2/"
rec_out = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/rec_rad/"
binned_gen_rad = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/gen_rad/singles/"
binned_gen = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/gen/singles"

gen_out = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/gen_rad/"


gen_45na_inb = "10000_20230625_2138"
gen_55na_inb = "10000_20230626_0753"
gen_nominal_1 = "10000_20230624_1247"
gen_nominal_2 = "10000_20230625_1911"


GEN = True
REC = False
if GEN:
    prefix = "Gen"
    out_prefix = "gen_"
    gen_sub_name = gen_nominal_2
    directory = binned_gen
    out = gen_out
    filename = "final_f18_gen_nominal_2_inb_binned.pkl"
elif REC:
    prefix = ""
    out_prefix = "rec_"
    directory = binned_rec
    out = rec_out
    filename = "final_f18_inb_rec_binned_t2.pkl"
else:
    prefix = ""
    out_prefix = "exp_"
    directory = binned_exp
    out = exp_out
    filename = "final_f18_inb_exp_binned_no_area.pkl"



    # files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

# for file in files:
#         print("Binning on {}".format(input_dir+file))
#         df = pd.read_pickle(input_dir+file)
#         print(df.columns.values)




# Get a list of all the files in the directory

file_list = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

#only keep 2 entries from file list:
#file_list = file_list#[:2]

# only keep the files with a string matching gen_45na_inb
file_list = [f for f in file_list if gen_sub_name in f]
print(file_list)


# Initialize DataFrame to store total counts and sums for weighted averages
counts_total = None
sum_weights = None
weighted_sums = None

index = 0
# Loop over each file
for file in file_list:
    # Load the DataFrame from the file
    df = pd.read_pickle(os.path.join(directory, file))#.head(3)
    # Set the index
    print(df)
    df.set_index(['tmin', 'pmin', 'xmin', 'qmin','tmax','pmax','xmax','qmax'], inplace=True)
    #print(df)
    # Calculate the weights for the current dataframe
    weights = df[prefix+'counts']

    if counts_total is None:
        #print("here")
        # If this is the first dataframe, initialize the totals
        counts_total = weights
        weighted_sums = df[['tave', 'pave', 'xave', 'qave', 'yave']].multiply(weights, axis=0)
    else:
        #print("there")
        # If this is not the first dataframe, add to the totals
        counts_total += weights
        weighted_sums += df[['tave', 'pave', 'xave', 'qave', 'yave']].multiply(weights, axis=0)
    
    index += 1


print(counts_total)
print(weighted_sums)
averages = weighted_sums.divide(counts_total, axis=0)
#Calculate the final weighted averages
# 
# # Add the total counts to the DataFrame
averages['counts'] = counts_total
averages.columns = averages.add_prefix(out_prefix).columns

# # The DataFrame "averages" now contains the combined data
df_combined = averages
print(df_combined)
df_combined.to_pickle(out+filename)

#keep only hwere tmin < 0.1 pmin <36 xmin <0.12 qmin < 1.4
# df_combined = df_combined[(df_combined.index.get_level_values('tmin') < 0.1) & (df_combined.index.get_level_values('pmin') == 36) & (df_combined.index.get_level_values('xmin') < 0.12) & (df_combined.index.get_level_values('qmin') < 1.4)]
# print(df_combined)

# print(df_combined)