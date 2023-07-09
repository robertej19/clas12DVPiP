from utils import const, physics, filestruct
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pandas as pd
import argparse
from itertools import combinations


df = pd.read_pickle("lund_output.pkl")
print(df)



#Main data
fs = filestruct.fs()
data_path = fs.data_path

config = "outbend"

if config == "inbend":
    #inbending configuration
    # inbending experimental data
    exp = [data_path + "exp_inbend/" + "20220511_f18_in_combined_157.root",]

        # No background merging
    gen_no_bkgrnd = [data_path+ "gen_inbend_norad/" +"norad_10000_20230123_0229_Fall_2018_Inbending_None_gen.root",
                    data_path+ "gen_inbend_norad/" +"norad_10000_20230704_1755_Fall_2018_Inbending_None_gen.root"]

    recon_no_bkgrnd = [data_path+ "rec_inbend_norad/" +"norad_10000_20230123_0229_Fall_2018_Inbending_None_recon.root",
                    data_path+ "rec_inbend_norad/" +"norad_10000_20230704_1755_Fall_2018_Inbending_None_recon.root"]


        # Nominal 50 nA background merging
    gen_nominal = [data_path+ "gen_inbend_norad/" +"norad_10000_20230703_1814_Fall_2018_Inbending_50nA_gen.root",
                    data_path+ "gen_inbend_norad/" +"norad_10000_20230704_1143_Fall_2018_Inbending_50nA_gen_high_cutoffs.root",
                    data_path+ "gen_inbend_norad/" +"norad_10000_20230705_1041_Fall_2018_Inbending_50nA_gen.root"]
    
    recon_nominal = [data_path+ "rec_inbend_norad/" +"norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon.root",
                    data_path+ "rec_inbend_norad/" +"norad_10000_20230704_1143_Fall_2018_Inbending_50nA_recon_high_cuts.root",
                    data_path+ "rec_inbend_norad/" +"norad_10000_20230705_1041_Fall_2018_Inbending_50nA_recon.root"]

        # Other configurations
    gen_other = [data_path+ "gen_inbend_norad/" +"norad_10000_20230705_1043_Fall_2018_Inbending_45nA_gen.root",
                    data_path+ "gen_inbend_norad/" +"norad_10000_20230705_1046_Fall_2018_Inbending_55nA_gen.root"]
    
    recon_other = [data_path+ "rec_inbend_norad/" +"norad_10000_20230705_1043_Fall_2018_Inbending_45nA_recon.root",
                    data_path+ "rec_inbend_norad/" +"norad_10000_20230705_1046_Fall_2018_Inbending_55nA_recon.root"]

        # radiative generator, 50 nA background merging

    gen_rad = [data_path+ "gen_inbend_rad/" +"rad_10000_20230126_1645_Fall_2018_Inbending_50nA_gen.root",
            data_path+ "gen_inbend_rad/" +"rad_10000_20230203_0905_Fall_2018_Inbending_50nA_gen.root",
            data_path+ "gen_inbend_rad/" +"rad_10000_20230228_1359_Fall_2018_Inbending_50nA_gen.root"]
    
    recon_rad = [data_path+ "rec_inbend_rad/" +"rad_10000_20230126_1645_Fall_2018_Inbending_50nA_recon.root",
            data_path+ "rec_inbend_rad/" +"rad_10000_20230203_0905_Fall_2018_Inbending_50nA_recon.root",
            data_path+ "rec_inbend_rad/" +"rad_10000_20230228_1359_Fall_2018_Inbending_50nA_recon.root"]
    
elif config == "outbend":
    # outbending configuration
    # outbending experimental data
    exp = [data_path + "exp_outbend/" + "20220511_f18_out_combined_171.root",]

        # No background merging 
    gen_no_bkgrnd = [data_path+ "gen_outbend_norad/" +"norad_10000_20230123_0229_Fall_2018_Outbending_100_None_gen.root",
                    data_path+ "gen_outbend_norad/" +"norad_10000_20230704_1755_Fall_2018_Outbending_100_None_gen.root"]

    recon_no_bkgrnd = [data_path+ "rec_outbend_norad/" +"norad_10000_20230123_0229_Fall_2018_Outbending_100_None_recon.root",
                    data_path+ "rec_outbend_norad/" +"norad_10000_20230704_1755_Fall_2018_Outbending_100_None_recon.root"]
    
        # Nominal 50 nA background merging
    gen_nominal = [data_path+ "gen_outbend_norad/" +"norad_10000_20230703_1814_Fall_2018_Outbending_100_50nA_gen.root",
                    data_path+ "gen_outbend_norad/" +"norad_10000_20230704_1143_Fall_2018_Outbending_100_50nA_gen_high_cutoffs.root",
                    data_path+ "gen_outbend_norad/" +"norad_10000_20230705_1050_Fall_2018_Outbending_100_50nA_gen.root",
                    data_path+ "gen_outbend_norad/" +"norad_10000_20230705_1051_Fall_2018_Outbending_100_50nA_gen.root",
                    data_path+ "gen_outbend_norad/" +"norad_10000_20230705_1055_Fall_2018_Outbending_100_50nA_gen.root"]
    
    recon_nominal = [data_path+ "rec_outbend_norad/" +"norad_10000_20230703_1814_Fall_2018_Outbending_100_50nA_recon.root",
                     data_path+ "rec_outbend_norad/" +"norad_10000_20230704_1143_Fall_2018_Outbending_100_50nA_recon_high_cuts.root",
                    data_path+ "rec_outbend_norad/" +"norad_10000_20230705_1050_Fall_2018_Outbending_100_50nA_recon.root",
                    data_path+ "rec_outbend_norad/" +"norad_10000_20230705_1051_Fall_2018_Outbending_100_50nA_recon.root",
                    data_path+ "rec_outbend_norad/" +"norad_10000_20230705_1055_Fall_2018_Outbending_100_50nA_recon.root"]
    
        # Other configurations
    gen_other = [data_path+ "gen_outbend_norad/" +"norad_10000_20230704_1644_Fall_2018_Outbending_101_40nA_gen.root",
                    data_path+ "gen_outbend_norad/" +"norad_10000_20230705_1053_Fall_2018_Outbending_100_40nA_gen.root"]
    
    recon_other = [data_path+ "rec_outbend_norad/" +"norad_10000_20230704_1644_Fall_2018_Outbending_101_40nA_recon.root",
                    data_path+ "rec_outbend_norad/" +"norad_10000_20230705_1053_Fall_2018_Outbending_100_40nA_recon.root"]
    
        # radiative generator, 50 nA background merging
    gen_rad = [data_path+ "gen_outbend_rad/" +"rad_10000_20230126_1645_Fall_2018_Outbending_100_50nA_gen.root",
            data_path+ "gen_outbend_rad/" +"rad_10000_20230203_0905_Fall_2018_Outbending_100_50nA_gen.root",
            data_path+ "gen_outbend_rad/" +"rad_10000_20230228_1359_Fall_2018_Outbending_100_50nA_gen.root"]
    
    recon_rad = [data_path+ "rec_outbend_rad/" +"rad_10000_20230126_1645_Fall_2018_Outbending_100_50nA_recon.root",
                data_path+ "rec_outbend_rad/" +"rad_10000_20230203_0905_Fall_2018_Outbending_100_50nA_recon.root",
                data_path+ "rec_outbend_rad/" +"rad_10000_20230228_1359_Fall_2018_Outbending_100_50nA_recon.root"]
    
else:
    print("ERROR: Invalid configuration specified!")
    exit(1)


datasets = [exp, gen_no_bkgrnd, recon_no_bkgrnd, gen_nominal, recon_nominal, gen_other, recon_other, gen_rad, recon_rad]

count = 0

for ds in datasets:
    for f in ds:
        print(f)
        print( os.path.getsize(f) )
        print(count)
        count += 1

sys.exit()


def main(input_args):
    # 1. Convert root
    convert_root(input_args)

    # 2. Apply exclusivity cuts
    apply_exclusivity_cuts(input_args)

    # 3. Bin events
    bin_events(input_args)

    # 4. Calculate cross section
    calculate_cross_section(input_args)

    # 5. Plot results
    plot_results(input_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to perform data processing')

    # Add your arguments here. For example:
    parser.add_argument('-f', '--file', type=str, help='Input file path', required=True)
    parser.add_argument('-o', '--output', type=str, help='Output directory path', required=True)
    parser.add_argument('-p', '--parameter', type=float, help='Some additional parameter', required=False)

    args = parser.parse_args()

    main(args)
