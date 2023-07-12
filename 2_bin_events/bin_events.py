import uproot
import pandas as pd
import numpy as np
import argparse
import os, sys
from icecream import ic
import matplotlib.pyplot as plt
from utils import filestruct
pd.set_option('mode.chained_assignment', None)


fs = filestruct.fs()

def bin_df(df,df_type="real"):
    prefix = "Gen" if df_type=="Gen" else ""


    df_np = df[["{}Q2".format(prefix),"{}xB".format(prefix),"{}t1".format(prefix),"{}phi1".format(prefix),"{}y".format(prefix)]].copy().to_numpy()

    num_cols = df_np.shape[1]
    blank_bin_edges = [-1000,1000]
    initalized_bin_edges = [blank_bin_edges]*num_cols

    q2_bin_edges,xb_bin_edges, t1_bin_edges, phi1_bin_edges = fs.Q2bins, fs.xBbins, fs.tbins, fs.phibins
    
    # Enable for rapid testing
    # xb_bin_edges = [0.1,0.4,0.8]
    # q2_bin_edges = [1,5,11]
    # t1_bin_edges =  [0.09,0.6,2]
    # phi1_bin_edges = [0, 180, 360]


    #Get number of columns
    num_cols = df_np.shape[1]


    initalized = [blank_bin_edges]*num_cols

    initalized[0] = q2_bin_edges
    initalized[1] = xb_bin_edges
    initalized[2] = t1_bin_edges
    initalized[3] = phi1_bin_edges


    number_of_counts_bin_values, edges = np.histogramdd(df_np, bins=initalized)

    weighted_q2_values, edges = np.histogramdd(df_np, bins=initalized,weights=df_np[:,0])
    weighted_xB_values, edges = np.histogramdd(df_np, bins=initalized,weights=df_np[:,1])
    weighted_t1_values, edges = np.histogramdd(df_np, bins=initalized,weights=df_np[:,2])
    weighted_phi1_values, edges = np.histogramdd(df_np, bins=initalized,weights=df_np[:,3])
    weighted_y_values, edges = np.histogramdd(df_np, bins=initalized,weights=df_np[:,4])


    q2_bin_averages = np.divide(weighted_q2_values,number_of_counts_bin_values).reshape(-1,1)
    xb_bin_averages = np.divide(weighted_xB_values,number_of_counts_bin_values).reshape(-1,1)
    t1_bin_averages = np.divide(weighted_t1_values,number_of_counts_bin_values).reshape(-1,1)
    phi1_bin_averages = np.divide(weighted_phi1_values,number_of_counts_bin_values).reshape(-1,1)
    y_bin_averages = np.divide(weighted_y_values,number_of_counts_bin_values).reshape(-1,1)


    number_of_counts_bin_values_reshaped = number_of_counts_bin_values.reshape(-1,1)



    q2_min = edges[0][:-1]
    q2_max = edges[0][1:]
    xb_min = edges[1][:-1]
    xb_max = edges[1][1:]
    t1_min = edges[2][:-1]
    t1_max = edges[2][1:]
    phi1_min = edges[3][:-1]
    phi1_max = edges[3][1:]


    num_of_binning_vars = 4


    all_min = np.array(np.meshgrid(t1_min,phi1_min,xb_min,q2_min)).T.reshape(-1,num_of_binning_vars)
    all_max = np.array(np.meshgrid(t1_max,phi1_max,xb_max,q2_max)).T.reshape(-1,num_of_binning_vars)
    all_together_now = np.concatenate((all_min, all_max), axis=1)


    all_together_now1 = np.concatenate((all_together_now,   t1_bin_averages, phi1_bin_averages,xb_bin_averages, q2_bin_averages, y_bin_averages,number_of_counts_bin_values_reshaped), axis=1)


    #calculate p_standard_dev
    

    # df_minibin = pd.DataFrame(all_together_now1, columns = ['qmin','xmin','tmin','pmin','qmax','xmax','tmax','pmax','qave','yave','xave','tave','pave',str(prefix)+'counts'])
    df_minibin = pd.DataFrame(all_together_now1, columns = ['tmin','pmin','xmin','qmin','tmax','pmax','xmax','qmax','tave','pave','xave','qave','yave',str(prefix)+'counts'])

    #create combination columns for ease of weighting in future steps. Currently only need for gen but might need in future
    df_minibin.loc[:, 'tave_weighted'] = df_minibin['tave']*df_minibin[str(prefix)+'counts']
    df_minibin.loc[:, 'pave_weighted'] = df_minibin['pave']*df_minibin[str(prefix)+'counts']
    df_minibin.loc[:, 'xave_weighted'] = df_minibin['xave']*df_minibin[str(prefix)+'counts']
    df_minibin.loc[:, 'qave_weighted'] = df_minibin['qave']*df_minibin[str(prefix)+'counts']
    df_minibin.loc[:, 'yave_weighted'] = df_minibin['yave']*df_minibin[str(prefix)+'counts']


    ########## FIX THIS - just include a logic check ##############
    #print("Total number of binned events: {}".format(df_minibin[prefix+'counts'].sum()))
    #print("Total number of original events: {}".format(total_num))

    columns_to_drop = [col for col in df_minibin.columns if '_weighted' in col]

    # Drop these columns from the DataFrame
    df = df_minibin.drop(columns=columns_to_drop)
    return df


if __name__ == "__main__":

    type = "exp"

    if type == "rec":
        input_dir = fs.inb_norad_rec_dvpip_dir
        output_dir = fs.inb_norad_rec_binned_dir

        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        for file in files:
                print("Binning on {}".format(input_dir+file))
                outfile_name = output_dir+"binned_"+file
                print("Saving to {}".format(outfile_name))
                df = pd.read_pickle(input_dir+file)
                print(len(df))
                df_binned = bin_df(df,df_type="rec")
                print(df_binned)
                df_binned.to_pickle(outfile_name)
                #print sum of counts as a check
                print(df_binned.sum(axis=0))

    elif type == "gen":
        # Define the directories
        rec_dir = "/mnt/d/GLOBUS/CLAS12/Thesis/1_potential_dvpip_events/inb/rec/"
        gen_dir = "/mnt/d/GLOBUS/CLAS12/Thesis/1_potential_dvpip_events/gen/"

        print('mapping')
        # Define the mapping DataFrame
        mapping = pd.DataFrame({
            "rec": [
                "10000_20230704_1143", "10000_20230704_1143", "10000_20230703_1814",
                "10000_20230703_1814", "10000_20230704_1755", "10000_20230704_1755",
                "10000_20230705_1057", "10000_20230705_1057", "10000_20230705_1035",
                "10000_20230705_1048", "10000_20230705_1037", "10000_20230705_1050",
                "10000_20230705_1041", "10000_20230705_1051", "10000_20230705_1043",
                "10000_20230705_1053", "10000_20230705_1046", "10000_20230705_1055",
                "10000_20230704_1644"
            ],
            "gen": [
                "10000_20230624_1457", "10000_20230624_1457", "10000_20230624_1247",
                "10000_20230624_1247", "10000_20230624_1629", "10000_20230624_1842",
                "10000_20230624_2100", "10000_20230625_0723", "10000_20230625_0932",
                "10000_20230625_1838", "10000_20230625_1555", "10000_20230625_1148",
                "10000_20230625_1911", "10000_20230625_2006", "10000_20230625_2138",
                "10000_20230626_0717", "10000_20230626_0753", "10000_20230626_1125",
                "10000_20230626_1337"
            ]
        })

        # Get the list of directories in 'gen'
        gen_dirs = next(os.walk(gen_dir))[1]

        # Get the list of filenames in 'rec'
        rec_files = next(os.walk(rec_dir))[2]

        list_of_gen_dirs_to_process = []

        # Iterate over each directory in 'gen'
        for gen_subdir in gen_dirs:
            # Check if the name of the directory contains a string that is in the 'gen' column in the mapping table
            if any(gen_name in gen_subdir for gen_name in mapping['gen']):
                # Get the corresponding 'rec' name(s) for this 'gen' name
                gen_name_reduced = gen_subdir[5:] #remove lund_
                rec_names = mapping.loc[mapping['gen'] == gen_name_reduced, 'rec'].values
                # Check if the corresponding 'rec' string is found in any of the filenames in the 'rec' directory
                if any(rec_name in rec_file for rec_name in rec_names for rec_file in rec_files):
                    # If yes, print the name of the directory
                    list_of_gen_dirs_to_process.append(gen_subdir)

        print(rec_files)
        print(list_of_gen_dirs_to_process)

        binned_gen_out_dir_base = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/"

        # for each gen directory to process, get a list of files in that directory
        for gen_subdir in list_of_gen_dirs_to_process:
            files = [f for f in os.listdir(gen_dir+gen_subdir) if os.path.isfile(os.path.join(gen_dir+gen_subdir, f))]
            print(files)
            # for each file, bin it
            for file in files:
                print("Binning on {}".format(gen_dir+gen_subdir+"/"+file))

                outfile_name = binned_gen_out_dir_base+"gen/binned_"+file

                print("Saving to {}".format(outfile_name))
                df = pd.read_pickle(gen_dir+gen_subdir+"/"+file)
                print(len(df))
                df_binned = bin_df(df,df_type="Gen")
                print(df_binned)
                df_binned.to_pickle(outfile_name)
                #print sum of counts as a check
                print(df_binned.sum(axis=0))

                outfile_name = binned_gen_out_dir_base+"gen_wq2_cut/binned_"+file

                #Now repeat, with including a cut on GenQ2 and GenW
                df = df[(df["GenQ2"] > .95) & (df["GenW"] > 1.95)]
                print(len(df))
                df_binned = bin_df(df,df_type="Gen")
                print(df_binned)
                df_binned.to_pickle(outfile_name)
                #print sum of counts as a check
                print(df_binned.sum(axis=0))

    elif type == "exp":


    else:
        pass




        # input_dir = fs.inb_norad_rec_epgg_dir
        # output_dir = fs.inb_norad_rec_binned_dir

        # files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        # for file in files:
        #         #restrict events to have GenW>2.0 and Q2>1.0
                                

        #         print("Binning on {}".format(input_dir+file))
        #         outfile_name = output_dir+"binned_"+file
        #         print("Saving to {}".format(outfile_name))
        #         df = pd.read_pickle(input_dir+file)
        #         print(len(df))
        #         df_binned = bin_df(df,df_type="Gen")
        #         print(df_binned)
        #         df_binned.to_pickle(outfile_name)
        #         #print sum of counts as a check
        #         print(df_binned.sum(axis=0))

    # binned_outb_exp = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/outb/exp/final_f18_outb_exp_binned_with_area.pkl"
    # binned_inb_exp = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/exp/final_f18_inb_exp_binned_with_area.pkl"
    # rec_out_inb = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/rec/"
    # gen_out_inb = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/gen/"
    # rec_out_outb = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/outb/rec/"
    # gen_out_outb = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/outb/gen/"

    # inbending_data = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/inb/exp/final_inbending_exclusive.pkl"
    # outbending_data = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/outb/exp/final_outbending_exclusive.pkl"
    # rec_in = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/inb/rec/norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon_fid_corr_smear.pkl"
    # rec_out = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/outb/rec/norad_10000_20230703_1814_Fall_2018_Outbending_100_50nA_recon_fid_corr_smear.pkl"


    # df = pd.read_pickle(inbending_data)
    # print(df)
    # df_binned = bin_df(df,df_type="real")
    # print(df_binned)
    # print(df_binned.sum(axis=0))
    # df_binned.to_pickle("sample_binned_inb.pkl")
    # #df_binned.to_pickle("binned_dvpip/f18_bkmrg_in_dvpp_rec_noseccut_binned.pkl")


    # #Run this code for testing as of 2/2023
    # dir_base = "/mnt/d/GLOBUS/CLAS12/Thesis/pickled_dvpip/merged_Fall_2018_Inbending_gen_test/"
    # for pklfile in os.listdir(dir_base):
    #     print("binning file {}".format(pklfile))
    #     df = pd.read_pickle(dir_base+pklfile)
    #     # # df_test = df.head(6)
    #     # # df_test.to_pickle("test_binning.pkl")
    #     # df = pd.read_pickle("test_binning.pkl")
    #     print(df)
    #     df_binned = bin_df(df,df_type="Gen")
    #     df_binned.to_pickle(dir_base+"../binned_"+pklfile)
    

    # sys.exit()
