
import os, sys
from icecream import ic
import json
import pandas as pd
from utils import make_histos


    
def make_all_histos(df,datatype="Recon",hists_2d=False,hists_1d=False,hists_overlap=False,saveplots=False,
                output_dir = "pics/",df_2=None,first_label="first",second_label="second",plot_title_identifiyer=""):

    var_prefix = ""
    if datatype=="Gen":
        var_prefix = "Gen"
    vals = df.columns.values
    #ic(vals)


    
    with open('utils/histo_config.json') as fjson:
        hftm = json.load(fjson)
    config = hftm["Ranges"][0]

    #Create set of 2D histos from JSON Specifications
    if hists_2d:
        for item in hftm:
            #print("in loop")

            if datatype=="Truth":
                try:
                    hm = hftm[item][0]
                    if hm["type"] == "2D":


                        for var_type in ["x","y"]:
                        
                            x_data = df["Gen"+hm["data_{}".format(var_type)]]
                            y_data = df[""+hm["data_{}".format(var_type)]]
                            if not x_data.isnull().values.any():
                                if not y_data.isnull().values.any():

                                    var_names = ["Gen "+hm["label_{}".format(var_type)],"Recon " + hm["label_{}".format(var_type)]]
                                    config_xy = [config[hm["type_{}".format(var_type)]],config[hm["type_{}".format(var_type)]]]
                                    ranges =  [config_xy[0][0],config_xy[0][0]]
                                    units = [config_xy[0][1],config_xy[0][1]]
                                    title = "{} vs. {}".format(var_names[0],var_names[0])
                                    filename = hm["filename"] # not currently used!

                                    #"Generated Events"
                                    #ic(x_data)
                                    #ic(y_data)
                                    if not x_data.empty and not y_data.empty:
                                        make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
                                                        saveplot=saveplots,pics_dir=output_dir+"hists_2D/",plot_title=title.replace("/",""),
                                                        filename=filename,units=units)
                            else:
                                print("WARNING: NULL VALUES FOUND FOR {} or {}".format(var_prefix+hm["data_{}".format(var_type)],var_prefix+hm["data_{}".format(var_type)]))
                except:
                    print("Exception found, skipping")
            else:
                try:
                    hm = hftm[item][0]
                    if hm["type"] == "2D":


                        x_data = df[var_prefix+hm["data_x"]]
                        y_data = df[var_prefix+hm["data_y"]]
                        if not x_data.isnull().values.any():
                            if not y_data.isnull().values.any():

                                var_names = [hm["label_x"],hm["label_y"]]
                                config_xy = [config[hm["type_x"]],config[hm["type_y"]]]
                                ranges =  [config_xy[0][0],config_xy[1][0]]
                                units = [config_xy[0][1],config_xy[1][1]]
                                title = "{} vs. {}, {}".format(var_names[0],var_names[1],datatype)
                                filename = hm["filename"] # not currently used!

                                #"Generated Events"
                                #ic(x_data)
                                #ic(y_data)
                                if not x_data.empty and not y_data.empty:
                                    make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
                                                    saveplot=saveplots,pics_dir=output_dir+"hists_2D/",plot_title=title.replace("/",""),
                                                    filename=filename,units=units)
                        else:
                            print("WARNING: NULL VALUES FOUND FOR {} or {}".format(var_prefix+hm["data_x"],var_prefix+hm["data_y"]))
                except:
                    print("Exception found, skipping")

    #Create set of 1D histos
    if hists_1d:
        for x_key in vals:
            #print("Creating 1 D Histogram for: {} ".format(x_key))
            xvals = df[x_key]
            if not xvals.empty:
                ranges = "none"
                if x_key == "ME_epgg":
                    ranges = [-0.12,0.12,100]
                elif x_key == "MM2_egg":
                    ranges = [0.75,1.05,100]
                elif x_key == "MM2_ep":
                    ranges = [-0.8,1.2,100]
                elif x_key == "MM2_epgg":
                    ranges = [-0.1,0.1,100]
                
                make_histos.plot_1dhist(xvals,[x_key,],ranges=ranges,second_x=False,first_label=first_label,
                        saveplot=saveplots,pics_dir=output_dir+"hists_1D/",plot_title=x_key)

    if hists_overlap:
        for x_key in vals:
            #print("Creating 1 D Histogram for: {} ".format(x_key))
            print(x_key)
            xvals_1 = df[x_key]
            xvals_2 = df_2[x_key]
            if not xvals_1.empty and not xvals_2.empty:
                make_histos.plot_1dhist(xvals_1,[x_key,],ranges="none",second_x=True,second_x_data=xvals_2,
                        saveplot=saveplots,pics_dir=output_dir+"hists_1D/",plot_title=x_key + ": "+ first_label + " vs. " + second_label,
                        first_label=first_label,second_label=second_label,plot_title_identifiyer=plot_title_identifiyer)
            


    
    # x_data = df_small_gen["GenxB"]
    # y_data = df_small_gen["GenQ2"]
    # x_data = df_small_gen["GenxB"]
    # y_data = df_small_gen["GenW"]
    # y_data = df_small_gen["Gent"]
    # x_data = df_small_gen["Genphi1"]
    # y_data = df_small_gen["GenPtheta"]
    # x_data = df_small_gen["GenPphi"]

    #Create set of overlapping 1D histos
    #       FILL OUT THIS SECTION XXXXXXXXXXXXXX
  




# start main
if __name__ == "__main__":

    test_gen_file = "/mnt/c/Users/rober/Dropbox/Bobby/Linux/work/CLAS12/mit-clas12-analysis/theana/paragon/analysis/threnody/0_convert_root_to_pickle/Gen/test/gen_test_norad/gen_test_norad_genOnly_4.pkl"
    test_gen_dir = "/mnt/c/Users/rober/Dropbox/Bobby/Linux/work/CLAS12/mit-clas12-analysis/theana/paragon/analysis/threnody/0_convert_root_to_pickle/Gen/test/gen_test_norad/"

#     #df = pd.read_pickle(test_gen_file)

#     # For plotting purposes, exlclude region W<1.9 GeV (abmnormality in generator makes plotting ugly)
#    # print("Length of df: {}".format(len(df)))


#     directory = test_gen_dir  # Update this path

#     #directory = "/mnt/d/GLOBUS/CLAS12/Thesis/0_raw_root_files/gen_inbend_norad/norad_10000_20230703_1814_Fall_2018_Inbending_50nA_gen/"
#     # Find all CSV files in the directory
#     files = [f for f in os.listdir(directory) if f.endswith('.pkl')]

#     # Initialize a list to store the dataframes
#     dfs = []

#     x = 'GenxB'
#     y = 'GenQ2'
#     # Load each file into a pandas DataFrame and append it to the list
#     for file in files:
        
#         filepath = os.path.join(directory, file)
#         print(file)
#         df_interm = pd.read_pickle(filepath)
#         # df_interm = df_interm[df_interm["GenW"] > 1.9]
#         # df_interm = df_interm[df_interm[x] > 240]
#         # df_interm = df_interm[df_interm[x] < 270]
#         # df_interm = df_interm[df_interm[y] > 60]
#         # df_interm = df_interm[df_interm[y] < 70]


#         df = df_interm
#         print(df['GenEpx'].nunique())
#         print(df[y].nunique())

        
        
#         print(df.columns.values)

#         import matplotlib.pyplot as plt
#         import matplotlib as mpl

#         # plt.hist(df[x], bins=500)
#         # plt.show()
#         # Assuming df is your DataFrame and it has columns 'phi1' and 'phi2'

#         plt.rcParams["font.size"] = "30"

#         # print the column values from the dataframe
#         for b in (50,100,300):            
#             # print(df[x])
#             # print(df[y])
#             #make figsize be 10x10
#             fig, ax = plt.subplots(figsize=(18,16))
#             plt.hist2d(df[x], df[y], bins=(b,b), norm=mpl.colors.LogNorm())
#             plt.colorbar()
#             plt.xlabel('xB')
#             plt.ylabel('Q2')
#             plt.title('Generated Events xB vs. Q2')
#             #save the plot
#             plt.savefig('2dhist_xB_Q2_{}_bins.png'.format(b))
#             plt.close()




#         sys.exit()
#     #     #require GenW > 1.9
#     #     dfs.append(df_interm)

#    Combine all the dataframes
#    df= pd.concat(dfs, ignore_index=True)

#   #print(df.columns.values)
    #df = pd.read_pickle("/mnt/d/GLOBUS/CLAS12/Thesis/1_potential_dvpip_events/gen/lund_10000_20230624_1247_norad.pkl")
    df = pd.read_pickle("test_f18_exp_corr.pkl")

    make_all_histos(df,datatype="Exp",hists_2d=True,hists_1d=True,hists_overlap=False,saveplots=True,
                   output_dir = "/mnt/d/GLOBUS/CLAS12/Thesis/plots/1_all_event_distros/exp/",df_2=None,first_label="first",second_label="second",plot_title_identifiyer="")





