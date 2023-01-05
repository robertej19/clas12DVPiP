import os, sys
import pandas as pd
import numpy as np
from utils import make_histos
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# def fit_function(x, A, beta, B, mu, sigma):
#         return (A * np.exp(-x/beta) + B * np.exp(-1.0 * (x - mu)**2 / (2 * sigma**2)))


# def fit_distribution():

        
#         # 3.) Generate exponential and gaussian data and histograms.
#         data = np.random.exponential(scale=2.0, size=100000)
#         data2 = np.random.normal(loc=3.0, scale=0.3, size=15000)
#         bins = np.linspace(0, 6, 61)
#         data_entries_1, bins_1 = np.histogram(data, bins=bins)
#         data_entries_2, bins_2 = np.histogram(data2, bins=bins)

#         # 4.) Add histograms of exponential and gaussian data.
#         data_entries = data_entries_1 + data_entries_2
#         binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])

#         # 5.) Fit the function to the histogram data.
#         popt, pcov = curve_fit(fit_function, xdata=binscenters, ydata=data_entries, p0=[20000, 2.0, 2000, 3.0, 0.3])
#         print(popt)

#         # 6.)
#         # Generate enough x values to make the curves look smooth.
#         xspace = np.linspace(0, 6, 100000)

#         # Plot the histogram and the fitted function.
#         plt.bar(binscenters, data_entries, width=bins[1] - bins[0], color='navy', label=r'Histogram entries')
#         plt.plot(xspace, fit_function(xspace, *popt), color='darkorange', linewidth=2.5, label=r'Fitted function')

#         # Make the plot nicer.
#         plt.xlim(0,6)
#         plt.xlabel(r'x axis')
#         plt.ylabel(r'Number of entries')
#         plt.title(r'Exponential decay with gaussian peak')
#         plt.legend(loc='best')
#         plt.show()
#         plt.clf()


def makeDVpi0P(df_epgg, pol = "inbending", sigma_multiplier=3, unique_identifyer="", datafilename="temporary_exclusivity_variances_",use_generic_cuts=True):

        #Variables listing:

        if use_generic_cuts:
                df_ex_cut_ranges = pd.read_pickle("/mnt/d/GLOBUS/CLAS12/APS2022/pickled_data/20220511_f18_in_combined_157_table_of_ex_cut_testing_new_binning_mechanism_old_scheme.pkl")
        else:
                df_ex_cut_ranges = pd.read_pickle(datafilename+unique_identifyer+".csv")




        print(" THE EXCLUSIVITY CUT RANGES ARE: ", df_ex_cut_ranges)
        print(" SIGMA MULTIPLIER IS: ", sigma_multiplier)

        ex_vars = ["Mpi0",
                "reconPi",
                "MPt",
                "MM2_epgg",
                "MM2_ep",
                "MM2_egg",
                "ME_epgg"]

        limits = ["hi","low"]

        
        cuts = []

        for var in ex_vars:
                df_mu_sigma = df_ex_cut_ranges.query("var_name == '" + var + "'")
                df_mu = df_mu_sigma["mu"].values[0]
                df_sigma = df_mu_sigma["sigma"].values[0]

                cut_low = df_mu - sigma_multiplier * df_sigma
                cut_hi = df_mu + sigma_multiplier * df_sigma

                app_cut_hi = df_epgg.loc[:, var] < cut_hi
                app_cut_low = df_epgg.loc[:, var] > cut_low

                app_cut = app_cut_hi & app_cut_low

                cuts.append(app_cut)
        
        print(len(cuts))
        cuts_combined = cuts[0] & cuts[1] & cuts[2] & cuts[3] & cuts[4] & cuts[5] & cuts[6]




        # cut_pi0upper = df_epgg.loc[:, "Mpi0"] < vars_dict["Mpi0"]["hi"]
        # cut_pi0lower = df_epgg.loc[:, "Mpi0"] > vars_dict["Mpi0"]["low"]

        # cut_recon_hi = df_epgg.loc[:, "reconPi"] < vars_dict["reconPi"]["hi"]  # recon gam angle
        # cut_recon_low = df_epgg.loc[:, "reconPi"] < vars_dict["reconPi"]["low"]  # recon gam angle

        # cut_mpt_hi = df_epgg.loc[:, "MPt"] < vars_dict["MPt"]["hi"]  # mpt
        # cut_mpt_low = df_epgg.loc[:, "MPt"] > vars_dict["MPt"]["low"]  # mpt

        # cut_mmepgg_hi = np.abs(df_epgg["MM2_epgg"]) < vars_dict["MM2_epgg"]["hi"]#0.0440  # mmepgg
        # cut_mmepgg_low = np.abs(df_epgg["MM2_epgg"]) > vars_dict["MM2_epgg"]["low"]#-0.0478  # mmepgg

        # cut_mmep_hi = df_epgg.loc[:, "MM2_ep"] < vars_dict["MM2_ep"]["hi"]#0.7  # mmep
        # cut_mmep_low = df_epgg.loc[:, "MM2_ep"] > vars_dict["MM2_ep"]["low"]6#-0.66  # mmep

        # cut_mmegg_hi = df_epgg.loc[:, "MM2_egg"] < vars_dict["MM2_egg"]["hi"] #0.95  # mm_egg
        # cut_mmegg_low = df_epgg.loc[:, "MM2_egg"] > vars_dict["MM2_egg"]["low"] # 0.8  # mm_egg

        # cut_meepgg_hi = df_epgg.loc[:, "ME_epgg"] < vars_dict["ME_epgg"]["hi"]  # meepgg
        # cut_meepgg_low = df_epgg.loc[:, "ME_epgg"] > vars_dict["ME_epgg"]["low"]  # meepgg



        #common cuts:
        df_epgg.loc[:, "closeness"] = np.abs(df_epgg.loc[:, "Mpi0"] - .1349766)
        cut_xBupper = df_epgg.loc[:, "xB"] < 1  # xB
        cut_xBlower = df_epgg.loc[:, "xB"] > 0  # xB
        cut_Q2 = (df_epgg.loc[:, "Q2"] > 1)# & (df_epgg.loc[:, "Q2"] <1.5)  # Q2
        cut_W = df_epgg.loc[:, "W"] > 2  # W

        # cut_Mpi0 = cut_pi0lower & cut_pi0upper
        # cut_recon = cut_recon_hi & cut_recon_low
        # cut_mpt = cut_mpt_hi & cut_mpt_low
        # cut_mmepgg = cut_mmepgg_hi & cut_mmepgg_low
        # cut_mmep = cut_mmep_hi & cut_mmep_low
        # cut_mmegg = cut_mmegg_hi & cut_mmegg_low
        # cut_meepgg = cut_meepgg_hi & cut_meepgg_low

        #Angle cuts
        cut_etheta_discrep = df_epgg.loc[:, "Etheta"] > 0 # W
        cut_Ptheta_low = df_epgg.loc[:, "Ptheta"] > 5  # W
        cut_Ptheta_hi = df_epgg.loc[:, "Ptheta"] < 65  # W
        cut_Ptheta_mid = df_epgg.loc[:, "Ptheta"] < 42#35  # W
        cut_Ptheta_mid2 = df_epgg.loc[:, "Ptheta"] >42  # W
        cut_Pphi_low = df_epgg.loc[:, "Pphi"] > -360  # W
        cut_Pphi_hi = df_epgg.loc[:, "Pphi"] < 360  # W
        cut_Gpz2_low = df_epgg.loc[:, "Gpz2"] > 0.4#0.4  # Gpz2
        cut_Gpz_hi = df_epgg.loc[:, "Gpz"] < 6000  # Gpz2
        cut_Ptheta_low_mid_hi = (cut_Ptheta_low & cut_Ptheta_mid)|(cut_Ptheta_mid2 & cut_Ptheta_hi)


        df_dvpi0 = df_epgg.loc[cut_xBupper & cut_xBlower & cut_Q2 & cut_W & 
                                cut_Ptheta_low_mid_hi &
                                cut_Pphi_low &
                                cut_Pphi_hi &
                                cuts_combined &                
                                cut_Gpz2_low &
                                cut_Gpz_hi &
                                cut_etheta_discrep, :]


        df_dvpi0 = df_dvpi0.sort_values(by=['closeness', 'Psector', 'Gsector'], ascending = [True, True, True])
        df_dvpi0 = df_dvpi0.loc[~df_dvpi0.event.duplicated(), :]
        df_dvpi0 = df_dvpi0.sort_values(by='event')        

        print(len(df_dvpi0))
        return df_dvpi0

def calc_ex_cut_mu_sigma(df_epgg, pol = "inbending",data_type="exp",proton_loc="All",
                        photon1_loc="All",photon2_loc="All",simple_exclusivity_cuts=False,
                        datafilename="temporary_exclusivity_variances_",
                        unique_identifyer=""):

        print(" UNIQUE IDENTIFYER IS: ", unique_identifyer)
        #Variables listing:
        print(len(df_epgg))
        df_out = df_epgg.head(1000000)
        df_out.to_pickle("temp_df_file_outbend.pkl")

        ex_vars = ["Mpi0",
                "reconPi",
                "MPt",
                "MM2_epgg",
                "MM2_ep",
                "MM2_egg",
                "ME_epgg"]
        

        other_vars = ["xB",
                "Q2",
                "W",
                "Etheta",
                "Ptheta",
                "Pphi",
                "Gpz",
                "Gpz2",
                "Psector",
                "Gsector",
                "Gsector2"]

        # for x_key in ex_vars:
        #         x_data = df_epgg[x_key]
        #         output_dir = "./"
        #         make_histos.plot_1dhist(x_data,[x_key,],ranges="none",second_x=False,first_label=data_type,logger=True,
        #                         saveplot=True,pics_dir=output_dir+"hists_1D/no_cuts/",plot_title=x_key+", "+data_type+"Before Cuts")
        

        #common cuts:
        df_epgg.loc[:, "closeness"] = np.abs(df_epgg.loc[:, "Mpi0"] - .1349766)
        cut_xBupper = df_epgg.loc[:, "xB"] < 1  # xB
        cut_xBlower = df_epgg.loc[:, "xB"] > 0  # xB
        cut_Q2 = (df_epgg.loc[:, "Q2"] > 1)# & (df_epgg.loc[:, "Q2"] <1.5)  # Q2
        cut_W = df_epgg.loc[:, "W"] > 2  # W

        df_epgg = df_epgg.loc[cut_xBupper & cut_xBlower & cut_Q2 & cut_W]

         #Collaboration approved
        cut_pi0upper = df_epgg.loc[:, "Mpi0"] < 0.168
        cut_pi0lower = df_epgg.loc[:, "Mpi0"] > 0.096
        cut_Mpi0 = cut_pi0lower & cut_pi0upper

        cut_recon = df_epgg.loc[:, "reconPi"] < 2  # recon gam angle

        cut_mpt_hi = df_epgg.loc[:, "MPt"] < 0.2  # mpt
        cut_mpt_low = df_epgg.loc[:, "MPt"] > -0.2  # mpt
        cut_mpt = cut_mpt_hi & cut_mpt_low

        cut_mmepgg_hi = np.abs(df_epgg["MM2_epgg"]) < 0.05#0.0440  # mmepgg
        cut_mmepgg_low = np.abs(df_epgg["MM2_epgg"]) > -0.05#-0.0478  # mmepgg
        cut_mmepgg = cut_mmepgg_hi & cut_mmepgg_low
        #Extra

        cut_mmep_hi = df_epgg.loc[:, "MM2_ep"] < 0.2#0.7  # mmep
        cut_mmep_low = df_epgg.loc[:, "MM2_ep"] > -0.196#-0.66  # mmep
        cut_mmep = cut_mmep_hi & cut_mmep_low

        cut_mmegg_hi = df_epgg.loc[:, "MM2_egg"] < 1.6 #0.95  # mm_egg
        cut_mmegg_low = df_epgg.loc[:, "MM2_egg"] > 0.16 # 0.8  # mm_egg
        cut_mmegg = cut_mmegg_hi & cut_mmegg_low

        cut_meepgg_hi = df_epgg.loc[:, "ME_epgg"] < 0.5  # meepgg
        cut_meepgg_low = df_epgg.loc[:, "ME_epgg"] > -0.5  # meepgg
        cut_meepgg = cut_meepgg_hi & cut_meepgg_low


        ex_cuts = [cut_Mpi0,
                cut_recon,
                cut_mpt,
                cut_mmepgg,
                cut_mmep,
                cut_mmegg,
                cut_meepgg]

        ex_cuts_names = ["cut_Mpi0",
                "cut_recon",
                "cut_mpt",
                "cut_mmepgg",
                "cut_mmep",
                "cut_mmegg",
                "cut_meepgg"]

        ex_cuts_dict = {"cut_Mpi0":cut_Mpi0,
                "cut_recon":cut_recon,
                "cut_mpt":cut_mpt,
                "cut_mmepgg":cut_mmepgg,
                "cut_mmep":cut_mmep,
                "cut_mmegg":cut_mmegg,
                "cut_meepgg":cut_meepgg}

        ex_cuts_ranges = [ [0.05,0.25,100],
                 [-0.1,5,100],
                 [0,0.5,100],
                 [-0.06,0.06,100],
                 [-1,1,100],
                 [-0.2,2.0,100],
                 [-1,1,100]]

        # ex_cuts_ranges = ["none",
        #         "none",
        #         "none",
        #         "none",
        #         "none",
        #         "none",
        #         "none"]

        ex_cuts2 = [cut_Mpi0]

        var_names = []
        mu_values = []
        sigmasquared_values = []

        for xind,x0_key in enumerate(ex_cuts_names):
                df_sample = df_epgg
                #print(xind)
                #print(x0_key)
                cut_key_list = [x for x in ex_cuts_names if ex_cuts_names.index(x) != xind]
                #print(cut_key_list)
                for key in cut_key_list:
                        df_sample = df_sample.loc[ex_cuts_dict[key]]
                


                # sys.exit()
                
                # for cut in cut_list:
                #         df_sample = df_sample.loc[cut]
                if 0 == 1:

                        for x_index,x_key in enumerate(ex_vars):
                                x_data = df_sample[x_key]
                                output_dir = "./"
                                ranges = ex_cuts_ranges[x_index]

                                make_histos.plot_1dhist(x_data,[x_key,],ranges=ranges,second_x=False,first_label=data_type,logger=False,
                                                saveplot=True,pics_dir=output_dir+"hists_1D/all_cuts_but_{}/".format(x0_key),plot_title=x_key+", "+data_type+" All Cuts Except "+x0_key)
                if 1 == 1:
                        
                        print("on xkey:",x0_key)
                        x_data = df_sample[ex_vars[ex_cuts_names.index(x0_key)]]



                        output_dir = "./"
                        ranges = ex_cuts_ranges[xind]



                        popt, pcov = make_histos.plot_1dhist(x_data,[x0_key,],ranges=ranges,second_x=False,first_label=data_type,logger=False,x0_key=x0_key,
                                        saveplot=True,pics_dir=output_dir+"hists_1D/",plot_title=ex_vars[ex_cuts_names.index(x0_key)]+", "+data_type+" All Cuts Except "+ex_vars[ex_cuts_names.index(x0_key)],fitdata=True)

                        # print(ex_vars[ex_cuts_names.index(x0_key)],
                        #                 "A = {}, Mu = {}, SigmaSquared = {}".format(*popt),
                        #                 "CovMatrix = {}".format(pcov))

                        var_names.append(ex_vars[ex_cuts_names.index(x0_key)])
                        mu_values.append(popt[1])
                        sigmasquared_values.append(popt[2])

                               
        q = np.column_stack([var_names,mu_values,np.sqrt(sigmasquared_values)])
        #print(q)
        df = pd.DataFrame (q, columns = ['var_name', 'mu', 'sigma'])
        #print (type(df.mu.values(0)))
        df['mu'] = df['mu'].astype(float)
        df['sigma'] = df['sigma'].astype(float)


        df.loc[:,"mu+3sigma"] = df.loc[:,"mu"] + 3*df.loc[:,"sigma"]
        df.loc[:,"mu-3sigma"] = df.loc[:,"mu"] - 3*df.loc[:,"sigma"]

        df.loc[:,"mu+2sigma"] = df.loc[:,"mu"] + 2*df.loc[:,"sigma"]
        df.loc[:,"mu-2sigma"] = df.loc[:,"mu"] - 2*df.loc[:,"sigma"]

        df.loc[:,"mu+4sigma"] = df.loc[:,"mu"] + 4*df.loc[:,"sigma"]
        df.loc[:,"mu-4sigma"] = df.loc[:,"mu"] - 4*df.loc[:,"sigma"]

        print(df)
        df.to_csv(datafilename+unique_identifyer+".csv")
        #exclusivity_cut_limits

        return df

if __name__ == "__main__":
    #test_df = pd.read_pickle("test_df_epgg.pkl")
    test_df = pd.read_pickle("test_files/temp_df_file_outbend.pkl")
    #print(test_df)
    df_dvpi0p_exp = calc_ex_cut_mu_sigma(test_df)

        
#    test_ranges = pd.read_pickle("temporary_exclusivity_variances.pkl")

 #   makeDVpi0P(test_df,test_ranges,4)

    #df_exp = pd.read_pickle("new_exp_convert_outbend.pkl",pol="outbending")
    #df_rec = pd.read_pickle("new_rec_convert_outbend_rad.pkl",pol="outbending")

    #df_dvpi0p_exp = makeDVpi0P(df_exp)
    #df_dvpi0p_rec = makeDVpi0P(df_rec)

    ##df_dvpi0p_exp.to_pickle("new_exp_dvpi0p_outbend.pkl")
    #df_dvpi0p_rec.to_pickle("new_rec_dvpi0p_outbend_rad.pkl")
