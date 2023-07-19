import pandas as pd
import numpy as np
import os, sys
from icecream import ic
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.optimize import curve_fit
from utils import filestruct, const, make_histos


import os
from PIL import Image
import numpy as np
import re

def fit_function(phi,A,B,C):
    #A + B*np.cos(2*phi) +C*np.cos(phi)
    rads = phi*np.pi/180
    #return (A * np.exp(-x/beta) + B * np.exp(-1.0 * (x - mu)**2 / (2 * sigma**2)))
    #A = T+L, B=TT, C=LT
    #A = black, B=blue, C=red
    return A + B*np.cos(2*rads) + C*np.cos(rads)


#pd.set_option('mode.chained_assignment', None)


PhysicsConstants = const.PhysicsConstants()


fs = filestruct.fs()

combined_df = pd.read_pickle("full_cross_section_clas12.pkl")
"""
['tmin' 'pmin' 'xmin' 'qmin' 'tmax' 'pmax' 'xmax' 'qmax' 'tave' 'pave'
 'xave' 'qave' 'yave' 'counts' 'true_xbq2_bin_volume'
 'nominal_xbq2_bin_volume' 'volume_ratio' 'tp_bin_volume' 'true_total_vol'
 'rec_tave' 'rec_pave' 'rec_xave' 'rec_qave' 'rec_yave' 'rec_counts'
 'gen_tave' 'gen_pave' 'gen_xave' 'gen_qave' 'gen_yave' 'gen_counts'
 'counts_rec_45na' 'counts_rec_55na' 'counts_rec_nominal_1'
 'counts_rec_nominal_2' 'counts_rec_rad' 'counts_gen_45na'
 'counts_gen_55na' 'counts_gen_nominal_1' 'counts_gen_nominal_2'
 'counts_gen_rad' 'acc_45na' 'acc_55na' 'acc_nominal_1' 'acc_nominal_2'
 'acc_rad' '45na-nom' '55na-nom' 'rad-nom' '45na-55na' 'nom-nom'
 '45na-nom-nom-nom' '55na-nom-nom-nom' 'rad-nom-nom-nom' 'acc_corr'
 'rad_corr_alt' 'acc_corr_err' 'acc_rad_err' 'rad_corr_alt_err'
 'counts_err' 'rec_counts_err' 'gen_counts_err' 'acc_corr_counts'
 'acc_corr_counts_err' 'xsec' 'xsec_err' 'y_bin' 'epsi_num' 'epsi_denom'
 'epsilon' 'Gamma' 'xsec_red' 'xsec_red_err' 'acc_corr_counts_err_alt'
 'xsec_err_alt' 'xsec_red_err_alt' 'sys_uncert_45na' 'sys_uncert_55na'
 'sys_uncert_recon' 'sys_uncert_acc-corr' 'sys_uncert_rad' 'stat_uncert'
 'total_uncert']

"""




clas_df = pd.read_pickle("3_calculate_xsec/CLAS6_struct_funcs_raw.pkl")
clas_df.replace(to_replace=r'−', value='-', regex=True, inplace=True)

for column in clas_df.columns:
    #convert type into float
    clas_df[column] = clas_df[column].astype(float)
print(clas_df.columns.values)

clas_dtp = pd.read_csv("3_calculate_xsec/xs_clas6.csv")
#set the columns
"""
'Q2_C6' 'xB_C6' 't_C6' 'tel_C6' 'telstat_C6' 'telsys_C6' 'lt_C6'
 'ltstat_C6' 'ltsys_C6' 'tt_C6' 'ttstat_C6' 'ttsys_C6'
"""
#the columns are the first row
#clas_dtp.columns = clas_dtp.iloc[0]

#print(clas_dtp)
#sys.exit()

show_xsec = True

if show_xsec:



    # grouping by 'xmin', 'qmin', 'tmin'

    # Create an empty DataFrame to store the fit parameters and their uncertainties
    #fit_results = pd.DataFrame(columns=['xmin', 'qmin', 'tmin', 'A','B','C','A_uncert','B_uncert','C_uncert'])
    fit_results = []
    # Iterate over groups
    groups = combined_df.groupby(['xmin', 'qmin', 'tmin'])

    ind = 0
    for name, group in groups:
        #only plot if there are more than 10 bins
        
        print(clas_df[['Q2_C6', 'xB_C6', 't_C6']].dtypes)
        print(group[['qmin', 'xmin', 'tmin', 'qmax', 'xmax', 'tmax']].dtypes)


        print(clas_df['Q2_C6'])
        print(group['qmin'].values[0])


        print(group['qmin'])

        mask = (clas_df['Q2_C6'] >= group['qmin'].values[0]) & (clas_df['Q2_C6'] <= group['qmax'].values[0]) & \
       (clas_df['xB_C6'] >= group['xmin'].values[0]) & (clas_df['xB_C6'] <= group['xmax'].values[0]) & \
       (clas_df['t_C6'] >= group['tmin'].values[0]) & (clas_df['t_C6'] <= group['tmax'].values[0])

        filtered_df = clas_df[mask]


        if len(group) < 5:
            print("skipping")
            continue
        # Also need to skip if there is no data between 'pave' = 100 and 'pave' = 260
        # select the elements with 100<pave<260
        filtered_group = group[(group['pave'] > 100) & (group['pave'] < 260)]

        if len(filtered_group) == 0:
            print("skipping")
            continue
        else:
            print("PLOTTING")
            plt.rcParams["font.size"] = "30"
            plt.figure(figsize=(20,14))

            slabel = "Stat. Err. from Sim."
            elabel = "Stat. Err. from Exp."
            #plot with larger marker size
            #for showing different uncertainties
            #plt.errorbar(group['pave'], group['xsec_red'], yerr=group['xsec_red_err'],fmt='r+', markersize=50,label=#slabel)
            #plot again but with red error bars
            xerr_value = 10
            plt.errorbar(group['pave'], group['xsec_red'], xerr=xerr_value,yerr=2*group['total_uncert'],fmt='r.',  markersize=5,elinewidth=5)
            #,label="CLAS12 Data")#elabel)

            plt.errorbar(group['pave'], group['xsec_red'], xerr=xerr_value,yerr=group['xsec_red_err_alt'],fmt='k.',  markersize=5,label="CLAS12 Data",elinewidth=5)#elabel)






            # fit the function to the data
            popt, pcov = curve_fit(fit_function, group['pave'], group['xsec_red'], sigma=group['xsec_red_err'], absolute_sigma=True)

            # print out the fit parameters
            print(f"A = {popt[0]}, B = {popt[1]}, C = {popt[2]}")

            # Calculate uncertainties
            perr = np.sqrt(np.diag(pcov))
            print(perr)
            print(popt)
            # Append the fit results to the DataFrame
            """fit_results = pd.concat([fit_results,pd.DataFrame({'xmin': name[0], 
                                            'qmin': name[1], 
                                            'tmin': name[2], 
                                            'A': popt[0],
                                            'B': popt[1], 
                                            'C': popt[2], 
                                            'A_uncert': perr[0],
                                            'B_uncert': perr[1],
                                            'C_uncert': perr[2]})], 
                                            ignore_index=False)"""

            # plot the fit
            #phis = np.linspace(group['pave'].min(), group['pave'].max(), 1000)
            phis = np.linspace(0,360,1000)
            plt.plot(phis, fit_function(phis, *popt), 'k-', label="CLAS12 Fit",linewidth=5)
            #

            #blue_line = mlines.Line2D([], [], color='r', marker='None', markersize=10, linestyle='-', label=slabel)
            #red_line = mlines.Line2D([], [], color='k', marker='None', markersize=10, linestyle='-', label=elabel)


            plt.xlabel('Lepton-Hadron Angle $\phi$')
            plt.ylabel('Reduced Cross Section (nb/$GeV^2$)')
            #pltt = 'Reduced Cross Section in bin ({})={}'.format(r'$x_{B,min} Q^2_{min} t_{min}$',str(name))
            pltt = '({})={}'.format(r'$x_{B,min} Q^2_{min} t_{min}$',str(name))
            
            plt.title(pltt)
            #plt.legend(handles=[blue_line, red_line])

            c6_tel = np.nan
            c6_tt = np.nan
            c6_lt = np.nan
            c6_tel_err = 0
            c6_tt_err = 0
            c6_lt_err = 0
            t_C6 = 0

            #c6_tel_stat = np.nan
            # add the rest!

            'Q2_C6' 'xB_C6' 't_C6' 'tel_C6' 'telstat_C6' 'telsys_C6' 'lt_C6'
            'ltstat_C6' 'ltsys_C6' 'tt_C6' 'ttstat_C6' 'ttsys_C6'
            pi = 3.14159

            print(group.columns.values)
            prefact_A = 1/(2*pi)
            prefact_B = prefact_A*group['epsilon'].mean()
            prefact_C = prefact_A*np.sqrt(2*group['epsilon'].mean()*(1+group['epsilon'].mean()))

            if len(filtered_df) > 0:
                mask = (clas_dtp['q'] >= group['qmin'].values[0]) & (clas_dtp['q'] <= group['qmax'].values[0]) & \
                (clas_dtp['x'] >= group['xmin'].values[0]) & (clas_dtp['x'] <= group['xmax'].values[0]) & \
                (clas_dtp['t'] >= group['tmin'].values[0]) & (clas_dtp['t'] <= group['tmax'].values[0])

                filtered_df_dtp = clas_dtp[mask]
                print(filtered_df_dtp)

                #plt.errorbar(filtered_df_dtp['p'], filtered_df_dtp['dsdtdp'], yerr=np.sqrt(filtered_df_dtp['stat']**2+filtered_df_dtp['sys']**2),fmt='r+', markersize=50,label='CLAS6')

                errband_width = np.sqrt(filtered_df_dtp['stat']**2+filtered_df_dtp['sys']**2).mean()

                



                # plot the CLAS6 fit if it exists
                print("PLOTTING CLAS6 FIT")

                phi = np.linspace(0, 360, 1000)  # Replace 100 with the desired number of points

                # Assuming taking the first row of the filtered DataFrame
                print("FILTERED DF IS:")
                print(group.columns.values)
                print(filtered_df)
                print("xsec values are")
                print(group['pave'])
                row = filtered_df.iloc[0]

                c6_tel = row['tel_C6']
                c6_tt = row['tt_C6']
                c6_lt = row['lt_C6']



                A = row['tel_C6']
                B = row['tt_C6']
                C = row['lt_C6']

                c6_tel_err = row['telstat_C6']
                c6_tt_err = row['ttstat_C6']
                c6_lt_err = row['ltstat_C6']
                t_C6 = row['t_C6']

                #fact = group['Gamma'].mean()/(2*pi)

                y = fit_function(phi, A*prefact_A, B*prefact_B, C*prefact_C)



                print(A,B,C)

                plt.plot(phi, y,'b-',label='CLAS6 Result',linewidth=5)
                #make line be 50% transparent
                #plt.plot(phi, y+errband_width,'k-',label='CLAS6 Fit',alpha=0.5)#
                #plt.plot(phi, y-errband_width,'k-',label='CLAS6 Fit',alpha=0.5)#
                
                y_bottom = y-errband_width
                y_top = y+errband_width

                plt.fill_between(phi, y_bottom, y_top, color='b', alpha=.1)#, label='CLAS6 Fit')

                #,linewidth = errband_width)

                phi = np.linspace(0, 360, 1000)  # Replace 100 with the desired number of points

                # Assuming taking the first row of the filtered DataFrame
                #print("FILTERED DF IS:")
                #print(group.columns.values)
                #print(filtered_df)
                #print("xsec values are")
                #print(group['pave'])
                # # row = filtered_df.iloc[0]

                # # A = row['tel_C6']+np.sqrt(row['telsys_C6']**2+row['telsys_C6']**2)
                # # B = row['tt_C6']+np.sqrt(row['ttsys_C6']**2+row['ttsys_C6']**2)
                # # C = row['lt_C6']+np.sqrt(row['ltsys_C6']**2+row['ltsys_C6']**2)
                # # pi = 3.14159
                # # #fact = group['Gamma'].mean()/(2*pi)
                # # fact = 1/(2*pi)
                # # fact2 = fact*group['epsilon'].mean()
                # # fact3 = fact*np.sqrt(2*group['epsilon'].mean()*(1+group['epsilon'].mean()))
                # # y = fit_function(phi, A*fact, B*fact2, C*fact3)

                # # print(A,B,C)

                # # plt.plot(phi, y,'k-',label='CLAS6 Fit high',linewidth=5)



                # # A = row['tel_C6']-np.sqrt(row['telsys_C6']**2+row['telsys_C6']**2)
                # # B = row['tt_C6']-np.sqrt(row['ttsys_C6']**2+row['ttsys_C6']**2)
                # # C = row['lt_C6']-np.sqrt(row['ltsys_C6']**2+row['ltsys_C6']**2)
                # # pi = 3.14159
                # # #fact = group['Gamma'].mean()/(2*pi)
                # # fact = 1/(2*pi)
                # # fact2 = fact*group['epsilon'].mean()
                # # fact3 = fact*np.sqrt(2*group['epsilon'].mean()*(1+group['epsilon'].mean()))
                # # y = fit_function(phi, A*fact, B*fact2, C*fact3)

                # # print(A,B,C)

                # # plt.plot(phi, y,'k-',label='CLAS6 Fit high',linewidth=5)
            
            fit_results.append([name[0],
            name[1],
            group['tave'].mean(),    
            popt[0]/prefact_A,
            popt[1]/prefact_B,
            popt[2]/prefact_C,
            perr[0]/prefact_A,
            perr[1]/prefact_B,
            perr[2]/prefact_C,
            c6_tel,
            c6_tt,
            c6_lt,
            c6_tel_err,
            c6_tt_err,
            c6_lt_err,
            t_C6
            ])



            
            plt.ylim(bottom=0)
            #plt.show()
            #plt.savefig("tdep_test/"+pltt+".png",bbox_inches='tight')
            #sys.exit()
            plt.close()
            # plt.show()
            # ind += 1
            # if ind > 20:
            #     break
                #plt.close()
fit_results_df = pd.DataFrame(fit_results,columns=['xmin', 'qmin', 'tave', 'A','B','C','A_uncert','B_uncert','C_uncert','c6_tel','c6_tt','c6_lt','c6_tel_err','c6_tt_err','c6_lt_err','tave_c6'])
print(fit_results_df)
fit_results_df.to_pickle("t_dep_of_xsec.pkl")
