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

combined_df = pd.read_pickle("full_cross_section_clas12_unfolded.pkl")





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

show_xsec1 = True
show_xsec = False
plot_ylabel = True
if show_xsec:


    # drop rows from combined_df where uncertainty/value > 1

    # make a histogram of xsec_red_err_unfolded


    # make histogram of total_uncert_unfolded
    # plt.hist(combined_df['total_uncert_unfolded']/combined_df['xsec_red_unfolded'], bins=100)
    # #set y axis to log
    # plt.yscale('log')
    # plt.show()
    # sys.exit()

    # grouping by 'xmin', 'qmin', 'tmin'
    groups = combined_df.groupby(['xmin', 'qmin', 'tmin'])



    ind = 0
    counter = 0
    for name, group in groups:
        #only plot if there are more than 10 bins
        print("in name")
        counter += 1
        print(counter)
        print(clas_df[['Q2_C6', 'xB_C6', 't_C6']].dtypes)
        print(group[['qmin', 'xmin', 'tmin', 'qmax', 'xmax', 'tmax']].dtypes)


        print(clas_df['Q2_C6'])
        print(group['qmin'].values[0])


        print(group['qmin'])

        mask = (clas_df['Q2_C6'] >= group['qmin'].values[0]) & (clas_df['Q2_C6'] <= group['qmax'].values[0]) & \
       (clas_df['xB_C6'] >= group['xmin'].values[0]) & (clas_df['xB_C6'] <= group['xmax'].values[0]) & \
       (clas_df['t_C6'] >= group['tmin'].values[0]) & (clas_df['t_C6'] <= group['tmax'].values[0])



if show_xsec1:

    combined_df = combined_df[combined_df['total_uncert_unfolded']/combined_df['xsec_red_unfolded'] < .7] #cutoff chosen emperically


    # grouping by 'xmin', 'qmin', 'tmin'

    # Create an empty DataFrame to store the fit parameters and their uncertainties
    #fit_results = pd.DataFrame(columns=['xmin', 'qmin', 'tmin', 'A','B','C','A_uncert','B_uncert','C_uncert'])
    fit_results = []
    # Iterate over groups
    groups = combined_df.groupby(['xmin', 'qmin', 'tmin'])

    ind = 0
    for name, group in groups:
        #only plot if there are more than 10 bins
        


        mask = (clas_df['Q2_C6'] >= group['qmin'].values[0]) & (clas_df['Q2_C6'] <= group['qmax'].values[0]) & \
       (clas_df['xB_C6'] >= group['xmin'].values[0]) & (clas_df['xB_C6'] <= group['xmax'].values[0]) & \
       (clas_df['t_C6'] >= group['tmin'].values[0]) & (clas_df['t_C6'] <= group['tmax'].values[0])

        filtered_df = clas_df[mask]
        filtered_group = group[(group['pave'] > 100) & (group['pave'] < 260)]

        plt.rcParams["font.size"] = "30"
        fig, ax = plt.subplots(figsize=(20,14))
        slabel = "Stat. Err. from Sim."
        elabel = "Stat. Err. from Exp."
        #plot with larger marker size
        #for showing different uncertainties
        #plt.errorbar(group['pave'], group['xsec_red'], yerr=group['xsec_red_err'],fmt='r+', markersize=50,label=#slabel)
        #plot again but with red error bars
        xerr_value = 5

        if len(group) < 3 or len(filtered_group) == 0:
            continue
        
            xerr_value = 0
            ax.errorbar(group['pave'], group['xsec_red_unfolded'], xerr=xerr_value,yerr=1.5*group['total_uncert_unfolded'],fmt='k.',  markersize=5,elinewidth=5,capsize=10, capthick=5)#elabel)
            xerr_value=9
            ax.errorbar(group['pave'], group['xsec_red_unfolded'], xerr=xerr_value,yerr=group['xsec_red_err_unfolded'],fmt='r.',  markersize=5,label="Unfolded",elinewidth=5)#,capsize=10, capthick=5)#elabel)
            xerr_value = 0
            ax.errorbar(group['pave'], group['xsec_red_unfolded'], xerr=xerr_value,yerr=0,fmt='r.',  markersize=5,elinewidth=5,capsize=10, capthick=5)#elabel)



            ax.plot(group['pave']-6, group['xsec_red'], marker='^', color='orange', markersize=21, label="Bin-by-Bin",linestyle="None")

        else:
            # fit the function to the data
            popt, pcov = curve_fit(fit_function, group['pave'], group['xsec_red_unfolded'], sigma=group['total_uncert_unfolded'], absolute_sigma=True)
            popt_top, pcov_top = curve_fit(fit_function, group['pave'], group['xsec_red_unfolded']+group['total_uncert_unfolded'], sigma=group['total_uncert_unfolded'], absolute_sigma=True)
            popt_bot, pcov_bot = curve_fit(fit_function, group['pave'], group['xsec_red_unfolded']-group['total_uncert_unfolded'], sigma=group['total_uncert_unfolded'], absolute_sigma=True)


            # calculate the standard deviation of the fitted parameters
            perr = np.sqrt(np.diag(pcov))

            perr_top = popt_top - popt
            perr_bot = popt - popt_bot

            # calculate the top and bottom bounds of the fit
            phis = np.linspace(0,360,1000)
            y = fit_function(phis, *popt)
            #y_top = fit_function(phis, *(popt + perr))
            y_top = fit_function(phis, *(popt_top))#+np.sqrt(np.diag(pcov_top))))
            #y_bottom = fit_function(phis, *(popt - perr))
            y_bottom = fit_function(phis, *(popt_bot))#-np.sqrt(np.diag(pcov_bot))))

            plt.plot(phis, y, 'b-', label="Trig. Fit",linewidth=5)
            # # # plt.fill_between(phis, y_bottom, y_top, color='b', alpha=0.2)  # this adds the band of uncertainty

            num_bands = 500  # number of bands in the gradient
            cmap = plt.get_cmap('winter')  # get the colormap

            for i in range(num_bands):
                # Create a range of color values
                color_val = i / num_bands
                # Calculate the y value for this band
                y_band_top = y_bottom + (y_top - y_bottom) * (i+1) / num_bands
                y_band_bottom = y_bottom + (y_top - y_bottom) * i / num_bands
                # Plot the band with the color from the colormap
                plt.fill_between(phis, y_band_bottom, y_band_top, color=cmap(color_val),alpha=0.1)
            xerr_value = 0
            plt.errorbar(group['pave'], group['xsec_red_unfolded'], xerr=xerr_value,yerr=1.5*group['total_uncert_unfolded'],fmt='k.',  markersize=5,elinewidth=5,capsize=10, capthick=5)#elabel)
            xerr_value=9
            plt.errorbar(group['pave'], group['xsec_red_unfolded'], xerr=xerr_value,yerr=group['xsec_red_err_unfolded'],fmt='r.',  markersize=5,label="Unfolded",elinewidth=5)#,capsize=10, capthick=5)#elabel)
            xerr_value = 0
            plt.errorbar(group['pave'], group['xsec_red_unfolded'], xerr=xerr_value,yerr=0,fmt='r.',  markersize=5,elinewidth=5,capsize=10, capthick=5)#elabel)
            #plt.errorbar(group['pave'], group['xsec_red_unfolded'], xerr=xerr_value,yerr=group['xsec_red_err_alt'],fmt='g.',  markersize=5,label="Unfolded",elinewidth=5)#elabel)



            plt.plot(group['pave']-6, group['xsec_red'], marker='^', color='orange', markersize=21, label="Bin-by-Bin",linestyle="None")

            #             #get errors from covariance matrix
            # perr = np.sqrt(np.diag(pcov))
            # print(perr)


            # errband_width

            # #y_bottom = fit_function(phis, *popt)-errband_width
            # #instead, calc y_bottom with errors from covariance matrix
            
            # y_top = fit_function(phis, *popt)+errband_width

            # plt.fill_between(phi, y_bottom, y_top, color='b', alpha=.1)#, label='CLAS6 Fit')



            #blue_line = mlines.Line2D([], [], color='r', marker='None', markersize=10, linestyle='-', label=slabel)
            #red_line = mlines.Line2D([], [], color='k', marker='None', markersize=10, linestyle='-', label=elabel)


        ax.set_xlabel('Lepton-Hadron Angle $\phi$')
        if plot_ylabel:
            ax.set_ylabel('Reduced Cross Section (nb/$GeV^2$)')
        #set xaxis range from 0 to 360
        ax.set_xlim([0,360])
        #set y bottom to 0
        ax.set_ylim(bottom=0)
        #pltt = 'Reduced Cross Section in bin ({})={}'.format(r'$x_{B,min} Q^2_{min} t_{min}$',str(name))
        pltt = '({})={}'.format(r'$x_{B,min} Q^2_{min} t_{min}$',str(name))
        #instead make plot title as averages xave qave tave
        #plot_title = '({})=({:.2f}, {:.2f}, {:.2f})'.format(r'$\langle x_{B}\rangle, \langle Q^2 \rangle, \lange t \rangle$',group['xave'].mean(),group['qave'].mean(),group['tave'].mean())
        plot_title = '{}={:.2f},{}={:.2f} GeV$^2$,{}={:.2f} GeV$^2$'.format(r'$\langle x_{B}\rangle$',group['xave'].mean(), r'$\langle Q^2 \rangle$',group['qave'].mean(), r'$\langle t \rangle$',group['tave'].mean())


        #ax.set_title("Your Plot Title", y=0.95)
        ax.set_title(plot_title)#,y=0.94,bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))



    

            #blue_line = mlines.Line2D([], [], color='r', marker='None', markersize=10, linestyle='-', label=slabel)
            #red_line = mlines.Line2D([], [], color='k', marker='None', markersize=10, linestyle='-', label=elabel)



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



        fit_results.append([name[0],
        name[1],
        name[2],
        group['xmax'].values[0],
        group['qmax'].values[0],
        group['tmax'].values[0],
        group['xave'].mean(),
        group['qave'].mean(),
        group['tave'].mean(),    
        popt[0]/prefact_A,
        popt[1]/prefact_B,
        popt[2]/prefact_C,
        popt_top[0]/prefact_A,
        popt_top[1]/prefact_B,
        popt_top[2]/prefact_C,
        popt_bot[0]/prefact_A,
        popt_bot[1]/prefact_B,
        popt_bot[2]/prefact_C,
        c6_tel,
        c6_tt,
        c6_lt,
        c6_tel_err,
        c6_tt_err,
        c6_lt_err,
        t_C6
        ])

        #print popt values and uncertainties
        print("popt values are")
        print(popt)
        print("perr values are")
        print(popt_top)
        print(popt_bot)



        
        plt.ylim(bottom=0)
        #plt.show()
        #plt.show()
        plt.savefig("tdep_test_unfolded_tight/"+pltt+".png")#,bbox_inches='tight')
        
        #sys.exit()
        plt.close()
        # plt.show()
        # ind += 1
        # if ind > 20:
        #     break
            #plt.close()
fit_results_df = pd.DataFrame(fit_results,columns=['xmin', 'qmin', 'tmin', 'xmax', 'qmax', 'tmax', 'xave', 'qave',
                                                 'tave', 'A','B','C','A_uncert_top','B_uncert_top','C_uncert_top',
                                                 'A_uncert_bot','B_uncert_bot','C_uncert_bot',
                                                 'c6_tel','c6_tt','c6_lt','c6_tel_err','c6_tt_err','c6_lt_err','tave_c6'])
print(fit_results_df)
fit_results_df.to_pickle("t_dep_of_xsec_unfolded.pkl")
