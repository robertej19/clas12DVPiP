import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.patches import Rectangle

# 1.) Necessary imports.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import argparse
import sys
import pandas as pd
from matplotlib.patches import Rectangle

#UNCHANGING CONSTANTS
M = 0.938272081 # target mass
me = 0.5109989461 * 0.001 # electron mass
ebeam = 10.604 # beam energy
pbeam = np.sqrt(ebeam * ebeam - me * me) # beam electron momentum
beam = [0, 0, pbeam] # beam vector
target = [0, 0, 0] # target vector
alpha = 1/137 #Fund const
prefix = alpha/(8*np.pi)
E = 10.6
Clas6_Sim_BeamTime = 11922445
Clas12_Sim_BeamTime = 16047494
Clas12_exp_luminosity = 5.5E40


def get_gamma(x,q2,BeamE):
    a8p = 1/137*(1/(8*3.14159))
    energies = [BeamE]
    for e in energies:
        y = q2/(2*x*e*M)
        num = 1-y-q2/(4*e*e)
        denom = 1- y + y*y/2 + q2/(4*e*e)
        #print(y,q2,e,num,denom)
        epsi = num/denom
        gamma = 1/(e*e)*(1/(1-epsi))*(1-x)/(x*x*x)*a8p*q2/(0.938*.938)

    return [gamma, epsi]


def fit_function(phi,A,B,C):
        #A + B*np.cos(2*phi) +C*np.cos(phi)
        rads = phi*np.pi/180
        #return (A * np.exp(-x/beta) + B * np.exp(-1.0 * (x - mu)**2 / (2 * sigma**2)))
        #A = T+L, B=TT, C=LT
        #A = black, B=blue, C=red
        return A + B*np.cos(2*rads) + C*np.cos(rads)


# def resid_weighted_c12(pars):
#     return (((y-fit_function(x,pars))**2)/sigma_c12).sum()

def constr0(pars):
    return fit_function(0,pars)

def constr180(pars):
    return fit_function(180,pars)

def conv_struct_to_abc(tel,lt,tt,epsilon):

    a =tel/6.28
    b = tt/6.28*epsilon
    c = lt/6.28*np.sqrt(2*epsilon*(1+epsilon))

    return [a,b,c]

def conv_abc_to_struct(a,b,c,epsilon):

    tel = a*6.28
    tt = b/epsilon*6.28
    lt = c/np.sqrt(2*epsilon*(1+epsilon))*6.28

    return [tel,tt,lt]

def conv_abc_err_to_struct_err(tel,tt,lt,a,b,c,a_err,b_err,c_err):

    tel_err = tel*a_err/a
    tt_err = tt*b_err/b
    lt_err = lt*c_err/c

    return [tel_err,tt_err,lt_err]





def fit_over_phi(x_data,y_data,y_errors,weighted=True):

    valid = ~(np.isnan(x_data) | np.isnan(y_data))

    if weighted:
        popt_0, pcov = curve_fit(fit_function, xdata=x_data[valid], ydata=y_data[valid], p0=[100,-60,-11],
                    sigma=y_errors[valid], absolute_sigma=True)

        popt, pcov = curve_fit(fit_function, xdata=x_data[valid], ydata=y_data[valid], p0=[popt_0[0],popt_0[1],popt_0[2]],
                    sigma=y_errors[valid], absolute_sigma=True)
    else:
        popt_0, pcov = curve_fit(fit_function, xdata=x_data[valid], ydata=y_data[valid], p0=[100,-60,-11])

        popt, pcov = curve_fit(fit_function, xdata=x_data[valid], ydata=y_data[valid], p0=[popt_0[0],popt_0[1],popt_0[2]])

    a,b,c = popt[0],popt[1],popt[2]

    a_err = np.sqrt(pcov[0][0])#*qmod
    b_err = np.sqrt(pcov[1][1])#*qmod
    c_err = np.sqrt(pcov[2][2])#*qmod

    #for constraint fitting, not used with scipy curve_fit
    # con1 = {'type': 'ineq', 'fun': constr0}
    # con2 = {'type': 'ineq', 'fun': constr180}
    # # con3 = {'type': 'ineq', 'fun': constr270}
    # cons = [con1,con2]

    return [a,b,c,a_err,b_err,c_err]




i=1

def comp_gk_c12_c6(qmin=1.5,qmax=2,xmin=0.2,xmax=0.25,tmin=0.2,tmax=0.3,plot_CLAS6=False):

    # INPUT BINNING
    E_beam_6 = 5.75 #Beam energy in GeV
    E_beam_12 = 10.6 #Beam energy in GeV

    Q2 = (qmin+qmax)/2
    xB = (xmin+xmax)/2
    t = (tmin+tmax)/2

    df_GK_calc = pd.read_csv('cross_section_pi0_575.txt', sep='\t', header=0)
    # Data Structure:
    #     Q2	xB	mt	sigma_T	sigma_L	sigma_LT	sigma_TT	W	y	epsilon	gammaa	tmin
    #  1.75 	 0.225 	 -0.020 	 nan 	 nan 	 -nan 	 nan 	 2.6282355 	 0.0806671 	 0.9961151 	 0.3190776 	 -0.0574737


    df_clas6 = pd.read_csv('xs_clas6.csv', header=0)
    # Data Structure:
    # q	x	t	p	dsdtdp	stat	sys
    # 1.15	0.132	0.12	63	59.4	15.3	13

    df_inbend_clas12 = pd.read_pickle('/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_inbending_rad_All_All_All_rad_f18_in_and_out_advanced_no_ang_cuts.pkl')
    df_inbend_clas12 = pd.read_pickle('/mnt/d/GLOBUS/CLAS12/APS2022/final_data_files/full_xsection_outbending_rad_All_All_All_rad_f18_in_and_out_advanced_no_ang_cuts.pkl')
    # Data Structure:
    #['qmin', 'xmin', 'tmin', 'pmin', 'qmax_x', 'xmax_x', 'tmax_x', 'pmax_x',
    #    'qave_exp', 'yave_x', 'xave_exp', 'tave_exp', 'pave_exp', 'counts_exp',
    #    'qmax_y', 'xmax_y', 'tmax_y', 'pmax_y', 'qave_rec', 'yave_y',
    #    'xave_rec', 'tave_rec', 'pave_rec', 'counts_rec', 'qmax', 'xmax',
    #    'tmax', 'pmax', 'qave_gen', 'yave', 'xave_gen', 'tave_gen', 'pave_gen',
    #    'counts_gen', 'gamma_exp', 'epsi_exp', 'binvol', 'acc_corr', 'xsec',
    #    'xsec_corr', 'xsec_corr_red', 'xsec_corr_red_nb', 'uncert_counts_exp',
    #    'uncert_counts_rec', 'uncert_counts_gen', 'uncert_xsec',
    #    'uncert_acc_corr', 'uncert_xsec_corr_red_nb']



    #Convert rows to be of type numeric
    df_GK_calc['sigma_T'] = pd.to_numeric(df_GK_calc["sigma_T"], errors='coerce')
    df_GK_calc['sigma_L'] = pd.to_numeric(df_GK_calc["sigma_L"], errors='coerce')
    df_GK_calc['sigma_LT'] = pd.to_numeric(df_GK_calc["sigma_LT"], errors='coerce')
    df_GK_calc['sigma_TT'] = pd.to_numeric(df_GK_calc["sigma_TT"], errors='coerce')





    # Note: This needs to calculated directly better. Discrepancy between funion above and that eMloyed in Pi0_GK code
    Gamma, epsilon_6 = get_gamma(xB,Q2,E_beam_6)
    Gamma_12, epsilon_12 = get_gamma(xB,Q2,E_beam_12)



    #prefactor not needed for reduced cross section
    #num_pre = 0.00004627006 # Include prefactor of alpha/16pi^2
    #W = 2.3564636 #calculated from some stuff, need to fix !!!!!!!!!!!!!!!!!!!!!!!!!
    #df_GK_calc['prefactor'] = num_pre*(W**2-M**2)/(E_beam_6**2*M**2*df_GK_calc['Q2']*(-1*df_GK_calc['epsilon']+1))



    query_clas6 = "q>={} and q<={} and x>={} and x<={} and t>={} and t<={}".format(qmin,qmax,xmin,xmax,tmin,tmax)
    query_GK_Model = "Q2>={} and Q2<={} and xB>={} and xB<={} and mt<={} and mt>={} ".format(qmin,qmax,xmin,xmax,-1*tmin,-1*tmax)
    query_clas12 = "qmin>={} and qmax<={} and xmin>={} and xmax<={} and tmin>={} and tmax<={}".format(qmin,qmax,xmin,xmax,tmin,tmax)

    df_GK_reduced = df_GK_calc.query(query_GK_Model)
    df_clas6_reduced = df_clas6.query(query_clas6)
    df_inbend_clas12_reduced = df_inbend_clas12.query(query_clas12)


    print(df_GK_reduced)
    print(df_GK_reduced['sigma_TT'].mean())

    #df_clas6_reduced['prefactor'] = df_GK_reduced['prefactor'].mean()
    df_clas6_reduced['sigma_T'] = df_GK_reduced['sigma_T'].mean()
    df_clas6_reduced['sigma_L'] = df_GK_reduced['sigma_L'].mean()
    df_clas6_reduced['sigma_LT'] = df_GK_reduced['sigma_LT'].mean()
    df_clas6_reduced['sigma_TT'] = df_GK_reduced['sigma_TT'].mean()
    #df_clas6_reduced['epsilon'] = df_GK_reduced['epsilon'].mean()
    #df_clas6_reduced['total_xsection'] =  df_clas6_reduced['prefactor']*(df_clas6_reduced['sigma_T']+df_clas6_reduced['epsilon']*df_clas6_reduced['sigma_L']+df_clas6_reduced['epsilon']*np.cos(2*df_clas6_reduced['p']*3.14159/180)*df_clas6_reduced['sigma_TT']+np.sqrt(2*df_clas6_reduced['epsilon']*(1+df_clas6_reduced['epsilon']))*np.cos(df_clas6_reduced['p']*3.14159/180)*df_clas6_reduced['sigma_LT'])
    #df_clas6_reduced['diff_xsection'] = df_clas6_reduced['dsdtdp']/df_clas6_reduced['total_xsection']


    sigma_c6 = np.sqrt(np.square(df_clas6_reduced['stat'])+np.square(df_clas6_reduced['sys']))


    #Published CLAS6 fit from ... somewhere, should replace with own fit / coMare
    # pub_tel =  389.766081871345
    # pub_lt = -13.45029239766086
    # pub_tt = -46.19883040935679

    # [rev_a,rev_b,rev_c] = conv_struct_to_abc(pub_tel,pub_lt,pub_tt,epsilon_6)
    #fit_y_data_weighted = fit_function(phi_vector,rev_a,rev_b,rev_c)


    phi_vector = np.linspace(0, 360, 100)

    #Plot CLAS6 fit

    #GK_curve_c12 =  1/6.28*(df_GK_reduced['sigma_T'].mean()+epsilon_12*df_GK_reduced['sigma_L'].mean()+epsilon_12*np.cos(2*phi_vector*3.14159/180)*df_GK_reduced['sigma_TT'].mean()+np.sqrt(2*epsilon_12*(1+epsilon_12))*np.cos(phi_vector*3.14159/180)*df_GK_reduced['sigma_LT'].mean())
    #GK_curve_c6 =  1/6.28*(df_GK_reduced['sigma_T'].mean()+epsilon_6*df_GK_reduced['sigma_L'].mean()+epsilon_6*np.cos(2*phi_vector*3.14159/180)*df_GK_reduced['sigma_TT'].mean()+np.sqrt(2*epsilon_6*(1+epsilon_6))*np.cos(phi_vector*3.14159/180)*df_GK_reduced['sigma_LT'].mean())

    GK_curve_c12 = fit_function(phi_vector,*conv_struct_to_abc(df_GK_reduced['sigma_T'].mean()+epsilon_12*df_GK_reduced['sigma_L'].mean(),df_GK_reduced['sigma_LT'].mean(),df_GK_reduced['sigma_TT'].mean(),epsilon_12))
    GK_curve_c6 = fit_function(phi_vector,*conv_struct_to_abc(df_GK_reduced['sigma_T'].mean()+epsilon_6*df_GK_reduced['sigma_L'].mean(),df_GK_reduced['sigma_LT'].mean(),df_GK_reduced['sigma_TT'].mean(),epsilon_6))


    df_inbend_clas12 = df_inbend_clas12.dropna()


    binscenters_c12 = df_inbend_clas12_reduced["pave_exp"]
    data_entries_c12 = df_inbend_clas12_reduced["xsec_corr_red_nb"]
    sigma_c12 = df_inbend_clas12_reduced["uncert_xsec_corr_red_nb"]



    ###A +    Bcos(2x) + Ccos(x)
    ###TEL +   ep*TT   + sqr*LT
    [a_weighted,b_weighted,c_weighted,a_err_weighted,b_err_weighted,c_err_weighted] = fit_over_phi(binscenters_c12,data_entries_c12,sigma_c12,weighted=True)
    [a_unweighted,b_unweighted,c_unweighted,a_err_unweighted,b_err_unweighted,c_err_unweighted] = fit_over_phi(binscenters_c12,data_entries_c12,sigma_c12,weighted=False)

    [a_weighted6,b_weighted6,c_weighted6,a_err_weighted6,b_err_weighted6,c_err_weighted6] = fit_over_phi(df_clas6_reduced['p'], df_clas6_reduced['dsdtdp'],sigma_c6,weighted=True)
    [a_unweighted6,b_unweighted6,c_unweighted6,a_err_unweighted6,b_err_unweighted6,c_err_unweighted6] = fit_over_phi(df_clas6_reduced['p'], df_clas6_reduced['dsdtdp'],sigma_c6,weighted=False)



    fit_c12_weighted = fit_function(phi_vector, a_weighted,b_weighted,c_weighted)
    fit_c12_unweighted = fit_function(phi_vector, a_unweighted,b_unweighted,c_unweighted)

    fit_c6_weighted = fit_function(phi_vector, a_weighted6,b_weighted6,c_weighted6)
    fit_c6_unweighted = fit_function(phi_vector, a_unweighted6,b_unweighted6,c_unweighted6)

    # Plotting the data


    # Create figure
    plt.rcParams["font.size"] = "20"
    fig, ax = plt.subplots(figsize =(14, 10))
    plot_title = "Reduced Cross Section at {}<Q$^2$<{}, {}<x$_B$<{}, {}<t<{}".format(qmin,qmax,xmin,xmax,tmin,tmax)
    plt.title(plot_title)
    plt.ylim([0,np.max([np.max(GK_curve_c12),np.max(fit_c12_weighted)])*1.3])
    ax.set_xlabel('$\phi$ ')
    ax.set_ylabel(r'$\frac{d\sigma^2}{dtd\phi}$'+ '  (nb/GeV$^2$)')


    if plot_CLAS6:
        #CLAS 6:
        #Plot CLAS6 datapoints
        plt.errorbar(df_clas6_reduced['p'], df_clas6_reduced['dsdtdp'],yerr=sigma_c6,color="black",fmt="o", markersize=10,label='CLAS6 Data')

        #Plot CLAS6 Data Fit
        plt.plot(phi_vector, fit_c6_weighted,'k',label="CLAS6 Data Fit Weighted")
        #plt.plot(phi_vector, fit_c6_unweighted,'k.',label="CLAS6 Data Fit Unweighted")

        #Plot CLAS6 GK Fit
        plt.plot(phi_vector, GK_curve_c6,'k--',label='GK Model 5.75 GeV Beam')

    #CLAS 12:
    #Plot CLAS12 datapoints
    plt.errorbar(df_inbend_clas12_reduced['pave_exp'], df_inbend_clas12_reduced['xsec_corr_red_nb'],yerr=sigma_c12,color="red",fmt="D", markersize=10,label='CLAS12 Data')

    #Plot CLAS12 Data Fit
    plt.plot(phi_vector, fit_c12_weighted,'r',label="CLAS12 Data Fit Weighted")
    #plt.plot(phi_vector, fit_c12_unweighted,'r.',label="CLAS12 Data Fit Unweighted")

    #Plot GK Model
    plt.plot(phi_vector, GK_curve_c12,'k--',label='GK Model 10.6 GeV Beam')


    ax.legend()#[dtedl_2022,dtedl_2014,extra], ("2022 GK fit","2014 GK fit","+ Data",))

    #plt.show()
    plt.savefig("gk_c12_c6_plots/reduced_xsec_{}_{}_{}_{}_{}_{}.png".format(qmin,qmax,xmin,xmax,tmin,tmax))
    plt.close()



qmins = [1.5,2]
qmaxs = [2,2.5]
xmins = [0.2,0.25,0.3]
xmaxs = [0.25,0.3,0.38]
tmins = [0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0]
tmaxs = [0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.5]

for qmi,qma,in zip(qmins,qmaxs):
    for xmi,xma in zip(xmins,xmaxs):
        for tmi,tma in zip(tmins,tmaxs):
            print("on {} {} {} {} {} {}".format(qmi,qma,xmi,xma,tmi,tma))
            try:
                comp_gk_c12_c6(qmi,qma,xmi,xma,tmi,tma)
            except Exception as e:
                print(e)
                pass
#comp_gk_c12_c6(qmi,qma,xmi,xma,tmi,tma)
#comp_gk_c12_c6()
    # in=1.5,qmax=2,xmin=0.2,xmax=0.25,tmin=0.2,tmax=0.3


# Basic plotting
if i==2:

    df = pd.read_csv('cross_section_pi0_575.txt', sep='\t', header=0)
    #df = pd.read_csv('cross_section_pi0_10600.txt', sep='\t', header=0)


    print(df.columns)
    query_GK_Model = "Q2==2.25 and xB==0.325"
    df_GK_calc = df.query(query_GK_Model)

    df_GK_calc['sigma_T'] = pd.to_numeric(df["sigma_T"], errors='coerce')
    df_GK_calc['sigma_L'] = pd.to_numeric(df["sigma_L"], errors='coerce')
    df_GK_calc['sigma_LT'] = pd.to_numeric(df["sigma_LT"], errors='coerce')
    df_GK_calc['sigma_TT'] = pd.to_numeric(df["sigma_TT"], errors='coerce')


    print(df_GK_calc)
    df_GK_calc.dropna()
    print(df_GK_calc)
    plt.rcParams["font.size"] = "20"


    fig, ax = plt.subplots(figsize =(18, 10))



    dtedl_2022 = plt.plot(-1*df_GK_calc['mt'], df_GK_calc['sigma_T']+df_GK_calc['epsilon']*df_GK_calc['sigma_L'],'k--',label="2022 GK Model")

    plt.plot(-1*df_GK_calc['mt'], df_GK_calc['sigma_LT'],'r--')
    plt.plot(-1*df_GK_calc['mt'], df_GK_calc['sigma_TT'],'b--')


    # df = pd.read_csv('cross_section_pi0_10600.txt', sep='\t', header=0)


    # print(df.columns)

    # df_GK_calc = df.query(query_GK_Model)

    # df_GK_calc['sigma_T'] = pd.to_numeric(df["sigma_T"], errors='coerce')
    # df_GK_calc['sigma_L'] = pd.to_numeric(df["sigma_L"], errors='coerce')
    # df_GK_calc['sigma_LT'] = pd.to_numeric(df["sigma_LT"], errors='coerce')
    # df_GK_calc['sigma_TT'] = pd.to_numeric(df["sigma_TT"], errors='coerce')


    # print(df_GK_calc)
    # df_GK_calc.dropna()
    # print(df_GK_calc)

    # #fig, ax = plt.subplots(figsize=(12, 6))

    # plt.plot(-1*df_GK_calc['mt'], df_GK_calc['epsilon'],'k')

    # #plt.plot(-1*df_GK_calc['mt'], df_GK_calc['sigma_T']+df_GK_calc['epsilon']*df_GK_calc['sigma_L'],'k')
    # #plt.plot(-1*df_GK_calc['mt'], df_GK_calc['sigma_T'],'r')

    # #plt.plot(-1*df_GK_calc['mt'], df_GK_calc['sigma_L'],'yo')
    # plt.plot(-1*df_GK_calc['mt'], df_GK_calc['sigma_LT'],'r')
    # plt.plot(-1*df_GK_calc['mt'], df_GK_calc['sigma_TT'],'b')








    #plt.show()

    print(df_GK_calc)


if i==2:

# Grabbed data from https://apps.automeris.io/wpd/
# From Fig 24 in Ivan 2014
# 2 < Q2 < 2.5 0.3<xB <= 0.38

    dtedl = pd.read_csv('c6_gk_comp/dtedl.txt', sep=',', header=None)
    dtt = pd.read_csv('c6_gk_comp/dtt.txt', sep=',', header=None)
    dlt = pd.read_csv('c6_gk_comp/dlt.txt', sep=',', header=None)
    c6_data_dtedl = pd.read_csv('c6_gk_comp/c6_data_dtedl.txt', sep=',', header=None)
    c6_data_dtt = pd.read_csv('c6_gk_comp/c6_data_dtt.txt', sep=',', header=None)
    c6_data_dlt = pd.read_csv('c6_gk_comp/c6_data_dlt.txt', sep=',', header=None)


    model = [dtedl, dtt, dlt]
    data = [c6_data_dtedl, c6_data_dtt, c6_data_dlt]

    colors = ['k','b','r']

    # for m,c in zip(model,colors):
    #     plt.plot(m[0], m[1],c)
    dtedl_2014 = plt.plot(dtedl[0], dtedl[1],'k',label="2014 GK Model")
    plt.plot(dtt[0], dtt[1],'b')
    plt.plot(dlt[0], dlt[1],'r')


    for d,c in zip(data,colors):
        plt.plot(d[0], d[1],c+'+')

    clas6_data = plt.plot(data[0][0], data[0][1],'k+',label="CLAS6 Data")


    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)


    ax.legend()#[dtedl_2022,dtedl_2014,extra], ("2022 GK fit","2014 GK fit","+ Data",))
    plt.ylim([-300,400])
    #plt.ylim([-3,1])

    plt.xlim([0,1.5])


    plot_title = "CLAS6 2014 Published Result vs. 2022 Implementation"
    ax.set_xlabel('-t (GeV$^2$)')
    ax.set_ylabel('d$\sigma$/dt (nb/GeV$^2$)')
    plt.title(plot_title)


    plt.show()

    print(dtedl)

