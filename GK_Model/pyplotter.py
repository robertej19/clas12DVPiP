import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.patches import Rectangle

# # # simcounts = 1000000

# # # mu, sigma = 1, 1.5 # mean and standard deviation
# # # s = np.random.normal(mu, sigma, simcounts)

# # # q=s
# # # #q = np.sqrt(np.square(s))
# # # #q = s*s-2*s-1

# # # ocunter = 0
# # # for ii in q:
# # #     if ii>(1+np.sqrt(2)):
# # #         ocunter += 1
# # #     else:
# # #         ocunter += 0


# # # print(ocunter/simcounts)

# # # sys.exit()
M = 0.938272081 # target mass
me = 0.5109989461 * 0.001 # electron mass
ebeam = 10.604 # beam energy
pbeam = np.sqrt(ebeam * ebeam - me * me) # beam electron momentum
beam = [0, 0, pbeam] # beam vector
target = [0, 0, 0] # target vector
alpha = 1/137 #Fund const
mp = 0.938 #Mass proton
prefix = alpha/(8*np.pi)
E = 10.6
Clas6_Sim_BeamTime = 11922445
Clas12_Sim_BeamTime = 16047494
Clas12_exp_luminosity = 5.5E40


def get_gamma(x,q2,BeamE):
    a8p = 1/137*(1/(8*3.14159))
    energies = [BeamE]
    for e in energies:
        y = q2/(2*x*e*mp)
        num = 1-y-q2/(4*e*e)
        denom = 1- y + y*y/2 + q2/(4*e*e)
        #print(y,q2,e,num,denom)
        epsi = num/denom
        gamma = 1/(e*e)*(1/(1-epsi))*(1-x)/(x*x*x)*a8p*q2/(0.938*.938)

    return [gamma, epsi]


i=1 

if i==1:
    df_small = pd.read_csv('cross_section_pi0_575.txt', sep='\t', header=0)
    #df = pd.read_csv('cross_section_pi0_10600.txt', sep='\t', header=0)


    #print(df.columns)

    df_small['sigma_T'] = pd.to_numeric(df_small["sigma_T"], errors='coerce')
    df_small['sigma_L'] = pd.to_numeric(df_small["sigma_L"], errors='coerce')
    df_small['sigma_LT'] = pd.to_numeric(df_small["sigma_LT"], errors='coerce')
    df_small['sigma_TT'] = pd.to_numeric(df_small["sigma_TT"], errors='coerce')




    num_pre = 0.00004627006 # Include prefactor of alpha/16pi^2
    E_lepton = 5.75
    #E_lepton = 10.6
    m = 0.938
    Q2 = 2.25
    s = E_lepton*2*m #E_lepton**2 + 2*E_lepton*m + m**2 ### Make sure this is correct
    W = 2.3564636
    df_small['prefactor'] = num_pre*(W**2-m**2)/(E_lepton**2*m**2*df_small['Q2']*(-1*df_small['epsilon']+1))

    xb = 0.34
    epsilon = 0.587074 # For 5.75 GeV beam
    epsi_mean_c6 = epsilon
    #epsilon= 0.901577 #for 10.6 GeV beam

    xbstuff = (1-xb)/(xb**3)
    epsilon_stuff = 1/(1-epsilon)

    Gamma = 1/137/8/3.14159/m**2/E_lepton**2*Q2*xbstuff*epsilon_stuff
    print(Gamma)
    print(df_small['prefactor']/Gamma)

    g, e = get_gamma(xb,Q2,E_lepton)
    print(g,e)
    print(g/Gamma)
    #sys.exit()


    base_query = "Q2==2.25 and xB==0.325 and mt<-0.24 and mt>-0.3"
    df_small2 = df_small.query(base_query)

    print(df_small2)

    #print(df_small2['prefactor'].values[0])

    df_clas6 = pd.read_csv('xs_clas6.csv', header=0)

    print(df_clas6.columns)

    base_query = "q>2 and q<2.5 and x>0.3 and x<0.38 and t>0.2 and t<0.3"

    df_2 = df_clas6.query(base_query)
    print(df_2)

    df_2['prefactor'] = df_small2['prefactor'].values[0]
    df_2['sigma_T'] = df_small2['sigma_T'].values[0]
    df_2['sigma_L'] = df_small2['sigma_L'].values[0]
    df_2['sigma_LT'] = df_small2['sigma_LT'].values[0]
    df_2['sigma_TT'] = df_small2['sigma_TT'].values[0]
    df_2['epsilon'] = df_small2['epsilon'].values[0]
    df_2['total_xsection'] =  df_2['prefactor']*(df_2['sigma_T']+df_2['epsilon']*df_2['sigma_L']+df_2['epsilon']*np.cos(2*df_2['p']*3.14159/180)*df_2['sigma_TT']+np.sqrt(2*df_2['epsilon']*(1+df_2['epsilon']))*np.cos(df_2['p']*3.14159/180)*df_2['sigma_LT'])
    df_2['diff_xsection'] = df_2['dsdtdp']/df_2['total_xsection']
    #df_2['p_prime0']

    plt.rcParams["font.size"] = "20"


    fig, ax = plt.subplots(figsize =(18, 10)) 


    plt.plot(df_2['p'], df_2['dsdtdp'],'o')
    #plt.plot(df_2['p'], df_2['total_xsection'],'r+')
    # df_2['sigma_T'] = df_small2['sigma_T'].values[0]
    # df_2['sigma_L'] = df_small2['sigma_L'].values[0]
    # df_2['sigma_LT'] = df_small2['sigma_LT'].values[0]
    # df_2['sigma_TT'] = df_small2['sigma_TT'].values[0]
    # df_2['epsilon'] = df_small2['epsilon'].values[0]
    phi = np.linspace(0, 360, 100)

    # pub_tel =  389.766081871345
    # pub_lt = -13.45029239766086
    # pub_tt = -46.19883040935679

    pub_tel =  df_small2['sigma_L'].values[0]*df_small2['epsilon'].values[0]+df_small2['sigma_T'].values[0]
    pub_lt = df_small2['sigma_LT'].values[0]
    pub_tt = df_small2['sigma_TT'].values[0]



    def fit_function(phi,A,B,C):
            #A + B*np.cos(2*phi) +C*np.cos(phi)
            rads = phi*np.pi/180
            #return (A * np.exp(-x/beta) + B * np.exp(-1.0 * (x - mu)**2 / (2 * sigma**2)))
            #A = T+L, B=TT, C=LT
            #A = black, B=blue, C=red
            return A + B*np.cos(2*rads) + C*np.cos(rads)

    rev_a =pub_tel/6.28
    rev_b = pub_tt/6.28*epsi_mean_c6
    rev_c = pub_lt/6.28*np.sqrt(2*epsi_mean_c6*(1+epsi_mean_c6))

    fit_y_data_weighted = fit_function(phi,rev_a,rev_b,rev_c)

    #total_xsection = 
    #total_xsection =  3.14*6.28*1/Gamma*df_small2['prefactor'].values[0]*(df_small2['sigma_T'].values[0]+df_small2['epsilon'].values[0]*df_small2['sigma_L'].values[0]+df_small2['epsilon'].values[0]*np.cos(2*phi*3.14159/180)*df_small2['sigma_TT'].values[0]+np.sqrt(2*df_small2['epsilon'].values[0]*(1+df_small2['epsilon'].values[0]))*np.cos(phi*3.14159/180)*df_small2['sigma_LT'].values[0])
    total_xsection =  1/6.28*(df_small2['sigma_T'].values[0]+df_small2['epsilon'].values[0]*df_small2['sigma_L'].values[0]+df_small2['epsilon'].values[0]*np.cos(2*phi*3.14159/180)*df_small2['sigma_TT'].values[0]+np.sqrt(2*df_small2['epsilon'].values[0]*(1+df_small2['epsilon'].values[0]))*np.cos(phi*3.14159/180)*df_small2['sigma_LT'].values[0])

    print(df_small2['prefactor'].values[0])
    print(Gamma)
    plt.plot(phi, fit_y_data_weighted,'r')
    plt.plot(phi, total_xsection,'b')

    




    plot_title = "CLAS6 2014 Published Result vs. 2022 Implementation"
    ax.set_xlabel('$\phi$ ')  
    ax.set_ylabel('d$\sigma^4$/dQ$^2$dx$_B$dtd$\phi$ (nb/GeV$^4$)')
    plt.title(plot_title)

    plt.show()
    print(pub_tel)
    print(pub_lt)
    print(pub_tt)


    #print(df_2)

    # c6_dtedl = 0.2511627906976744, 389.766081871345
    # c6_dlt = 0.2511627906976744, -13.45029239766086
    # c6_dtt = 0.2511627906976744, -46.19883040935679




# Basic plotting
if i==2:

    df = pd.read_csv('cross_section_pi0_575.txt', sep='\t', header=0)
    #df = pd.read_csv('cross_section_pi0_10600.txt', sep='\t', header=0)


    print(df.columns)
    base_query = "Q2==2.25 and xB==0.325"
    df_small = df.query(base_query)

    df_small['sigma_T'] = pd.to_numeric(df["sigma_T"], errors='coerce')
    df_small['sigma_L'] = pd.to_numeric(df["sigma_L"], errors='coerce')
    df_small['sigma_LT'] = pd.to_numeric(df["sigma_LT"], errors='coerce')
    df_small['sigma_TT'] = pd.to_numeric(df["sigma_TT"], errors='coerce')


    print(df_small)
    df_small.dropna()
    print(df_small)
    plt.rcParams["font.size"] = "20"


    fig, ax = plt.subplots(figsize =(18, 10)) 



    dtedl_2022 = plt.plot(-1*df_small['mt'], df_small['sigma_T']+df_small['epsilon']*df_small['sigma_L'],'k--',label="2022 GK Model")

    plt.plot(-1*df_small['mt'], df_small['sigma_LT'],'r--')
    plt.plot(-1*df_small['mt'], df_small['sigma_TT'],'b--')


    # df = pd.read_csv('cross_section_pi0_10600.txt', sep='\t', header=0)


    # print(df.columns)

    # df_small = df.query(base_query)

    # df_small['sigma_T'] = pd.to_numeric(df["sigma_T"], errors='coerce')
    # df_small['sigma_L'] = pd.to_numeric(df["sigma_L"], errors='coerce')
    # df_small['sigma_LT'] = pd.to_numeric(df["sigma_LT"], errors='coerce')
    # df_small['sigma_TT'] = pd.to_numeric(df["sigma_TT"], errors='coerce')


    # print(df_small)
    # df_small.dropna()
    # print(df_small)

    # #fig, ax = plt.subplots(figsize=(12, 6))

    # plt.plot(-1*df_small['mt'], df_small['epsilon'],'k')

    # #plt.plot(-1*df_small['mt'], df_small['sigma_T']+df_small['epsilon']*df_small['sigma_L'],'k')
    # #plt.plot(-1*df_small['mt'], df_small['sigma_T'],'r')

    # #plt.plot(-1*df_small['mt'], df_small['sigma_L'],'yo')
    # plt.plot(-1*df_small['mt'], df_small['sigma_LT'],'r')
    # plt.plot(-1*df_small['mt'], df_small['sigma_TT'],'b')








    #plt.show()

    print(df_small)


if i==2:

# Grabbed data from https://apps.automeris.io/wpd/
# From Fig 24 in Ivan 2014
# 2 < Q2 < 2.5 0.3<xb < 0.38

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

