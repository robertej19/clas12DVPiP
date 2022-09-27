import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
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

if 1==1:
    df_small = pd.read_csv('cross_section_pi0_575.txt', sep='\t', header=0)
    #df = pd.read_csv('cross_section_pi0_10600.txt', sep='\t', header=0)


    #print(df.columns)

    df_small['sigma_T'] = pd.to_numeric(df_small["sigma_T"], errors='coerce')
    df_small['sigma_L'] = pd.to_numeric(df_small["sigma_L"], errors='coerce')
    df_small['sigma_LT'] = pd.to_numeric(df_small["sigma_LT"], errors='coerce')
    df_small['sigma_TT'] = pd.to_numeric(df_small["sigma_TT"], errors='coerce')


    print(df_small)


    num_pre = 1#0.00004627006 # Include prefactor of alpha/16pi^2
    E_lepton = 10.6#5.75
    m = 0.938
    Q2 = 2.25
    s = E_lepton*2*m #E_lepton**2 + 2*E_lepton*m + m**2 ### Make sure this is correct
    df_small['prefactor'] = num_pre*(s-m**2)/(E_lepton**2*m**2*df_small['Q2']*(-1*df_small['epsilon']+1))

    base_query = "Q2==2.25 and xB==0.275 and mt<-0.24 and mt>-0.3"
    df_small2 = df_small.query(base_query)

    print(df_small2)

    #print(df_small2['prefactor'].values[0])

    df_clas6 = pd.read_csv('xs_clas6.csv', header=0)

    print(df_clas6.columns)

    base_query = "q>2 and q<2.5 and x>0.25 and x<0.3 and t>0.2 and t<0.3"

    df_2 = df_clas6.query(base_query)
    

    df_2['prefactor'] = df_small2['prefactor'].values[0]
    df_2['sigma_T'] = df_small2['sigma_T'].values[0]
    df_2['sigma_L'] = df_small2['sigma_L'].values[0]
    df_2['sigma_LT'] = df_small2['sigma_LT'].values[0]
    df_2['sigma_TT'] = df_small2['sigma_TT'].values[0]
    df_2['epsilon'] = df_small2['epsilon'].values[0]
    df_2['total_xsection'] =  df_2['prefactor']*(df_2['sigma_T']+df_2['epsilon']*df_2['sigma_L']+df_2['epsilon']*np.cos(2*df_2['p']*3.14159/180)*df_2['sigma_TT']+np.sqrt(2*df_2['epsilon']*(1+df_2['epsilon']))*np.cos(df_2['p'])*df_2['sigma_LT'])
    df_2['diff_xsection'] = df_2['dsdtdp']/df_2['total_xsection']


    plt.plot(df_2['p'], df_2['dsdtdp'],'o')
    plt.plot(df_2['p'], df_2['total_xsection'],'r')
    plt.show()



    print(df_2)






# Basic plotting
if 0==1:

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


    plt.plot(-1*df_small['mt'], df_small['epsilon'],'o')

    # plt.plot(-1*df_small['mt'], df_small['sigma_L'],'+')
    # plt.plot(-1*df_small['mt'], df_small['sigma_LT'],'ro')
    # plt.plot(-1*df_small['mt'], df_small['sigma_TT'],'r+')




    df = pd.read_csv('cross_section_pi0_10600.txt', sep='\t', header=0)


    print(df.columns)

    df_small = df.query(base_query)

    df_small['sigma_T'] = pd.to_numeric(df["sigma_T"], errors='coerce')
    df_small['sigma_L'] = pd.to_numeric(df["sigma_L"], errors='coerce')
    df_small['sigma_LT'] = pd.to_numeric(df["sigma_LT"], errors='coerce')
    df_small['sigma_TT'] = pd.to_numeric(df["sigma_TT"], errors='coerce')


    print(df_small)
    df_small.dropna()
    print(df_small)

    #fig, ax = plt.subplots(figsize=(12, 6))

    plt.plot(-1*df_small['mt'], df_small['epsilon'],'k')

    #plt.plot(-1*df_small['mt'], df_small['sigma_T']+df_small['epsilon']*df_small['sigma_L'],'k')
    #plt.plot(-1*df_small['mt'], df_small['sigma_T'],'r')

    #plt.plot(-1*df_small['mt'], df_small['sigma_L'],'yo')
    plt.plot(-1*df_small['mt'], df_small['sigma_LT'],'r')
    plt.plot(-1*df_small['mt'], df_small['sigma_TT'],'b')





    #plt.ylim([-300,400])
    plt.ylim([-3,1])

    plt.xlim([0,2])


    plt.show()

    print(df_small)