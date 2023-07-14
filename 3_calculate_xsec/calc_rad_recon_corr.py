import pandas as pd
import numpy as np
import os, sys
from icecream import ic

from utils import filestruct

#pd.set_option('mode.chained_assignment', None)

fs = filestruct.fs()

rec_45na_inb = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/rec/singles_t2/binned_dvpip_events_norad_10000_20230705_1043_Fall_2018_Inbending_45nA_recon.pkl"
rec_55na_inb = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/rec/singles_t2/binned_dvpip_events_norad_10000_20230705_1046_Fall_2018_Inbending_55nA_recon.pkl"
rec_nominal_1 = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/rec/singles_t2/binned_dvpip_events_norad_10000_20230703_1814_Fall_2018_Inbending_50nA_recon.pkl"
rec_nominal_2 = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/rec/singles_t2/binned_dvpip_events_norad_10000_20230705_1041_Fall_2018_Inbending_50nA_recon.pkl"
rec_rad = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/rec_rad/final_f18_inb_rec_binned_rad_t2.pkl"


gen_45na_inb = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/gen/final_f18_gen_45na_inb_binned.pkl"
gen_55na_inb = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/gen/final_f18_gen_55na_inb_binned.pkl"
gen_nominal_1 = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/gen/final_f18_gen_nominal_1_inb_binned.pkl"
gen_nominal_2 = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/gen/final_f18_gen_nominal_2_inb_binned.pkl"
gen_rad =  "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/gen_rad/final_f18_inb_gen_rad_binned.pkl"


df_45na = pd.read_pickle(rec_45na_inb)
#rename column 'counts' to 'counts_rec_45na'
df_45na.rename(columns={'counts':'counts_rec_45na'}, inplace=True)
#drop 'tave' 'pave' 'xave' 'qave' 'yave' from dataframe
df_45na.drop(['tave', 'pave', 'xave', 'qave', 'yave'], axis=1, inplace=True) 

df_55na = pd.read_pickle(rec_55na_inb)
#rename column 'counts' to 'counts_rec_55na'
df_55na.rename(columns={'counts':'counts_rec_55na'}, inplace=True)
df_55na.drop(['tave', 'pave', 'xave', 'qave', 'yave'], axis=1, inplace=True)

df_nominal_1 = pd.read_pickle(rec_nominal_1)
#rename column 'counts' to 'counts_rec_nominal_1'
df_nominal_1.rename(columns={'counts':'counts_rec_nominal_1'}, inplace=True)
df_nominal_1.drop(['tave', 'pave', 'xave', 'qave', 'yave'], axis=1, inplace=True)

df_nominal_2 = pd.read_pickle(rec_nominal_2)
#rename column 'counts' to 'counts_rec_nominal_2'
df_nominal_2.rename(columns={'counts':'counts_rec_nominal_2'}, inplace=True)
df_nominal_2.drop(['tave', 'pave', 'xave', 'qave', 'yave'], axis=1, inplace=True)

df_rad = pd.read_pickle(rec_rad)
#rename column 'counts' to 'counts_rec_rad'
df_rad.rename(columns={'rec_counts':'counts_rec_rad'}, inplace=True)
df_rad.drop(['rec_tave', 'rec_pave', 'rec_xave', 'rec_qave', 'rec_yave'], axis=1, inplace=True)

df_gen_45na = pd.read_pickle(gen_45na_inb)
#rename column 'counts' to 'counts_gen_45na'
df_gen_45na.rename(columns={'gen_counts':'counts_gen_45na'}, inplace=True)
df_gen_45na.drop(['gen_tave', 'gen_pave', 'gen_xave', 'gen_qave', 'gen_yave'], axis=1, inplace=True)

df_gen_55na = pd.read_pickle(gen_55na_inb)
#rename column 'counts' to 'counts_gen_55na'
df_gen_55na.rename(columns={'gen_counts':'counts_gen_55na'}, inplace=True)
df_gen_55na.drop(['gen_tave', 'gen_pave', 'gen_xave', 'gen_qave', 'gen_yave'], axis=1, inplace=True)


df_gen_nominal_1 = pd.read_pickle(gen_nominal_1)
#rename column 'counts' to 'counts_gen_nominal_1'
df_gen_nominal_1.rename(columns={'gen_counts':'counts_gen_nominal_1'}, inplace=True)
df_gen_nominal_1.drop(['gen_tave', 'gen_pave', 'gen_xave', 'gen_qave', 'gen_yave'], axis=1, inplace=True)

df_gen_nominal_2 = pd.read_pickle(gen_nominal_2)
#rename column 'counts' to 'counts_gen_nominal_2'
df_gen_nominal_2.rename(columns={'gen_counts':'counts_gen_nominal_2'}, inplace=True)
df_gen_nominal_2.drop(['gen_tave', 'gen_pave', 'gen_xave', 'gen_qave', 'gen_yave'], axis=1, inplace=True)

df_gen_rad = pd.read_pickle(gen_rad)
#rename column 'counts' to 'counts_gen_rad'
df_gen_rad.rename(columns={'gen_counts':'counts_gen_rad'}, inplace=True)
df_gen_rad.drop(['gen_tave', 'gen_pave', 'gen_xave', 'gen_qave', 'gen_yave'], axis=1, inplace=True)

from functools import reduce

#make a list of all dataframes
dfs = [df_45na,df_55na,df_nominal_1,df_nominal_2,df_rad,df_gen_45na,df_gen_55na,df_gen_nominal_1,df_gen_nominal_2,df_gen_rad]

merged_df = reduce(lambda left,right: pd.merge(left,right,on=['tmin', 'pmin', 'xmin', 'qmin', 'tmax', 'pmax', 'xmax', 'qmax']), dfs)


print(merged_df.columns.values)
print(merged_df)

#calculate rec/gen ratios for all bins
merged_df['acc_45na'] = merged_df['counts_rec_45na']/merged_df['counts_gen_45na']
merged_df['acc_55na'] = merged_df['counts_rec_55na']/merged_df['counts_gen_55na']
merged_df['acc_nominal_1'] = merged_df['counts_rec_nominal_1']/merged_df['counts_gen_nominal_1']
merged_df['acc_nominal_2'] = merged_df['counts_rec_nominal_2']/merged_df['counts_gen_nominal_2']
merged_df['acc_rad'] = merged_df['counts_rec_rad']/merged_df['counts_gen_rad']

#now take ratios
merged_df['45na-nom'] = merged_df['acc_45na']/merged_df['acc_nominal_1']
merged_df['55na-nom'] = merged_df['acc_55na']/merged_df['acc_nominal_1']
merged_df['rad-nom'] = merged_df['acc_rad']/merged_df['acc_nominal_1']
merged_df['45na-55na'] = merged_df['acc_45na']/merged_df['acc_55na']
merged_df['nom-nom'] = merged_df['acc_nominal_1']/merged_df['acc_nominal_2']

#now take ratios of ratios
merged_df['45na-nom-nom-nom'] = merged_df['45na-nom']/merged_df['nom-nom']
merged_df['55na-nom-nom-nom'] = merged_df['55na-nom']/merged_df['nom-nom']
merged_df['rad-nom-nom-nom'] = merged_df['rad-nom']/merged_df['nom-nom']


#make a histogram of the ratios
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,3, figsize=(12,6))
ax[0].hist(merged_df['45na-nom-nom-nom'], bins=100, range=[0,2])
ax[0].set_xlabel("45na-nom-nom-nom")
ax[0].set_ylabel("Counts")
ax[1].hist(merged_df['55na-nom-nom-nom'], bins=100, range=[0,2])
ax[1].set_xlabel("55na-nom-nom-nom")
ax[1].set_ylabel("Counts")
ax[2].hist(merged_df['rad-nom-nom-nom'], bins=100, range=[0,2])
ax[2].set_xlabel("rad-nom-nom-nom")
ax[2].set_ylabel("Counts")
plt.show()

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,4, figsize=(12,6))
ax[0].hist(merged_df['45na-nom'], bins=100, range=[0,2])
ax[0].set_xlabel("45na-nom-nom-nom")
ax[0].set_ylabel("Counts")
ax[1].hist(merged_df['55na-nom'], bins=100, range=[0,2])
ax[1].set_xlabel("55na-nom-nom-nom")
ax[1].set_ylabel("Counts")
ax[2].hist(merged_df['rad-nom'], bins=100, range=[0,2])
ax[2].set_xlabel("rad-nom-nom-nom")
ax[2].set_ylabel("Counts")
ax[3].hist(merged_df['nom-nom'], bins=100,range=[0,2])
ax[3].set_xlabel("nom-nom-nom-nom")
ax[3].set_ylabel("Counts")
plt.show()


merged_df.to_pickle("rad_recon_uncertainty_df.pkl")



#ratios:
#(rec_45na_inb/gen_45na_inb)/(rec_nominal_1/gen_nominal_1)/(rec_nominal_2/gen_nominal_2)/(rec_nominal_1/gen_nominal_1)
#(rec_55na_inb/gen_55na_inb)/(rec_nominal_1/gen_nominal_1)/(rec_nominal_2/gen_nominal_2)/(rec_nominal_1/gen_nominal_1)
#(rec_rad/gen_rad)/(rec_nominal_1/gen_nominal_1)/(rec_nominal_2/gen_nominal_2)/(rec_nominal_1/gen_nominal_1)


