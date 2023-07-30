import pandas as pd
from icecream import ic
import numpy as np
import matplotlib.pyplot as plt

dir_name = "/mnt/d/GLOBUS/CLAS12/Thesis/plots/bin_migration_purity_studies/"
file_name = "bin_purities_inb_norad_t1.pkl"
file_name_2 = "bin_purities_inb_norad_t2.pkl"
file_name_3 = "bin_purities_inb_rad_t1.pkl"
file_name_4 = "bin_purities_inb_rad_t2.pkl"

df = pd.read_pickle(dir_name+file_name)
df_2 = pd.read_pickle(dir_name+file_name_2)
df_3 = pd.read_pickle(dir_name+file_name_3)
df_4 = pd.read_pickle(dir_name+file_name_4)

#drop rows with purity = -1
df = df[df['purity'] != -1]
df_2 = df_2[df_2['purity'] != -1]
df_3 = df_3[df_3['purity'] != -1]
df_4 = df_4[df_4['purity'] != -1]
# drop rows with efficiency = -1
df = df[df['efficiency'] != -1]
df_2 = df_2[df_2['efficiency'] != -1]
df_3 = df_3[df_3['efficiency'] != -1]
df_4 = df_4[df_4['efficiency'] != -1]

#get mean of 'purity' and 'efficiency' columns
print("t1 mean purity: ",df['purity'].mean())
print("t1 mean efficiency: ",df['efficiency'].mean())
print("t2 mean purity: ",df_2['purity'].mean())
print("t2 mean efficiency: ",df_2['efficiency'].mean())
print("t1 rad mean purity: ",df_3['purity'].mean())
print("t1 rad mean efficiency: ",df_3['efficiency'].mean())
print("t2 rad mean purity: ",df_4['purity'].mean())
print("t2 rad mean efficiency: ",df_4['efficiency'].mean())

print(df.columns.values)
#make a histogram of bin purities
# fig, ax = plt.subplots(figsize=(14,10))
# ax.hist(df['purity'],bins=100,range=(0,1),alpha=0.7,label='t1')
# ax.hist(df_2['purity'],bins=100,range=(0,1),alpha=0.7,label='t2')
# ax.hist(df_3['purity'],bins=100,range=(0,1),alpha=0.7,label='t1 rad')
# ax.hist(df_4['purity'],bins=100,range=(0,1),alpha=0.7,label='t2 rad')
# ax.set(xlabel='Bin Purity', ylabel='Number of Bins')
# ax.set_title('Histogram of Bin Purity')
# plt.legend()
# plt.show()#fig.savefig(dir_name+'histogram_of_bin_purity.png')
# #plt.close()

import matplotlib as mpl

#make 1d histogram of bin purity
fig, ax = plt.subplots(figsize=(14,10))
ax.hist(df['purity'],bins=100,alpha=0.7,label='t1')
ax.hist(df_2['purity'],bins=100,alpha=0.7,label='t2')
plt.show()

#increase fontsize to 20
mpl.rcParams.update({'font.size': 20})
#make 2D histogram of bin purity vs bin efficency
purity = df['purity']
efficiency = df['efficiency']

fig, ax = plt.subplots(figsize=(14,10))

plt.hist2d(purity, efficiency, bins=(25, 25), norm=mpl.colors.LogNorm(),range=((0,1),(0,1)))
plt.colorbar()#label='counts in bin')

#add 
plt.xlabel('Bin Purity')
plt.ylabel('Bin Efficiency')
#add title
plt.title('Distribution of Bin Purity vs Bin Efficiency')

plt.show()
#save figure
fig.savefig(dir_name+'t1_bin_purity_vs_bin_efficiency.png')