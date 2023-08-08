import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os, sys
from PIL import Image
import numpy as np
import re
# import line2D for custom legend
from matplotlib.lines import Line2D

from utils import filestruct, const, make_histos
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline

df = pd.read_pickle("t_int_of_xsec_with_fit.pkl")
# df has values ['xave' 'qave' 'A' 'B' 'A_error' 'B_error']

df['B_error'] = df['B_error']*1.5

plt.rcParams["font.size"] = "30"
plt.figure(figsize=(20,14))
# Define marker sizes based on 'qave' values
marker_sizes = df['qave']*df['qave']*40  # Adjust the scaling factor (10 in this case) to your liking
print(marker_sizes)
# Scatter plot with colormap based on 'qave' and size based on 'qave'
cmap = plt.get_cmap('plasma')
norm = plt.Normalize(df['qave'].min(), df['qave'].max())

sc = plt.scatter(df['xave'], df['B'], c=df['qave'], s=marker_sizes, cmap=cmap, norm=norm)

# Get desi
colors = cmap(norm(df['qave'].values))

# Create a colorbar with the original 'qave' values
def get_tick_locations(values):
    ticks = [values.min(), values.max()]
    interval = (values.max() - values.min()) / 5
    for i in range(1, 5):
        ticks.append(values.min() + i * interval)
    return sorted(ticks)

# Get desired tick locations for 'qave'
# Get desired tick locations for 'qave'
ticks = get_tick_locations(df['qave'].unique())

# Create a colorbar with the desired ticks
cbar = plt.colorbar(sc, label='qave', ticks=ticks)
cbar.set_label(r'$\langle Q^2 \rangle$', rotation=270, labelpad=15)
cbar.set_ticklabels(['{:.1f}'.format(val) for val in ticks])
# Adding vertical error bars with colors matched to scatter points
for x, y, yerr, color in zip(df['xave'], df['B'], df['B_error'], colors):
    plt.errorbar(x, y, yerr=yerr,markeredgecolor=color, color=color, capsize=10,capthick=5, elinewidth=5)

plt.xlabel(r'$\langle x_B \rangle$')
plt.ylabel('B Parameter')
plt.title("B Parameter Across $Q^2$ and $x_B$")

plt.tight_layout()
#plt.show()#
plt.savefig("B_vs_Q2_and_xB.png")
plt.close()

plt.figure(figsize=(20,14))


# Define marker sizes based on 'qave' values
marker_sizes = df['xave']*df['xave']*4000  # Adjust the scaling factor (10 in this case) to your liking
print(marker_sizes)
# Scatter plot with colormap based on 'qave' and size based on 'qave'
cmap = plt.get_cmap('plasma')
norm = plt.Normalize(df['xave'].min(), df['xave'].max())

sc = plt.scatter(df['qave'], df['B'], c=df['xave'], s=marker_sizes, cmap=cmap, norm=norm)

# Get desi
colors = cmap(norm(df['xave'].values))
# Create a colorbar with the original 'qave' values
def get_tick_locations(values):
    ticks = [values.min(), values.max()]
    interval = (values.max() - values.min()) / 5
    for i in range(1, 5):
        ticks.append(values.min() + i * interval)
    return sorted(ticks)

# Get desired tick locations for 'qave'
ticks = get_tick_locations(df['xave'].unique())

# Create a colorbar with the desired ticks
cbar = plt.colorbar(sc, label='xave', ticks=ticks)
cbar.set_label(r'$\langle x_B \rangle$', rotation=270, labelpad=15)
cbar.set_ticklabels(['{:.1f}'.format(val) for val in ticks])
# Adding vertical error bars with colors matched to scatter points
for x, y, yerr, color in zip(df['qave'], df['B'], df['B_error'], colors):
    plt.errorbar(x, y, yerr=yerr, linestyle='none', color=color, capsize=10,capthick=5, elinewidth=5)

plt.xlabel(r'$\langle Q^2 \rangle$')
plt.ylabel('B Parameter')
plt.title("B Parameter Across $x_B$ and $Q^2$")
#plt.title(r'$\langle x_B \rangle$'+'={:.2f}'.format(group['xave'].mean())+r', $\langle Q^2 \rangle$'+'={:.2f}'.format(group['qave'].mean())+' GeV$^2$',
#                  y=0.94,x=0.4,bbox=dict(facecolor='white', alpha=1, edgecolor='none'))

plt.tight_layout()
#plt.show()
plt.savefig("B_vs_xB_and_Q2.png")
plt.close()