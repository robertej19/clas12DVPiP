import uproot
import pandas as pd
import numpy as np
import argparse
import os, sys
from icecream import ic
import matplotlib.pyplot as plt
from copy import copy
from utils.utils import dot
from utils.utils import mag
from utils.utils import mag2
from utils.utils import cosTheta
from utils.utils import angle
from utils.utils import cross
from utils.utils import vecAdd
from utils.utils import pi0Energy
from utils.utils import pi0InvMass
from utils.utils import getPhi
from utils.utils import getTheta
from utils.utils import getEnergy
from utils.utils import readFile
from utils import make_histos
from utils import histo_plotting
from utils import filestruct
from matplotlib.patches import Rectangle
pd.set_option('mode.chained_assignment', None)

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



a = 8
b = 338


qmin = 4.0
xmin = 0.3
tmin = 0.6

plt.rcParams["font.size"] = "20"

fig, ax = plt.subplots(figsize =(14, 10)) 


df = pd.read_pickle("/mnt/d/GLOBUS/CLAS12/APS2022/pickled_dvpip/raw_data_f2018_inbending_20220113_dvpip_exp.pkl")

df2 = df[["Q2","xB","Epy","Epx"]].copy()#.head(10)

dfnp = df2.to_numpy()


print(dfnp.size)

#Get number of columns
num_cols = dfnp.shape[1]
blank_bin_edges = [-1000,1000]
q2_bin_edges = [1,2,3,40]
xb_bin_edges = [0.1,0.3,0.6,0.9]

initalized = [blank_bin_edges]*num_cols

initalized[0] = q2_bin_edges
initalized[1] = xb_bin_edges

QQQ_bin_values, edges = np.histogramdd(dfnp, bins=initalized)

print(df2)
ic(QQQ_bin_values)
ic(edges)

for q2_index,(q2_bin_min,q2_bin_max) in enumerate(zip(q2_bin_edges[0:-1],q2_bin_edges[1:])):
    print("Q2 bin range of {} to {}".format(q2_bin_min,q2_bin_max))
    for xb_index,(xb_bin_min,xb_bin_max) in enumerate(zip(xb_bin_edges[0:-1],xb_bin_edges[1:])):
        print("XB bin range of {} to {}".format(xb_bin_min,xb_bin_max))
        print(QQQ_bin_values[q2_index][xb_index])
