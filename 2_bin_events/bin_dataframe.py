


# 1.) Necessary imports.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import argparse
import sys, os
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib as mpl
from icecream import ic
import scipy.integrate as integrate
from scipy.optimize import root_scalar



def bottom_line(x):
    return (4-0.938**2)/(1/x-1) 
    
    #y = Q^2/(2ME xB) < (10.604-2) / (10.604)
def top_line(x):
    return (10.604-2) / (10.604)*2*0.938*x*10.604

def right_line(y):
    return 1/((4-0.938**2)/y+1)

def left_line(y):
    return y/((10.604-2) / (10.604)*2*0.938*10.604)


#generate sample xb q2 t phi bins with array length 3



df = pd.read_pickle("first10.pkl")
#print(df.columns.values)
#print the xb q2 t phi columns of the dataframe
print(df[['xB','Q2','t1','phi1']])
#make 2d histogram of the dataframe

# Define bin edges
xb_bins = [0.1,0.4,0.8]
q2_bins =  [1,5,11]
t_bins =  [0.09,0.6,2]
phi_bins =  [0,180,360]

# Add bin labels to original dataframe
df['xb_bin'] = pd.cut(df['xB'], bins=xb_bins)
df['q2_bin'] = pd.cut(df['Q2'], bins=q2_bins)
df['t_bin'] = pd.cut(df['t1'], bins=t_bins)
df['phi_bin'] = pd.cut(df['phi1'], bins=phi_bins)

# Calculate weighted bin centers for each bin
df['xb_weighted'] = df['xB'] * df['xB']
df['q2_weighted'] = df['Q2'] * df['Q2']
df['t_weighted'] = df['t1'] * df['t1']
df['phi_weighted'] = df['phi1'] * df['phi1']

df_binned = df.groupby(['xb_bin', 'q2_bin', 't_bin', 'phi_bin']).agg(
    count=pd.NamedAgg(column='xB', aggfunc='size'),  # Count of entries
    xb_center=pd.NamedAgg(column='xb_weighted', aggfunc=np.mean),  # Weighted bin center for xb
    q2_center=pd.NamedAgg(column='q2_weighted', aggfunc=np.mean),  # Weighted bin center for q2
    t_center=pd.NamedAgg(column='t_weighted', aggfunc=np.mean),  # Weighted bin center for t
    phi_center=pd.NamedAgg(column='phi_weighted', aggfunc=np.mean),  # Weighted bin center for phi
).reset_index()