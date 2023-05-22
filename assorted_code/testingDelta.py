import pandas as pd
from icecream import ic
import numpy as np
from utils.make_histos import plot_1dhist

#df = pd.read_pickle('/mnt/d/GLOBUS/CLAS12/APS2022/pickled_data/20220511_f18_out_combined_171.pkl')


recondata = "1933_rad_Fall_2018_Outbending_100_recon.pkl"

df = pd.read_pickle('/mnt/d/GLOBUS/CLAS12/APS2022/pickled_data/{}'.format(recondata))



print(df)

W = df['W']


#np.histogram(W, bins=100)


plot_1dhist(W,['W'])

"""
t(x_data,vars,ranges="none",second_x=False,second_x_data=[],logger=False,first_label="rad",second_label="norad",x0_key="None",
            saveplot=False,pics_dir="none",plot_title="none",first_color="blue",sci_on=False,plot_title_identifiyer="",fitdata=False):
   """ 