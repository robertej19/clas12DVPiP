import uproot
import pandas as pd
import numpy as np
import argparse
import itertools
import os, sys
from icecream import ic
import matplotlib
#matplotlib.use('Agg') 


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
from utils import get_integrated_lumi
import matplotlib as mpl

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
import matplotlib as mpl

# 1.) Necessary imports.    
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import argparse
import sys 
import pandas as pd
from matplotlib.patches import Rectangle

from matplotlib import pylab as plt

import numpy as np
from scipy.optimize import minimize

from utils import filestruct
pd.set_option('mode.chained_assignment', None)

import random 
import sys
import os, subprocess
import argparse
import shutil
import time
from datetime import datetime 
import json

from convert_root_to_pickle import convert_GEN_NORAD_root_to_pkl
from convert_root_to_pickle import convert_GEN_RAD_root_to_pkl
from convert_root_to_pickle import new_convert_real_to_pkl
from convert_root_to_pickle import new_convert_rec_to_pkl


from utils import filestruct
pd.set_option('mode.chained_assignment', None)

import random 
import sys
import os, subprocess
import argparse
import shutil
import time
from datetime import datetime 
import json

# For analysis flow
#from make_dvpip_cuts import makeDVpi0
#from exclusivity_cuts.sangbaek_exclusivity_cuts import makeDVpi0P
from exclusivity_cuts.new_exclusivity_cuts import makeDVpi0P
from exclusivity_cuts.new_exclusivity_cuts import calc_ex_cut_mu_sigma


from bin_events import bin_df




from full_flow import run_analysis


run_name = "rad_f18_new_simple_excuts_with_range"

mag_configs = ["inbending",]#,"outbending"]
generator_type = "rad"
proton_locs = ["All",]
photon1_locs = ["All",]
photon2_locs = ["All",]
sigma_multis = [4,2,3]

for mc in mag_configs:
    for sigma_multiplier in sigma_multis:
        for pl in proton_locs:
            for p1l in photon1_locs:
                for p2l in photon2_locs:
                    #print("hi")
                    print("ON SIGMA, MAG CONFIG: {},{}".format(sigma_multiplier,mc))
                    run_analysis(mc,generator_type,unique_identifyer=run_name,#"for_aps_gen_plots_norad_bigplots",
                                det_proton_loc=pl,det_photon1_loc=p1l,det_photon2_loc=p2l,
                                convert_roots = 0,
                                make_exclusive_cuts = 1,
                                plot_initial_distros = 0,
                                plot_final_distros = 0,
                                bin_all_events = 1,
                                bin_gen = 1,
                                calc_xsection = 0,
                                plot_reduced_xsec_and_fit = 0,
                                calc_xsection_c12_only = 1,
                                plot_reduced_xsec_and_fit_c12_only = 1,
                                plot_1_D_hists = 0,
                                simple_exclusivity_cuts=False,
                                emergency_stop = 0,
                                comp_2_config=False,
                                gen_ex_cut_table=True,
                                sigma_multiplier=sigma_multiplier)
