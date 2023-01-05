import pandas as pd
from icecream import ic

class fs:
	def __init__(self):
		self.phibins_test = [0,18,36,54,72,90,108,126,144,162,180,198,216,234,252,270,288,306,324,342,360]
		self.tbins_test =  [0.02,0.3,0.4,10.6]
		self.xBbins_test = [0.025,.3,0.38,0.958]
		self.q2bins_test = [0.2,2.5,23]
		


		#self.tbins = [0,0.2,0.4,0.6,0.8,1,2,4,7,12]
		#self.xBbins = [0,0.2,.4,.6,.8, 1]
		#self.q2bins = [0,1,1.5,2,2.5,3,3.5,4,4.5,5,6,10,14]


		self.xBbins = [0.175,0.25,.3,0.35,.4,0.45,.5,.55,.6,.65,.7]
		self.q2bins =  [1.25,4.25,5.25,6.25,7.25,10.25]
		#self.tbins =  [0.09,0.15,0.2,0.3,0.4,0.6,1,1.5,2,3,4.5,6,12]
		self.tbins =  [0.12,0.32,0.52,0.72,0.92,1.12]
		#self.phibins =  [0,18,36,54,72,90,108,126,144,162,180,198,216,234,252,270,288,306,324,342,360]
		self.phibins =  [0,36,72,108,144,180,216,252,288,324,360]
	

		# self.xBbins = [0.175,0.275,0.375,0.475,0.675]
		# self.q2bins =  [1.25,2.25,3.25,4.25,6.25,9.25]
		# #self.tbins =  [0.09,0.15,0.2,0.3,0.4,0.6,1,1.5,2,3,4.5,6,12]
		# self.tbins =  [0.12,0.32,0.52,0.72,0.92,1.12]
		# #self.phibins =  [0,18,36,54,72,90,108,126,144,162,180,198,216,234,252,270,288,306,324,342,360]
		# self.phibins =  [0,36,72,108,144,180,216,252,288,324,360]
	

		# self.xBbins = [0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7]
		# self.q2bins =  [1.0,1.5,2.0,2.5,3.0,3.25,4.25,6.25,7.25,11.25]
		# #self.tbins =  [0.09,0.15,0.2,0.3,0.4,0.6,1,1.5,2,3,4.5,6,12]
		# self.tbins =  [0.12,0.32,0.52,0.72,0.92,1.12]
		# #self.phibins =  [0,18,36,54,72,90,108,126,144,162,180,198,216,234,252,270,288,306,324,342,360]
		# self.phibins =  [0,36,72,108,144,180,216,252,288,324,360]

		# For small bin testing
		# self.xBbins = [0.1,0.15,0.2]
		# self.q2bins =  [1.5,2.0,2.5,]
		# self.tbins =  [0.09,0.15,0.2]
		# self.phibins =  [0,18,36,54,72,90,108,126,144,162,180,198,216,234,252,270,288,306,324,342,360]

		# self.xBbins = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7,0.85,1]
		# self.q2bins =  [1.0,1.3,1.6,1.9,2.2,2.5,2.8,3.1,9.0,14.0]
		# self.tbins =  [0.09,0.15,0.2,0.3,0.4,0.6,1,1.5,2,3,4.5,6,12]
		# self.phibins =  [0,18,36,54,72,90,108,126,144,162,180,198,216,234,252,270,288,306,324,342,360]


		self.path_to_exp_outbending_root = "exp_outbend/"

		self.path_to_rec_outbending_rad_root = "rad_outbend_rec/"
		self.path_to_gen_outbending_rad_root = "rad_outbend_gen/"

		self.path_to_rec_outbending_norad_root = "norad_outbend_rec/"
		self.path_to_gen_outbending_norad_root = "norad_outbend_gen/"


		self.path_to_exp_inbending_root = "exp_inbend/"

		self.path_to_rec_inbending_rad_root = "rad_inbend_rec/"
		self.path_to_gen_inbending_rad_root = "rad_inbend_gen/"

		self.path_to_rec_inbending_norad_root = "norad_inbend_rec/"
		self.path_to_gen_inbending_norad_root = "norad_inbend_gen/"



		self.f18_inbending_total_lumi = 5.511802214933477e+40
		self.f18_inbending_total_lumi_inv_nb = 5.511802214933477e+7
		self.Ebeam = 10.6 #GeV, CLAS12 beam energy
		self.m_p = 0.938 #GeV, proton mass
		self.alpha = 1/137 #, fine structure constant
		



		