import pandas as pd
from icecream import ic

class fs:
	def __init__(self):
		self.data_path = "/mnt/d/GLOBUS/CLAS12/Thesis/0_raw_root_files/"

		# self.phibins_test = [0,18,36,54,72,90,108,126,144,162,180,198,216,234,252,270,288,306,324,342,360]
		# self.tbins_test =  [0.02,0.3,0.4,10.6]
		# self.xBbins_test = [0.025,.3,0.38,0.958]
		# self.q2bins_test = [0.2,2.5,23]
		

		# FINAL BINS
		#true bins
		self.xBbins = [0.1,0.15,0.20,0.25,0.30,0.38,0.48,0.58,0.80]
		self.Q2bins =  [1,1.5,2,2.5,3]#,3.5]#,4,4.5,5.5,7,11]
		self.tbins =  [0.09,0.15,0.2,0.3,0.4,0.6,1,1.5,2]
		self.phibins =  [0,36,72,108,144,180,216,252,288,324,360]
		# self.xBbins = [0.1,0.15,0.20,0.25,0.3,0.38,0.48]#,0.58]#,0.80]
		# #self.xBbins = [0.30,0.38,0.48,0.58,0.80]
		# self.Q2bins =  [1,1.5,2,2.5,3]#,3.5,4]#,4.5]#,5.5,7,11]
		# self.tbins =  [0.09,0.15,0.2,0.3]#,0.4,0.6,1]#,1.5,2]
		# self.phibins =  [0,36,72,108,144,180,216,252,288,324,360]
		# # THIS WORKS!
		# self.xBbins = [0.1,0.15,0.20,0.25,0.3,0.38,0.48]#,0.58]#,0.80]
		# #self.xBbins = [0.30,0.38,0.48,0.58,0.80]
		# self.Q2bins =  [1,1.5,2,2.5,3,3.5,4]#,4.5]#,5.5,7,11]
		# self.tbins =  [0.09,0.15,0.2,0.3,0.4,0.6,1]#,1.5,2]
		# self.phibins =  [0,36,72,108,144,180,216]#,252,288,324,360]
		#self.phibins =  [0,90,180,270,360]
		#
		# Testing
		# self.xBbins = [0.1,0.15,0.20,0.25#,0.30,0.38,0.48,0.58,0.80]
		# self.Q2bins =  [1,1.5,2,2.5]#,3,3.5,4,4.5,5.5,7,11]
		# self.tbins =  [0.09,0.15,0.2,0.3]#,0.4,0.6,1,1.5,2]
		# self.phibins =  [36,72,108,144]#,180,216,252,288,324,360]
		# # self.xBbins = [0.1,0.20,0.30,0.48,0.80]
		# self.Q2bins =  [1,1.5,3,7,11]
		# self.tbins =  [0.09,0.15,.4,1.5,2]
		# self.phibins = [0,144,180,288,324,360]
		#for testing:
		# self.xBbins = [0.1,0.3,0.80]
		# self.Q2bins =  [1,3,11]
		# self.tbins =  [0.0,0.7,2]
		# self.phibins =  [0,150,360]
	
		self.inb_exp_epgg_dir = "/mnt/d/GLOBUS/CLAS12/Thesis/1_potential_dvpip_events/inb/exp/"
		self.inb_exp_dvpip_dir = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/inb/exp/"
		self.inb_exp_binned_dir = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/exp/"

		self.inb_norad_rec_root_dir = "/mnt/d/GLOBUS/CLAS12/Thesis/0_raw_root_files/rec_inbend_norad/"
		self.inb_norad_rec_epgg_dir = "/mnt/d/GLOBUS/CLAS12/Thesis/1_potential_dvpip_events/inb/rec/"

		self.inb_rad_rec_root_dir = "/mnt/d/GLOBUS/CLAS12/Thesis/0_raw_root_files/rec_inbend_rad/"
		self.inb_rad_rec_epgg_dir = "/mnt/d/GLOBUS/CLAS12/Thesis/1_potential_dvpip_events/inb/rec_rad/"

		self.inb_norad_rec_epgg_dir_reproc = "/mnt/d/GLOBUS/CLAS12/Thesis/1_potential_dvpip_events/inb/rec_reproc/"

		self.inb_norad_rec_dvpip_dir = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/inb/rec/"
		self.inb_rad_rec_dvpip_dir = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/inb/rec_rad/"
		self.inb_norad_rec_dvpip_dir_reproc = "/mnt/d/GLOBUS/CLAS12/Thesis/2_selected_dvpip_events/inb/rec_reproc/"

		self.inb_norad_rec_binned_dir = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/rec/"
		self.inb_rad_rec_binned_dir = "/mnt/d/GLOBUS/CLAS12/Thesis/3_binned_dvpip/inb/rec_rad/"


		self.norad_gen_events_dir = "/mnt/d/GLOBUS/CLAS12/Thesis/1_potential_dvpip_events/gen/"
		#self.tbins = [0,0.2,0.4,0.6,0.8,1,2,4,7,12]
		#self.xBbins = [0,0.2,.4,.6,.8, 1]
		#self.q2bins = [0,1,1.5,2,2.5,3,3.5,4,4.5,5,6,10,14]

		#self.xBbins = [0.175,0.5,.7]
		#self.q2bins =  [1,1.25,10.25]
		#self.tbins =  [0.09,0.15,0.2,0.3,0.4,0.6,1,1.5,2,3,4.5,6,12]
		#self.tbins =  [0.12,1.12]
		#self.phibins =  [0,18,36,54,72,90,108,126,144,162,180,198,216,234,252,270,288,306,324,342,360]
		#self.phibins =  [0,360]
	



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


		# self.path_to_exp_outbending_root = "exp_outbend/"

		# self.path_to_rec_outbending_rad_root = "rad_outbend_rec/"
		# self.path_to_gen_outbending_rad_root = "rad_outbend_gen/"

		# self.path_to_rec_outbending_norad_root = "norad_outbend_rec/"
		# self.path_to_gen_outbending_norad_root = "norad_outbend_gen/"


		# self.path_to_exp_inbending_root = "exp_inbend/"

		# self.path_to_rec_inbending_rad_root = "rad_inbend_rec/"
		# self.path_to_gen_inbending_rad_root = "rad_inbend_gen/"

		# self.path_to_rec_inbending_norad_root = "norad_inbend_rec/"
		# self.path_to_gen_inbending_norad_root = "norad_inbend_gen/"



		self.f18_inbending_total_lumi = 5.511802214933477e+40
		self.f18_inbending_total_lumi_inv_fb = 5.511802214933477e+7

		self.f18_inbending_total_lumi_inv_nb = 5.511802214933477e+7
		#self.Ebeam = 10.604 #GeV, CLAS12 beam energy MOVE TO CONST.PY
		#self.alpha = 1/137 #, fine structure constant MOVE TO CONST.PY
		



		