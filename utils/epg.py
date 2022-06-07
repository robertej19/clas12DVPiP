#!/usr/bin/env python3
"""
Modules set up pandas DataFrame for epg business, DVCS and DVpi0P.
"""

import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils.const import *
from utils.physics import *

class epg:
	"""
	Parent class setting up dvcs and dvpi0p
	"""
	def __init(self):
		pass

	def setDVCSvars(self, correction=None):
		#set up dvcs variables
		df_epg = self.df_epg

		ele = [df_epg['Epx'], df_epg['Epy'], df_epg['Epz']]
		df_epg.loc[:, 'Ep'] = mag(ele)
		df_epg.loc[:, 'Ee'] = getEnergy(ele, me)
		df_epg.loc[:, 'Etheta'] = getTheta(ele)
		df_epg.loc[:, 'Ephi'] = getPhi(ele)

		pro = [df_epg['Ppx'], df_epg['Ppy'], df_epg['Ppz']]
		df_epg.loc[:, 'Pp'] = mag(pro)
		df_epg.loc[:, 'Pe'] = getEnergy(pro, M)
		df_epg.loc[:, 'Ptheta'] = getTheta(pro)
		df_epg.loc[:, 'Pphi'] = getPhi(pro)

		gam = [df_epg['Gpx'], df_epg['Gpy'], df_epg['Gpz']]
		df_epg.loc[:, 'Gp'] = mag(gam)
		df_epg.loc[:, 'Ge'] = getEnergy(gam, 0)
		df_epg.loc[:, 'Gtheta'] = getTheta(gam)
		df_epg.loc[:, 'Gphi'] = getPhi(gam)

		Ppt = mag([df_epg['Ppx'], df_epg['Ppy'], 0])

		if correction:
			# if self.rec:
				# newProz = np.where(df_epg["Ptheta"]<=27, df_epg["Ppz"] + 0.03905658* np.exp(-2.14597285*df_epg["Ppz"]), 
				#                   df_epg["Ppz"] + 0.12507378 * np.exp(-2.62346876*df_epg["Ppz"]))
				# ratio = newProz/df_epg["Ppz"]
				# newProx = np.where(df_epg["Ptheta"]<=27, df_epg["Ppx"], df_epg["Ppx"] + 0.008 * np.sin(df_epg["Pphi"]/180*np.pi + 2.1932))
				# newProy = np.where(df_epg["Ptheta"]<=27, df_epg["Ppy"], df_epg["Ppy"] + 0.008 * np.sin(df_epg["Pphi"]/180*np.pi + 0.6224))		
				
				# # ratio2 = np.where(df_epg["Ptheta"]<=27, 1, (Ppt - 0.001 + 0.13 * np.exp(-5.01982791 * df_epg["Ppz"])) / Ppt)
				# ratio2 = np.where(df_epg["Ptheta"]<=27, 1, newProz/df_epg["Ppz"])
				# newProx = df_epg["Ppx"].multiply(ratio2, axis="index")
				# newProy = df_epg["Ppy"].multiply(ratio2, axis="index")
				# df_epg.loc[:, "Ppx"] = newProx
				# df_epg.loc[:, "Ppy"] = newProy
				# df_epg.loc[:, "Ppz"] = newProz

			#energy loss
			const = np.select([df_epg.Ptheta<27, (df_epg.Ptheta>=27) & (df_epg.Ptheta<42), df_epg.Ptheta>=42],
			                  [-0.0123049 + 0.00028887*df_epg.Ptheta, -0.138227479 + 8.07557430*0.001*df_epg.Ptheta -1.34807927*0.0001*df_epg.Ptheta*df_epg.Ptheta, -0.0275235])
			coeff = np.select([df_epg.Ptheta<27, (df_epg.Ptheta>=27) & (df_epg.Ptheta<42), df_epg.Ptheta>=42],
			                  [0.01528006 - 0.00024079*df_epg.Ptheta, 5.65817597*0.01 -2.36903348*0.001*df_epg.Ptheta + 4.93780046*0.00001*df_epg.Ptheta*df_epg.Ptheta, 0.03998975])    

			CorrectedPp = const + coeff/df_epg.loc[:, "Pp"] + df_epg.loc[:, "Pp"]

			const = np.select([df_epg.Ptheta<19.5, (df_epg.Ptheta>=19.5) & (df_epg.Ptheta<27), (df_epg.Ptheta>=27) & (df_epg.Ptheta<39), (df_epg.Ptheta>=39) & (df_epg.Ptheta<42), df_epg.Ptheta>=42],
			                  [2.63643690*0.01, 0.50047232 -0.03834672 *df_epg.Ptheta + 0.00071967*df_epg.Ptheta*df_epg.Ptheta, 6.91308654 - 0.439839300*df_epg.Ptheta +6.83075548*0.001*df_epg.Ptheta*df_epg.Ptheta, 1.59424606, 1.47198581*10])
			coeff = np.select([df_epg.Ptheta<19.5, (df_epg.Ptheta>=19.5) & (df_epg.Ptheta<27), (df_epg.Ptheta>=27) & (df_epg.Ptheta<39), (df_epg.Ptheta>=39) & (df_epg.Ptheta<42), df_epg.Ptheta>=42],
			                  [-1.46440415, 74.99891704  -6.1576777*df_epg.Ptheta + 0.11469137*df_epg.Ptheta*df_epg.Ptheta, 682.909471 - 43.9551177 * df_epg.Ptheta + 0.682383790 * df_epg.Ptheta * df_epg.Ptheta, -8.19627119, -23.55701865])    
			coeff2 = np.select([df_epg.Ptheta<19.5, (df_epg.Ptheta>=19.5) & (df_epg.Ptheta<27), (df_epg.Ptheta>=27) & (df_epg.Ptheta<39), (df_epg.Ptheta>=39) & (df_epg.Ptheta<42), df_epg.Ptheta>=42],
			                  [-3.47690993, 47.71351973 -4.34918241*df_epg.Ptheta + 0.08841191*df_epg.Ptheta*df_epg.Ptheta, 100.33995753 - 6.96600416*df_epg.Ptheta + 0.11223046*df_epg.Ptheta*df_epg.Ptheta, -1.25261927, -0.40113733])    

			CorrectedPtheta = const + coeff*np.exp(coeff2*df_epg.loc[:, "Pp"]) + df_epg.loc[:, "Ptheta"]

			const = np.select([df_epg.Ptheta<16.5, (df_epg.Ptheta>=16.5) & (df_epg.Ptheta<27), (df_epg.Ptheta>=27) & (df_epg.Ptheta<42), df_epg.Ptheta>=42],
			                  [-0.190662844, -0.20725736 -0.00675627 *df_epg.Ptheta + 0.0007863*df_epg.Ptheta*df_epg.Ptheta, 12.1881698 - 0.78906294*df_epg.Ptheta +0.01297898*df_epg.Ptheta*df_epg.Ptheta, -4.59743066*10])
			coeff = np.select([df_epg.Ptheta<16.5, (df_epg.Ptheta>=16.5) & (df_epg.Ptheta<27), (df_epg.Ptheta>=27) & (df_epg.Ptheta<42), df_epg.Ptheta>=42],
			                  [6.48745941, 142.96379788  -16.66339055*df_epg.Ptheta + 0.51311212*df_epg.Ptheta*df_epg.Ptheta, 2.1853046 + 5.78521226 * df_epg.Ptheta - 0.09727796 * df_epg.Ptheta * df_epg.Ptheta, 7.46969457*10])    
			coeff2 = np.select([df_epg.Ptheta<16.5, (df_epg.Ptheta>=16.5) & (df_epg.Ptheta<27), (df_epg.Ptheta>=27) & (df_epg.Ptheta<42), df_epg.Ptheta>=42],
			                  [-3.14646608, 17.39529095 -1.78403359*df_epg.Ptheta + 0.0335692*df_epg.Ptheta*df_epg.Ptheta, -1.03655317*10 + 0.161333213*df_epg.Ptheta -1.29625675*0.001*df_epg.Ptheta*df_epg.Ptheta, -4.41246899*0.1])    

			CorrectedPphi = const + coeff*np.exp(coeff2*df_epg.loc[:, "Pp"]) + df_epg.loc[:, "Pphi"]

			df_epg.loc[:, "Pp"] = CorrectedPp
			df_epg.loc[:, "Ptheta"] = CorrectedPtheta
			df_epg.loc[:, "Pphi"] = CorrectedPphi

			df_epg.loc[:, "Ppx"] = df_epg.loc[:, "Pp"]*np.sin(np.radians(df_epg.loc[:, "Ptheta"]))*np.cos(np.radians(df_epg.loc[:, "Pphi"]))
			df_epg.loc[:, "Ppy"] = df_epg.loc[:, "Pp"]*np.sin(np.radians(df_epg.loc[:, "Ptheta"]))*np.sin(np.radians(df_epg.loc[:, "Pphi"]))
			df_epg.loc[:, "Ppz"] = df_epg.loc[:, "Pp"]*np.cos(np.radians(df_epg.loc[:, "Ptheta"]))

			pro = [df_epg['Ppx'], df_epg['Ppy'], df_epg['Ppz']]
			df_epg.loc[:, 'Pe'] = getEnergy(pro, M)
			
			# df_epg.loc[:, 'Pp'] = mag(pro)
			# df_epg.loc[:, 'Ptheta'] = getTheta(pro)
			# df_epg.loc[:, 'Pphi'] = getPhi(pro)

			if not self.rec:
				df_epg["Gpz"] = np.select([df_epg.Gpz>=2, (df_epg.Gpz<2) & (df_epg.Gpz>1), df_epg.Gpz<=1],[df_epg.Gpz+0.13, df_epg.Gpz+0.13*(df_epg.Gpz-1), df_epg.Gpz])
				df_epg["Gpx"] = np.select([df_epg.Gpz>=2, (df_epg.Gpz<2) & (df_epg.Gpz>1), df_epg.Gpz<=1],[df_epg.Gpx+0.13*df_epg.Gpx/df_epg.Gpz, df_epg.Gpx+0.13*(df_epg.Gpz-1)*df_epg.Gpx/df_epg.Gpz, df_epg.Gpx])
				df_epg["Gpy"] = np.select([df_epg.Gpz>=2, (df_epg.Gpz<2) & (df_epg.Gpz>1), df_epg.Gpz<=1],[df_epg.Gpy+0.13*df_epg.Gpy/df_epg.Gpz, df_epg.Gpy+0.13*(df_epg.Gpz-1)*df_epg.Gpy/df_epg.Gpz, df_epg.Gpy])

				gam = [df_epg['Gpx'], df_epg['Gpy'], df_epg['Gpz']]
				df_epg.loc[:, 'Gp'] = mag(gam)
				df_epg.loc[:, 'Ge'] = getEnergy(gam, 0)
				df_epg.loc[:, 'Gtheta'] = getTheta(gam)
				df_epg.loc[:, 'Gphi'] = getPhi(gam)

		VGS = [-df_epg['Epx'], -df_epg['Epy'], pbeam - df_epg['Epz']]
		v3l = cross(beam, ele)
		v3h = cross(pro, VGS)
		v3g = cross(VGS, gam)
		VmissG = [-df_epg["Epx"] - df_epg["Ppx"], -df_epg["Epy"] - df_epg["Ppy"],
		          pbeam - df_epg["Epz"] - df_epg["Ppz"]]
		VmissP = [-(df_epg["Epx"] + df_epg["Gpx"]), -(df_epg["Epy"] + df_epg["Gpy"]),
		          -(-pbeam + df_epg["Epz"] + df_epg["Gpz"])]
		Vmiss = [-(df_epg["Epx"] + df_epg["Ppx"] + df_epg["Gpx"]), -(df_epg["Epy"] + df_epg["Ppy"] + df_epg["Gpy"]),
		         -(-pbeam + df_epg["Epz"] + df_epg["Ppz"] + df_epg["Gpz"])]
		costheta = cosTheta(VGS, gam)

		df_epg.loc[:, 'Mpx'], df_epg.loc[:, 'Mpy'], df_epg.loc[:, 'Mpz'] = Vmiss

		# binning kinematics
		df_epg.loc[:,'Q2'] = -((ebeam - df_epg['Ee'])**2 - mag2(VGS))
		df_epg.loc[:,'nu'] = (ebeam - df_epg['Ee'])
		df_epg.loc[:,'y'] = df_epg['nu']/ebeam
		df_epg.loc[:,'xB'] = df_epg['Q2'] / 2.0 / M / df_epg['nu']
		df_epg.loc[:,'t1'] = 2 * M * (df_epg['Pe'] - M)
		df_epg.loc[:,'t2'] = (M * df_epg['Q2'] + 2 * M * df_epg['nu'] * (df_epg['nu'] - np.sqrt(df_epg['nu'] * df_epg['nu'] + df_epg['Q2']) * costheta))\
		/ (M + df_epg['nu'] - np.sqrt(df_epg['nu'] * df_epg['nu'] + df_epg['Q2']) * costheta)
		df_epg.loc[:,'W'] = np.sqrt(np.maximum(0, (ebeam + M - df_epg['Ee'])**2 - mag2(VGS)))

		# trento angles
		df_epg.loc[:,'phi1'] = angle(v3l, v3h)
		df_epg.loc[:,'phi1'] = np.where(dot(v3l, pro) > 0, 360.0 -
		                          df_epg['phi1'], df_epg['phi1'])
		df_epg.loc[:,'phi2'] = angle(v3l, v3g)
		df_epg.loc[:,'phi2'] = np.where(dot(VGS, cross(v3l, v3g)) <
		                          0, 360.0 - df_epg['phi2'], df_epg['phi2'])

		# exclusivity variables
		df_epg.loc[:,'MM2_epg'] = (-M - ebeam + df_epg["Ee"] +
		                     df_epg["Pe"] + df_epg["Ge"])**2 - mag2(Vmiss)
		df_epg.loc[:,'ME_epg'] = (M + ebeam - df_epg["Ee"] - df_epg["Pe"] - df_epg["Ge"])
		df_epg.loc[:,'MM2_ep'] = (-M - ebeam + df_epg["Ee"] + df_epg["Pe"])**2 - mag2(VmissG)
		df_epg.loc[:,'MM2_eg'] = (-M - ebeam + df_epg["Ee"] + df_epg["Ge"])**2 - mag2(VmissP)
		df_epg.loc[:,'MPt'] = np.sqrt((df_epg["Epx"] + df_epg["Ppx"] + df_epg["Gpx"])**2 +
		                        (df_epg["Epy"] + df_epg["Ppy"] + df_epg["Gpy"])**2)
		df_epg.loc[:,'coneAngle'] = angle(ele, gam)
		df_epg.loc[:,'reconGam'] = angle(gam, VmissG)
		df_epg.loc[:,'coplanarity'] = angle(v3h, v3g)
		self.df_epg = df_epg

	def makeDVCS(self):
		#make dvcs pairs
		df_dvcs = self.df_epg
		df_dvcs = df_dvcs[df_dvcs["MM2_eg"] > 0]  # mmeg

		cut_xBupper = df_dvcs["xB"] < 1  # xB
		cut_xBlower = df_dvcs["xB"] > 0  # xB
		cut_Q2 = df_dvcs["Q2"] > 1  # Q2
		cut_W = df_dvcs["W"] > 2  # W
		cut_Ee = df_dvcs["Ee"] > 2  # Ee
		cut_Ge = df_dvcs["Ge"] > 3  # Ge
		cut_Pp = mag([df_dvcs["Ppx"], df_dvcs["Ppy"], df_dvcs["Ppz"]]) > 0.12  # Pp
		cut_Vz = np.abs(df_dvcs["Evz"] - df_dvcs["Pvz"]) < 2.5 + 2.5 / mag([df_dvcs["Ppx"], df_dvcs["Ppy"], df_dvcs["Ppz"]])

		#	Exclusivity cuts
		cut_mmepg = np.abs(df_dvcs["MM2_epg"]) < 0.1  # mmepg
		cut_mmep = np.abs(df_dvcs["MM2_ep"]) < 0.6  # mmep
		cut_mmegupper = df_dvcs["MM2_eg"] < 3  # mmeg
		cut_mmeglower = df_dvcs["MM2_eg"] > 0  # mmeg
		cut_meepgupper = df_dvcs["ME_epg"] < 1.5  # meepg
		cut_meepglower = df_dvcs["ME_epg"] > -0.5  # meepg
		cut_mpt = df_dvcs["MPt"] < 0.25  # mpt
		cut_cone = df_dvcs["coneAngle"] > 5  # coneangle
		cut_recon = df_dvcs["reconGam"] < 2.5  # recon gam angle
		cut_coplanarity = df_dvcs["coplanarity"] < 25  # coplanarity angle
		if "Esector" in df_dvcs:
			cut_sector = df_dvcs["Esector"]!=df_dvcs["Gsector"]
		else:
			cut_sector = 1

		df_dvcs = df_dvcs[cut_xBupper & cut_xBlower & cut_Q2 & cut_W & cut_Ee & cut_Ge & cut_Pp & cut_Vz & cut_mmepg & cut_mmep &
		                 cut_mmegupper & cut_mmeglower & cut_meepgupper & cut_meepglower & cut_mpt & cut_cone & cut_recon & cut_sector]
		self.df_dvcs = df_dvcs               
		# #cut by detector status
		# cut_pFD = df_dvcs["Pstat"] < 4000  # FD
		# cut_pCD = df_dvcs["Pstat"] > 4000  # CD
		# cut_gFD = df_dvcs["Gstat"] > 2000  # FD
		# cut_gFT = df_dvcs["Gstat"] < 2000  # FT

	def setDVpi0vars(self, correction=None):
		#set up pi0 variables
		df_epgg = self.df_epgg

		# useful objects
		ele = [df_epgg['Epx'], df_epgg['Epy'], df_epgg['Epz']]
		df_epgg.loc[:, 'Ep'] = mag(ele)
		df_epgg.loc[:, 'Ee'] = getEnergy(ele, me)
		df_epgg.loc[:, 'Etheta'] = getTheta(ele)
		df_epgg.loc[:, 'Ephi'] = getPhi(ele)

		pro = [df_epgg['Ppx'], df_epgg['Ppy'], df_epgg['Ppz']]
		df_epgg.loc[:, 'Pp'] = mag(pro)
		df_epgg.loc[:, 'Pe'] = getEnergy(pro, M)
		df_epgg.loc[:, 'Ptheta'] = getTheta(pro)
		df_epgg.loc[:, 'Pphi'] = getPhi(pro)

		gam = [df_epgg['Gpx'], df_epgg['Gpy'], df_epgg['Gpz']]
		df_epgg.loc[:, 'Gp'] = mag(gam)
		df_epgg.loc[:, 'Ge'] = getEnergy(gam, 0)
		df_epgg.loc[:, 'Gtheta'] = getTheta(gam)
		df_epgg.loc[:, 'Gphi'] = getPhi(gam)

		gam2 = [df_epgg['Gpx2'], df_epgg['Gpy2'], df_epgg['Gpz2']]
		df_epgg.loc[:, 'Gp2'] = mag(gam2)
		df_epgg.loc[:,'Ge2'] = getEnergy(gam2, 0)
		df_epgg.loc[:, 'Gtheta2'] = getTheta(gam2)
		df_epgg.loc[:, 'Gphi2'] = getPhi(gam2)

		Ppt = mag([df_epgg['Ppx'], df_epgg['Ppy'], 0])

		if correction:

			const = np.select([df_epgg.Ptheta<27, (df_epgg.Ptheta>=27) & (df_epgg.Ptheta<42), df_epgg.Ptheta>=42],
			                  [-0.0123049 + 0.00028887*df_epgg.Ptheta, -0.138227479 + 8.07557430*0.001*df_epgg.Ptheta -1.34807927*0.0001*df_epgg.Ptheta*df_epgg.Ptheta, -0.0275235])
			coeff = np.select([df_epgg.Ptheta<27, (df_epgg.Ptheta>=27) & (df_epgg.Ptheta<42), df_epgg.Ptheta>=42],
			                  [0.01528006 - 0.00024079*df_epgg.Ptheta, 5.65817597*0.01 -2.36903348*0.001*df_epgg.Ptheta + 4.93780046*0.00001*df_epgg.Ptheta*df_epgg.Ptheta, 0.03998975])    

			CorrectedPp = const + coeff/df_epgg.loc[:, "Pp"] + df_epgg.loc[:, "Pp"]

			const = np.select([df_epgg.Ptheta<19.5, (df_epgg.Ptheta>=19.5) & (df_epgg.Ptheta<27), (df_epgg.Ptheta>=27) & (df_epgg.Ptheta<39), (df_epgg.Ptheta>=39) & (df_epgg.Ptheta<42), df_epgg.Ptheta>=42],
			                  [2.63643690*0.01, 0.50047232 -0.03834672 *df_epgg.Ptheta + 0.00071967*df_epgg.Ptheta*df_epgg.Ptheta, 6.91308654 - 0.439839300*df_epgg.Ptheta +6.83075548*0.001*df_epgg.Ptheta*df_epgg.Ptheta, 1.59424606, 1.47198581*10])
			coeff = np.select([df_epgg.Ptheta<19.5, (df_epgg.Ptheta>=19.5) & (df_epgg.Ptheta<27), (df_epgg.Ptheta>=27) & (df_epgg.Ptheta<39), (df_epgg.Ptheta>=39) & (df_epgg.Ptheta<42), df_epgg.Ptheta>=42],
			                  [-1.46440415, 74.99891704  -6.1576777*df_epgg.Ptheta + 0.11469137*df_epgg.Ptheta*df_epgg.Ptheta, 682.909471 - 43.9551177 * df_epgg.Ptheta + 0.682383790 * df_epgg.Ptheta * df_epgg.Ptheta, -8.19627119, -23.55701865])    
			coeff2 = np.select([df_epgg.Ptheta<19.5, (df_epgg.Ptheta>=19.5) & (df_epgg.Ptheta<27), (df_epgg.Ptheta>=27) & (df_epgg.Ptheta<39), (df_epgg.Ptheta>=39) & (df_epgg.Ptheta<42), df_epgg.Ptheta>=42],
			                  [-3.47690993, 47.71351973 -4.34918241*df_epgg.Ptheta + 0.08841191*df_epgg.Ptheta*df_epgg.Ptheta, 100.33995753 - 6.96600416*df_epgg.Ptheta + 0.11223046*df_epgg.Ptheta*df_epgg.Ptheta, -1.25261927, -0.40113733])    

			CorrectedPtheta = const + coeff*np.exp(coeff2*df_epgg.loc[:, "Pp"]) + df_epgg.loc[:, "Ptheta"]

			const = np.select([df_epgg.Ptheta<16.5, (df_epgg.Ptheta>=16.5) & (df_epgg.Ptheta<27), (df_epgg.Ptheta>=27) & (df_epgg.Ptheta<42), df_epgg.Ptheta>=42],
			                  [-0.190662844, -0.20725736 -0.00675627 *df_epgg.Ptheta + 0.0007863*df_epgg.Ptheta*df_epgg.Ptheta, 12.1881698 - 0.78906294*df_epgg.Ptheta +0.01297898*df_epgg.Ptheta*df_epgg.Ptheta, -4.59743066*10])
			coeff = np.select([df_epgg.Ptheta<16.5, (df_epgg.Ptheta>=16.5) & (df_epgg.Ptheta<27), (df_epgg.Ptheta>=27) & (df_epgg.Ptheta<42), df_epgg.Ptheta>=42],
			                  [6.48745941, 142.96379788  -16.66339055*df_epgg.Ptheta + 0.51311212*df_epgg.Ptheta*df_epgg.Ptheta, 2.1853046 + 5.78521226 * df_epgg.Ptheta - 0.09727796 * df_epgg.Ptheta * df_epgg.Ptheta, 7.46969457*10])    
			coeff2 = np.select([df_epgg.Ptheta<16.5, (df_epgg.Ptheta>=16.5) & (df_epgg.Ptheta<27), (df_epgg.Ptheta>=27) & (df_epgg.Ptheta<42), df_epgg.Ptheta>=42],
			                  [-3.14646608, 17.39529095 -1.78403359*df_epgg.Ptheta + 0.0335692*df_epgg.Ptheta*df_epgg.Ptheta, -1.03655317*10 + 0.161333213*df_epgg.Ptheta -1.29625675*0.001*df_epgg.Ptheta*df_epgg.Ptheta, -4.41246899*0.1])    

			CorrectedPphi = const + coeff*np.exp(coeff2*df_epgg.loc[:, "Pp"]) + df_epgg.loc[:, "Pphi"]

			df_epgg.loc[:, "Pp"] = CorrectedPp
			df_epgg.loc[:, "Ptheta"] = CorrectedPtheta
			df_epgg.loc[:, "Pphi"] = CorrectedPphi

			df_epgg.loc[:, "Ppx"] = df_epgg.loc[:, "Pp"]*np.sin(np.radians(df_epgg.loc[:, "Ptheta"]))*np.cos(np.radians(df_epgg.loc[:, "Pphi"]))
			df_epgg.loc[:, "Ppy"] = df_epgg.loc[:, "Pp"]*np.sin(np.radians(df_epgg.loc[:, "Ptheta"]))*np.sin(np.radians(df_epgg.loc[:, "Pphi"]))
			df_epgg.loc[:, "Ppz"] = df_epgg.loc[:, "Pp"]*np.cos(np.radians(df_epgg.loc[:, "Ptheta"]))

			# if self.rec:
			# 	newProz = np.where(df_epgg["Ptheta"]<=27, df_epgg["Ppz"] + 0.03905658* np.exp(-2.14597285*df_epgg["Ppz"]), 
			# 	                  df_epgg["Ppz"] + 0.12507378 * np.exp(-2.62346876*df_epgg["Ppz"]))
			# 	ratio = newProz/df_epgg["Ppz"]
			# 	newProx = np.where(df_epgg["Ptheta"]<=27, df_epgg["Ppx"], df_epgg["Ppx"] + 0.008 * np.sin(df_epgg["Pphi"]/180*np.pi + 2.1932))
			# 	newProy = np.where(df_epgg["Ptheta"]<=27, df_epgg["Ppy"], df_epgg["Ppy"] + 0.008 * np.sin(df_epgg["Pphi"]/180*np.pi + 0.6224))		
			# 	# ratio2 = np.where(df_epgg["Ptheta"]<=27, 1, (Ppt - 0.001 + 0.13 * np.exp(-5.01982791 * df_epgg["Ppz"])) / Ppt)
			# 	ratio2 = np.where(df_epgg["Ptheta"]<=27, 1, newProz/df_epgg["Ppz"])
			# 	newProx = df_epgg["Ppx"].multiply(ratio2, axis="index")
			# 	newProy = df_epgg["Ppy"].multiply(ratio2, axis="index")
			# 	df_epgg.loc[:, "Ppx"] = newProx
			# 	df_epgg.loc[:, "Ppy"] = newProy
			# 	df_epgg.loc[:, "Ppz"] = newProz

			pro = [df_epgg['Ppx'], df_epgg['Ppy'], df_epgg['Ppz']]
			# df_epgg.loc[:, 'Pp'] = mag(pro)
			df_epgg.loc[:, 'Pe'] = getEnergy(pro, M)
			# df_epgg.loc[:, 'Ptheta'] = getTheta(pro)
			# df_epgg.loc[:, 'Pphi'] = getPhi(pro)

			if not self.rec:
				df_epgg["Gpz"] = np.select([df_epgg.Gpz>=2, (df_epgg.Gpz<2) & (df_epgg.Gpz>1), df_epgg.Gpz<=1],[df_epgg.Gpz+0.13, df_epgg.Gpz+0.13*(df_epgg.Gpz-1), df_epgg.Gpz])
				df_epgg["Gpx"] = np.select([df_epgg.Gpz>=2, (df_epgg.Gpz<2) & (df_epgg.Gpz>1), df_epgg.Gpz<=1],[df_epgg.Gpx+0.13*df_epgg.Gpx/df_epgg.Gpz, df_epgg.Gpx+0.13*(df_epgg.Gpz-1)*df_epgg.Gpx/df_epgg.Gpz, df_epgg.Gpx])
				df_epgg["Gpy"] = np.select([df_epgg.Gpz>=2, (df_epgg.Gpz<2) & (df_epgg.Gpz>1), df_epgg.Gpz<=1],[df_epgg.Gpy+0.13*df_epgg.Gpy/df_epgg.Gpz, df_epgg.Gpy+0.13*(df_epgg.Gpz-1)*df_epgg.Gpy/df_epgg.Gpz, df_epgg.Gpy])
				df_epgg["Gpz2"] = np.select([df_epgg.Gpz2>=2, (df_epgg.Gpz2<2) & (df_epgg.Gpz2>1), df_epgg.Gpz2<=1],[df_epgg.Gpz2+0.13, df_epgg.Gpz2+0.13*(df_epgg.Gpz2-1), df_epgg.Gpz2])
				df_epgg["Gpx2"] = np.select([df_epgg.Gpz2>=2, (df_epgg.Gpz2<2) & (df_epgg.Gpz2>1), df_epgg.Gpz2<=1],[df_epgg.Gpx2+0.13*df_epgg.Gpx2/df_epgg.Gpz2, df_epgg.Gpx2+0.13*(df_epgg.Gpz2-1)*df_epgg.Gpx2/df_epgg.Gpz2, df_epgg.Gpx2])
				df_epgg["Gpy2"] = np.select([df_epgg.Gpz2>=2, (df_epgg.Gpz2<2) & (df_epgg.Gpz2>1), df_epgg.Gpz2<=1],[df_epgg.Gpy2+0.13*df_epgg.Gpy2/df_epgg.Gpz2, df_epgg.Gpy2+0.13*(df_epgg.Gpz2-1)*df_epgg.Gpy2/df_epgg.Gpz2, df_epgg.Gpy2])

				gam = [df_epgg['Gpx'], df_epgg['Gpy'], df_epgg['Gpz']]
				df_epgg.loc[:, 'Gp'] = mag(gam)
				df_epgg.loc[:, 'Ge'] = getEnergy(gam, 0)
				df_epgg.loc[:, 'Gtheta'] = getTheta(gam)
				df_epgg.loc[:, 'Gphi'] = getPhi(gam)

				gam2 = [df_epgg['Gpx2'], df_epgg['Gpy2'], df_epgg['Gpz2']]
				df_epgg.loc[:, 'Gp2'] = mag(gam2)
				df_epgg.loc[:,'Ge2'] = getEnergy(gam2, 0)
				df_epgg.loc[:, 'Gtheta2'] = getTheta(gam2)
				df_epgg.loc[:, 'Gphi2'] = getPhi(gam2)

		pi0 = vecAdd(gam, gam2)
		VGS = [-df_epgg['Epx'], -df_epgg['Epy'], pbeam - df_epgg['Epz']]
		v3l = cross(beam, ele)
		v3h = cross(pro, VGS)
		v3g = cross(VGS, gam)
		VmissPi0 = [-df_epgg["Epx"] - df_epgg["Ppx"], -df_epgg["Epy"] -
		            df_epgg["Ppy"], pbeam - df_epgg["Epz"] - df_epgg["Ppz"]]
		VmissP = [-df_epgg["Epx"] - df_epgg["Gpx"] - df_epgg["Gpx2"], -df_epgg["Epy"] -
		            df_epgg["Gpy"] - df_epgg["Gpy2"], pbeam - df_epgg["Epz"] - df_epgg["Gpz"] - df_epgg["Gpz2"]]
		Vmiss = [-df_epgg["Epx"] - df_epgg["Ppx"] - df_epgg["Gpx"] - df_epgg["Gpx2"],
		            -df_epgg["Epy"] - df_epgg["Ppy"] - df_epgg["Gpy"] - df_epgg["Gpy2"],
		            pbeam - df_epgg["Epz"] - df_epgg["Ppz"] - df_epgg["Gpz"] - df_epgg["Gpz2"]]

		df_epgg.loc[:, 'Mpx'], df_epgg.loc[:, 'Mpy'], df_epgg.loc[:, 'Mpz'] = Vmiss

		# binning kinematics
		df_epgg.loc[:,'Q2'] = -((ebeam - df_epgg['Ee'])**2 - mag2(VGS))
		df_epgg.loc[:,'nu'] = (ebeam - df_epgg['Ee'])
		df_epgg.loc[:,'xB'] = df_epgg['Q2'] / 2.0 / M / df_epgg['nu']
		df_epgg.loc[:,'t'] = 2 * M * (df_epgg['Pe'] - M)
		df_epgg.loc[:,'W'] = np.sqrt(np.maximum(0, (ebeam + M - df_epgg['Ee'])**2 - mag2(VGS)))
		df_epgg.loc[:,'MPt'] = np.sqrt((df_epgg["Epx"] + df_epgg["Ppx"] + df_epgg["Gpx"] + df_epgg["Gpx2"])**2 +
		                         (df_epgg["Epy"] + df_epgg["Ppy"] + df_epgg["Gpy"] + df_epgg["Gpy2"])**2)

		# exclusivity variables
		df_epgg.loc[:,'MM2_ep'] = (-M - ebeam + df_epgg["Ee"] +
		                     df_epgg["Pe"])**2 - mag2(VmissPi0)
		df_epgg.loc[:,'MM2_egg'] = (-M - ebeam + df_epgg["Ee"] +
		                     df_epgg["Ge"] + df_epgg["Ge2"])**2 - mag2(VmissP)
		df_epgg.loc[:,'MM2_epgg'] = (-M - ebeam + df_epgg["Ee"] + df_epgg["Pe"] +
		                     df_epgg["Ge"] + df_epgg["Ge2"])**2 - mag2(Vmiss)
		df_epgg.loc[:,'ME_epgg'] = (M + ebeam - df_epgg["Ee"] - df_epgg["Pe"] - df_epgg["Ge"] - df_epgg["Ge2"])
		df_epgg.loc[:,'Mpi0'] = pi0InvMass(gam, gam2)
		df_epgg.loc[:,'reconPi'] = angle(VmissPi0, pi0)
		df_epgg.loc[:,"Pie"] = df_epgg['Ge'] + df_epgg['Ge2']
		self.df_epgg = df_epgg

	def makeDVpi0(self):
		#make dvpi0 pairs
		df_epgg = self.df_epgg
		print(len(df_epgg))
		cut_xBupper = df_epgg["xB"] < 1  # xB
		cut_xBlower = df_epgg["xB"] > 0  # xB
		cut_Q2 = df_epgg["Q2"] > 1  # Q2
		cut_W = df_epgg["W"] > 2  # W
		cut_Vz = np.abs(df_epgg["Evz"] - df_epgg["Pvz"]) < 2.5 + 2.5 / mag([df_epgg["Ppx"], df_epgg["Ppy"], df_epgg["Ppz"]])

		# Exclusivity cuts
		cut_mmep = df_epgg["MM2_ep"] < 0.7  # mmep
		cut_meepgg = df_epgg["ME_epgg"] < 0.7  # meepgg
		cut_mpt = df_epgg["MPt"] < 0.2  # mpt
		cut_recon = df_epgg["reconPi"] < 2  # recon gam angle
		cut_pi0upper = df_epgg["Mpi0"] < 0.2
		cut_pi0lower = df_epgg["Mpi0"] > 0.07
		if ("Esector" in df_epgg.columns):
			cut_sector = (df_epgg.loc[:, "Esector"]!=df_epgg.loc[:, "Gsector"]) & (df_epgg.loc[:, "Esector"]!=df_epgg.loc[:, "Gsector2"])
		else:
			cut_sector = 1

		df_dvpi0 = df_epgg.loc[cut_xBupper & cut_xBlower & cut_Q2 & cut_W & cut_Vz & cut_mmep & cut_meepgg &
		                   cut_mpt & cut_recon & cut_pi0upper & cut_pi0lower & cut_sector, :]

		self.df_dvpi0 = df_dvpi0

		# #cut by detector status
		# cut_pFD = df_dvpi0["Pstat"] < 4000  # FD
		# cut_pCD = df_dvpi0["Pstat"] > 4000  # CD
		# cut_gFD = df_dvpi0["Gstat"] > 2000  # FD
		# cut_gFT = df_dvpi0["Gstat"] < 2000  # FT
		# cut_g2FD = df_dvpi0["Gstat2"] > 2000  # FD
		# cut_g2FT = df_dvpi0["Gstat2"] < 2000  # FT
		# cut_FTFT = cut_gFT & cut_g2FT
		# cut_FDFT = (cut_gFD & cut_g2FD) | (cut_gFT & cut_g2FT)
		# cut_FDFD = cut_gFD & cut_g2FD

	def getEPGG(self):
		#returning pd.DataFrame of epgg
		self.setDVpi0vars()
		return self.df_epgg

	def getDVpi0(self, correction = None):
		#returning pd.DataFrame of dvpi0
		self.setDVpi0vars(correction)
		self.makeDVpi0()
		return self.df_dvpi0

	def getDVCS(self, correction = None, sub2g = None):
		#returning pd.DataFrame of dvcs
		self.setDVCSvars(correction)
		self.makeDVCS()
		if(sub2g):
			self.pi02gSubtraction()
		return self.df_dvcs

	def pi02gSubtraction(self):
		#exclude dvpi0 from dvcs. use only when both set up.
		df_dvcs = self.df_dvcs
		pi0to2gammas = df_dvcs["event"].isin(self.df_dvpi0["event"])
		df_dvcs = df_dvcs[~pi0to2gammas]
		self.df_dvcs = df_dvcs

class epgFromROOT(epg):
	#class to read root to make epg pairs, inherited from epg
	def __init__(self, fname, entry_stop = None, mc = False, rec = False):
		self.fname = fname
		self.mc = mc
		self.rec = rec
		self.readEPG(entry_stop)

	def readFile(self):
		#read root using uproot
		self.file = uproot.open(self.fname)
		self.tree = self.file["T"]
		print(self.tree.keys)

	def closeFile(self):
		#close file for saving memory
		self.file = None
		self.tree = None

	def readEPG(self, entry_stop = None):
		#save data into df_epg, df_epgg for parent class epg
		self.readFile()

		df_logistics = pd.DataFrame()
		df_electron = pd.DataFrame()
		df_proton = pd.DataFrame()
		df_gamma = pd.DataFrame()

		logistics = ["nmg"]
		eleKeys = ["Epx", "Epy", "Epz", "Evx", "Evy", "Evz", "Esector"]
		proKeys = ["Ppx", "Ppy", "Ppz", "Pvz", "Pstat", "Psector"]
		gamKeys = ["Gpx", "Gpy", "Gpz", "GorigIndx", "Gstat", "Gsector"]
		if (self.mc):
			eleKeys = eleKeys[:-1]
			proKeys = proKeys[:-3]
			gamKeys = gamKeys[:-3]

		for key in logistics:
			df_logistics[key] = self.tree[key].array(library="pd", entry_stop=entry_stop)

		for key in eleKeys:
			df_electron[key] = self.tree[key].array(library="pd", entry_stop=entry_stop)

		for key in proKeys:
			df_proton[key] = self.tree[key].array(library="pd", entry_stop=entry_stop)

		for key in gamKeys:
			df_gamma[key] = self.tree[key].array(library="pd", entry_stop=entry_stop)

		if (self.rec):
			df_MC = pd.DataFrame()
			df_MC2 = pd.DataFrame()
			MCKeys = ["MCEpx", "MCEpy", "MCEpz", "MCEvx", "MCEvy", "MCEvz", "MCPpx", "MCPpy", "MCPpz"]
			MCKeys2 = ["MCGpx", "MCGpy", "MCGpz"]
			for key in MCKeys:
				df_MC[key] = self.tree[key].array(library="pd", entry_stop=entry_stop)
			for key in MCKeys2:
				df_MC2[key] = self.tree[key].array(library="pd", entry_stop=entry_stop)

			ele = [df_MC['MCEpx'], df_MC['MCEpy'], df_MC['MCEpz']]
			pro = [df_MC['MCPpx'], df_MC['MCPpy'], df_MC['MCPpz']]
			gam = [df_MC2['MCGpx'], df_MC2['MCGpy'], df_MC2['MCGpz']]

			df_MC.loc[:, 'MCEe'] = getEnergy(ele, me)
			df_MC.loc[:, 'MCPe'] = getEnergy(pro, M)
			df_MC2.loc[:, 'MCGe'] = getEnergy(gam, 0)

			df_MC.loc[:, 'MCEp'] = mag(ele)
			df_MC.loc[:, 'MCPp'] = mag(pro)
			df_MC2.loc[:, 'MCGp'] = mag(gam)

			df_MC.loc[:, 'MCEtheta'] = getTheta(ele)
			df_MC.loc[:, 'MCPtheta'] = getTheta(pro)
			df_MC2.loc[:, 'MCGtheta'] = getTheta(gam)

			df_MC.loc[:, 'MCEphi'] = getPhi(ele)
			df_MC.loc[:, 'MCPphi'] = getPhi(pro)
			df_MC2.loc[:, 'MCGphi'] = getPhi(gam)

			df_MC['event'] = df_MC.index.get_level_values('entry')
			df_MC2['event'] = df_MC2.index.get_level_values('entry')

			df_MC3 = pd.merge(df_MC, df_MC2,
			                 how='outer', on='event')

			self.df_MC = df_MC3

		self.closeFile()

		df_electron = df_electron.astype({"Epx": float, "Epy": float, "Epz": float, "Evz": float})
		df_proton = df_proton.astype({"Ppx": float, "Ppy": float, "Ppz": float, "Pvz": float})
		df_gamma = df_gamma.astype({"Gpx": float, "Gpy": float, "Gpz": float})

		df_logistics.loc[:,'event'] = df_logistics.index
		df_electron.loc[:,'event'] = df_electron.index.get_level_values('entry')
		df_proton.loc[:,'event'] = df_proton.index.get_level_values('entry')
		df_gamma.loc[:,'event'] = df_gamma.index.get_level_values('entry')

		# df_electron = pd.merge(df_electron, df_logistics,
		# 						how='outer', on='event')
		if ('Pstat' in df_proton):
			df_proton = df_proton[df_proton["Pstat"]<4000]
			df_gamma = df_gamma[df_gamma["Gstat"]>2000]

		#if ('PorigIndx' not in df_proton.index):
	#		df_proton['PorigIndx'] = df_proton.index.get_level_values('subentry')
		if ('GorigIndx' not in df_gamma.index):
			df_gamma['GorigIndx'] = df_gamma.index.get_level_values('subentry')

		df_gg = pd.merge(df_gamma, df_gamma,
		                 how='outer', on='event', suffixes=("", "2"))
		df_gg = df_gg[df_gg["GorigIndx"] < df_gg["GorigIndx2"]]
		df_ep = pd.merge(df_electron, df_proton, how='outer', on='event')
		df_epg = pd.merge(df_ep, df_gamma, how='outer', on='event')
		df_epg = df_epg[~np.isnan(df_epg["Gpx"])]
		df_epgg = pd.merge(df_ep, df_gg, how='outer', on='event')
		df_epgg = df_epgg[~np.isnan(df_epgg["Ppx"])]
		df_epgg = df_epgg[~np.isnan(df_epgg["Gpx"])]
		df_epgg = df_epgg[~np.isnan(df_epgg["Gpx2"])]

		self.df_ep = df_ep
		self.df_epg = df_epg
		self.df_epgg = df_epgg

class epgFromLund(epg):
	#class to read lund format to make epg pairs, inherited from epg
	def __init__(self, fname, entry_stop = None):
		self.fname = fname
		self.readEPG(entry_stop)

	def readFile(self):
		#read tsv file using python built-in functions
		with open(self.fname,"r") as file:
		    self.data = file.read()
	
	def closeFile(self):
		#close file for saving memory
		self.data = None

	def readEPG(self, entry_stop = None):
		#save data into df_epg, df_epgg for parent class epg
		self.readFile()
		partArray = []

		txtlst = self.data.split("\n")
		for ind, line in enumerate(txtlst[:-1]):
			if entry_stop:
				if ind==entry_stop:
					break
			if ind %400000 == 0:
				print("On event {}".format(ind/4))
			if ind % 4 == 0:
				header = line
				eleLine = txtlst[ind+1]
				eleQuantities = eleLine.split()
				Epx = eleQuantities[6]
				Epy = eleQuantities[7]
				Epz = eleQuantities[8]
				Evx = eleQuantities[11]
				Evy = eleQuantities[12]
				Evz = eleQuantities[13]
				proLine = txtlst[ind+2]
				proQuantities = proLine.split()
				Ppx = proQuantities[6]
				Ppy = proQuantities[7]
				Ppz = proQuantities[8]
				Pvz = proQuantities[13]
				gamLine = txtlst[ind+3]
				gamQuantities = gamLine.split()
				Gpx = gamQuantities[6]
				Gpy = gamQuantities[7]
				Gpz = gamQuantities[8]
				partArray.append([float(Epx), float(Epy), float(Epz), float(Evx), float(Evy), float(Evz), float(Ppx), float(Ppy), float(Ppz), float(Gpx), float(Gpy), float(Gpz)])

		self.df_epg = pd.DataFrame(partArray, columns = ["Epx", "Epy", "Epz", "Evx", "Evy", "Evz", "Ppx", "Ppy", "Ppz", "Pvz", "Gpx", "Gpy", "Gpz"])
		self.df_epgg = None
		self.closeFile()