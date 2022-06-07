def makeDVpi0(self):
        #make dvpi0 pairs
        df_epgg = self.df_epgg

        df_epgg.loc[:, "closeness"] = np.abs(df_epgg.loc[:, "Mpi0"] - .1349766)

        cut_xBupper = df_epgg.loc[:, "xB"] < 1  # xB
        cut_xBlower = df_epgg.loc[:, "xB"] > 0  # xB
        cut_Q2 = df_epgg.loc[:, "Q2"] > 1  # Q2
        cut_W = df_epgg.loc[:, "W"] > 2  # W

        # proton reconstruction quality
        cut_FD_proton = (df_epgg.loc[:, "Psector"]<7) & (df_epgg.loc[:, "Ptheta"]<35)
        cut_CD_proton = (df_epgg.loc[:, "Psector"]>7) & (df_epgg.loc[:, "Ptheta"]>45) & (df_epgg.loc[:, "Ptheta"]<65)
        cut_proton = (cut_FD_proton)|(cut_CD_proton)

        # Exclusivity cuts
        cut_mmep = df_epgg.loc[:, "MM2_ep"] < 0.7  # mmep
        cut_meepgg = df_epgg.loc[:, "ME_epgg"] < 0.7  # meepgg
        cut_mpt = df_epgg.loc[:, "MPt"] < 0.2  # mpt
        cut_recon = df_epgg.loc[:, "reconPi"] < 2  # recon gam angle
        cut_pi0upper = df_epgg.loc[:, "Mpi0"] < 0.2
        cut_pi0lower = df_epgg.loc[:, "Mpi0"] > 0.07
        cut_sector = (df_epgg.loc[:, "Esector"]!=df_epgg.loc[:, "Gsector"]) & (df_epgg.loc[:, "Esector"]!=df_epgg.loc[:, "Gsector2"])
        #cut_Vz = np.abs(df_epgg["Evz"] - df_epgg["Pvz"]) < 2.5 + 2.5 / mag([df_epgg["Ppx"], df_epgg["Ppy"], df_epgg["Ppz"]])

        df_dvpi0 = df_epgg.loc[cut_xBupper & cut_xBlower & cut_Q2 & cut_W & cut_proton & cut_mmep & cut_meepgg &
                           cut_mpt & cut_recon & cut_pi0upper & cut_pi0lower & cut_sector, :]

        #For an event, there can be two gg's passed conditions above.
        #Take only one gg's that makes pi0 invariant mass
        #This case is very rare.
        #For now, duplicated proton is not considered.
        df_dvpi0 = df_dvpi0.sort_values(by=['closeness', 'Psector', 'Gsector'], ascending = [True, True, True])
        df_dvpi0 = df_dvpi0.loc[~df_dvpi0.event.duplicated(), :]
        df_dvpi0 = df_dvpi0.sort_values(by='event')        
        self.df_dvpi0 = df_dvpi0 #done with saving x



    def makeDVpi0P_DVCS(self, pol = "inbending"):
        #make dvpi0 pairs
        df_dvpi0p = self.df_epgg

        #common cuts
        cut_xBupper = df_dvpi0p["xB"] < 1  # xB
        cut_xBlower = df_dvpi0p["xB"] > 0  # xB
        cut_Q2 = df_dvpi0p["Q2"] > 1  # Q2
        cut_W = df_dvpi0p["W"] > 2  # W
        cut_Ee = df_dvpi0p["Ee"] > 2  # Ee
        cut_Ge = df_dvpi0p["Ge"] > 3  # Ge
        cut_Esector = df_dvpi0p["Esector"]!=df_dvpi0p["Gsector"]
        cut_Ppmax = df_dvpi0p.Pp < 0.8  # Pp
        # cut_Vz = np.abs(df_dvpi0p["Evz"] - df_dvpi0p["Pvz"]) < 2.5 + 2.5 / mag([df_dvpi0p["Ppx"], pi0SimInb_forDVCS["Ppy"], pi0SimInb_forDVCS["Ppz"]])
        cut_common = cut_xBupper & cut_xBlower & cut_Q2 & cut_W & cut_Ee & cut_Ge & cut_Esector & cut_Ppmax

        df_dvpi0p = df_dvpi0p[cut_common]

        # proton reconstruction quality
        # cut_FD_proton = (df_epgg.loc[:, "Psector"]<7) & (df_epgg.loc[:, "Ptheta"]<35)
        # cut_CD_proton = (df_epgg.loc[:, "Psector"]>7) & (df_epgg.loc[:, "Ptheta"]>45) & (df_epgg.loc[:, "Ptheta"]<65)
        # cut_proton = (cut_FD_proton)|(cut_CD_proton)
        cut_proton = 1

        df_dvpi0p.loc[:, "config"] = 0
        
        if pol == 'inbending':
            #CDFT
            cut_Pp1_CDFT = df_dvpi0p.Pp > 0.25  # Pp
            cut_Psector_CDFT = df_dvpi0p.Psector>7
            cut_Ptheta_CDFT = df_dvpi0p.Ptheta<60
            cut_Gsector_CDFT = df_dvpi0p.Gsector>7
            cut_mmep1_CDFT = df_dvpi0p["MM2_ep"] < 0.599  # mmep
            cut_mmep2_CDFT = df_dvpi0p["MM2_ep"] > -0.528  # mmep
            cut_mpi01_CDFT = df_dvpi0p["Mpi0"] < 0.161  # mpi0
            cut_mpi02_CDFT = df_dvpi0p["Mpi0"] > 0.115  # mpi0
            cut_mmegg1_CDFT = df_dvpi0p["MM2_egg"] < 2.178  # mmegg
            cut_mmegg2_CDFT = df_dvpi0p["MM2_egg"] > -0.429  # mmegg
            cut_meepgg1_CDFT = df_dvpi0p["ME_epgg"] < 0.908  # meepgg
            cut_meepgg2_CDFT = df_dvpi0p["ME_epgg"] > -0.906  # meepgg
            cut_mpt_CDFT = df_dvpi0p["MPt"] < 0.195  # mpt
            cut_recon_CDFT = df_dvpi0p["reconPi"] < 1.520  # recon gam angle
            cut_mmepgg1_CDFT = np.abs(df_dvpi0p["MM2_epgg"]) < 0.0405  # mmepgg
            cut_mmepgg2_CDFT = np.abs(df_dvpi0p["MM2_epgg"]) > -0.0442  # mmepgg

            cut_CDFT = (cut_Pp1_CDFT & cut_Psector_CDFT & cut_Ptheta_CDFT & cut_Gsector_CDFT &
                        cut_mmep1_CDFT & cut_mmep2_CDFT & cut_mpi01_CDFT & cut_mpi02_CDFT & 
                        cut_mmegg1_CDFT & cut_mmegg2_CDFT & cut_meepgg1_CDFT & cut_meepgg2_CDFT &
                        cut_mpt_CDFT & cut_recon_CDFT & cut_mmepgg1_CDFT & cut_mmepgg2_CDFT)


            #CD
            cut_Pp1_CD = df_dvpi0p.Pp > 0.25  # Pp
            cut_Psector_CD = df_dvpi0p.Psector>7
            cut_Ptheta_CD = df_dvpi0p.Ptheta<60
            cut_Gsector_CD = df_dvpi0p.Gsector<7
            cut_mmep1_CD = df_dvpi0p["MM2_ep"] < 0.410  # mmep
            cut_mmep2_CD = df_dvpi0p["MM2_ep"] > -0.401  # mmep
            cut_mpi01_CD = df_dvpi0p["Mpi0"] < 0.164  # mpi0
            cut_mpi02_CD = df_dvpi0p["Mpi0"] > 0.105  # mpi0
            cut_mmegg1_CD = df_dvpi0p["MM2_egg"] < 2.099  # mmegg
            cut_mmegg2_CD = df_dvpi0p["MM2_egg"] > -0.303  # mmegg
            cut_meepgg1_CD = df_dvpi0p["ME_epgg"] < 0.871  # meepgg
            cut_meepgg2_CD = df_dvpi0p["ME_epgg"] > -0.819  # meepgg
            cut_mpt_CD = df_dvpi0p["MPt"] < 0.172  # mpt
            cut_recon_CD = df_dvpi0p["reconPi"] < 1.101  # recon gam angle
            cut_mmepgg1_CD = np.abs(df_dvpi0p["MM2_epgg"]) < 0.0232  # mmepgg
            cut_mmepgg2_CD = np.abs(df_dvpi0p["MM2_epgg"]) > -0.0276  # mmepgg

            cut_CD = (cut_Pp1_CD & cut_Psector_CD & cut_Ptheta_CD & cut_Gsector_CD &
                        cut_mmep1_CD & cut_mmep2_CD & cut_mpi01_CD & cut_mpi02_CD & 
                        cut_mmegg1_CD & cut_mmegg2_CD & cut_meepgg1_CD & cut_meepgg2_CD &
                        cut_mpt_CD & cut_recon_CD & cut_mmepgg1_CD & cut_mmepgg2_CD)

            #FD
            cut_Pp1_FD = df_dvpi0p.Pp > 0.35  # Pp
            cut_Psector_FD = df_dvpi0p.Psector<7
            cut_Ptheta_FD = df_dvpi0p.Ptheta>2.477
            cut_Gsector_FD = df_dvpi0p.Gsector<7
            cut_mmep1_FD = df_dvpi0p["MM2_ep"] < 0.553  # mmep
            cut_mmep2_FD = df_dvpi0p["MM2_ep"] > -0.569  # mmep
            cut_mpi01_FD = df_dvpi0p["Mpi0"] < 0.167  # mpi0
            cut_mpi02_FD = df_dvpi0p["Mpi0"] > 0.104  # mpi0
            cut_mmegg1_FD = df_dvpi0p["MM2_egg"] < 1.771  # mmegg
            cut_mmegg2_FD = df_dvpi0p["MM2_egg"] > -0.0598  # mmegg
            cut_meepgg1_FD = df_dvpi0p["ME_epgg"] < 0.805  # meepgg
            cut_meepgg2_FD = df_dvpi0p["ME_epgg"] > -0.813  # meepgg
            cut_mpt_FD = df_dvpi0p["MPt"] < 0.231  # mpt
            cut_recon_FD = df_dvpi0p["reconPi"] < 1.098  # recon gam angle
            cut_mmepgg1_FD = np.abs(df_dvpi0p["MM2_epgg"]) < 0.0246  # mmepgg
            cut_mmepgg2_FD = np.abs(df_dvpi0p["MM2_epgg"]) > -0.0287  # mmepgg

            cut_FD = (cut_Pp1_FD & cut_Psector_FD & cut_Ptheta_FD & cut_Gsector_FD &
                        cut_mmep1_FD & cut_mmep2_FD & cut_mpi01_FD & cut_mpi02_FD & 
                        cut_mmegg1_FD & cut_mmegg2_FD & cut_meepgg1_FD & cut_meepgg2_FD &
                        cut_mpt_FD & cut_recon_FD & cut_mmepgg1_FD & cut_mmepgg2_FD)

        elif pol == "outbending":
            #CDFT
            cut_Pp1_CDFT = df_dvpi0p.Pp > 0.25  # Pp
            cut_Psector_CDFT = df_dvpi0p.Psector>7
            cut_Ptheta_CDFT = df_dvpi0p.Ptheta<60
            cut_Gsector_CDFT = df_dvpi0p.Gsector>7
            cut_mmep1_CDFT = df_dvpi0p["MM2_ep"] < 0.617  # mmep
            cut_mmep2_CDFT = df_dvpi0p["MM2_ep"] > -0.521  # mmep
            cut_mpi01_CDFT = df_dvpi0p["Mpi0"] < 0.163  # mpi0
            cut_mpi02_CDFT = df_dvpi0p["Mpi0"] > 0.112  # mpi0
            cut_mmegg1_CDFT = df_dvpi0p["MM2_egg"] < 2.182  # mmegg
            cut_mmegg2_CDFT = df_dvpi0p["MM2_egg"] > -0.432  # mmegg
            cut_meepgg1_CDFT = df_dvpi0p["ME_epgg"] < 0.906  # meepgg
            cut_meepgg2_CDFT = df_dvpi0p["ME_epgg"] > -0.887  # meepgg
            cut_mpt_CDFT = df_dvpi0p["MPt"] < 0.213  # mpt
            cut_recon_CDFT = df_dvpi0p["reconPi"] < 1.531  # recon gam angle
            cut_mmepgg1_CDFT = np.abs(df_dvpi0p["MM2_epgg"]) < 0.0440  # mmepgg
            cut_mmepgg2_CDFT = np.abs(df_dvpi0p["MM2_epgg"]) > -0.0473  # mmepgg

            cut_CDFT = (cut_Pp1_CDFT & cut_Psector_CDFT & cut_Ptheta_CDFT & cut_Gsector_CDFT &
                        cut_mmep1_CDFT & cut_mmep2_CDFT & cut_mpi01_CDFT & cut_mpi02_CDFT & 
                        cut_mmegg1_CDFT & cut_mmegg2_CDFT & cut_meepgg1_CDFT & cut_meepgg2_CDFT &
                        cut_mpt_CDFT & cut_recon_CDFT & cut_mmepgg1_CDFT & cut_mmepgg2_CDFT)


            #CD
            cut_Pp1_CD = df_dvpi0p.Pp > 0.25  # Pp
            cut_Psector_CD = df_dvpi0p.Psector>7
            cut_Ptheta_CD = df_dvpi0p.Ptheta<60
            cut_Gsector_CD = df_dvpi0p.Gsector<7
            cut_mmep1_CD = df_dvpi0p["MM2_ep"] < 0.362  # mmep
            cut_mmep2_CD = df_dvpi0p["MM2_ep"] > -0.340  # mmep
            cut_mpi01_CD = df_dvpi0p["Mpi0"] < 0.167  # mpi0
            cut_mpi02_CD = df_dvpi0p["Mpi0"] > 0.104  # mpi0
            cut_mmegg1_CD = df_dvpi0p["MM2_egg"] < 2.089  # mmegg
            cut_mmegg2_CD = df_dvpi0p["MM2_egg"] > -0.363  # mmegg
            cut_meepgg1_CD = df_dvpi0p["ME_epgg"] < 0.806  # meepgg
            cut_meepgg2_CD = df_dvpi0p["ME_epgg"] > -0.810  # meepgg
            cut_mpt_CD = df_dvpi0p["MPt"] < 0.191  # mpt
            cut_recon_CD = df_dvpi0p["reconPi"] < 1.274  # recon gam angle
            cut_mmepgg1_CD = np.abs(df_dvpi0p["MM2_epgg"]) < 0.0215  # mmepgg
            cut_mmepgg2_CD = np.abs(df_dvpi0p["MM2_epgg"]) > -0.0259  # mmepgg

            cut_CD = (cut_Pp1_CD & cut_Psector_CD & cut_Ptheta_CD & cut_Gsector_CD &
                        cut_mmep1_CD & cut_mmep2_CD & cut_mpi01_CD & cut_mpi02_CD & 
                        cut_mmegg1_CD & cut_mmegg2_CD & cut_meepgg1_CD & cut_meepgg2_CD &
                        cut_mpt_CD & cut_recon_CD & cut_mmepgg1_CD & cut_mmepgg2_CD)

            #FD
            cut_Pp1_FD = df_dvpi0p.Pp > 0.35  # Pp
            cut_Psector_FD = df_dvpi0p.Psector<7
            cut_Ptheta_FD = df_dvpi0p.Ptheta>2.477
            cut_Gsector_FD = df_dvpi0p.Gsector<7
            cut_mmep1_FD = df_dvpi0p["MM2_ep"] < 0.614  # mmep
            cut_mmep2_FD = df_dvpi0p["MM2_ep"] > -0.612  # mmep
            cut_mpi01_FD = df_dvpi0p["Mpi0"] < 0.168  # mpi0
            cut_mpi02_FD = df_dvpi0p["Mpi0"] > 0.103  # mpi0
            cut_mmegg1_FD = df_dvpi0p["MM2_egg"] < 1.871  # mmegg
            cut_mmegg2_FD = df_dvpi0p["MM2_egg"] > -0.192  # mmegg
            cut_meepgg1_FD = df_dvpi0p["ME_epgg"] < 0.823  # meepgg
            cut_meepgg2_FD = df_dvpi0p["ME_epgg"] > -0.781  # meepgg
            cut_mpt_FD = df_dvpi0p["MPt"] < 0.220  # mpt
            cut_recon_FD = df_dvpi0p["reconPi"] < 1.386  # recon gam angle
            cut_mmepgg1_FD = np.abs(df_dvpi0p["MM2_epgg"]) < 0.0291  # mmepgg
            cut_mmepgg2_FD = np.abs(df_dvpi0p["MM2_epgg"]) > -0.0354  # mmepgg

            cut_FD = (cut_Pp1_FD & cut_Psector_FD & cut_Ptheta_FD & cut_Gsector_FD &
                        cut_mmep1_FD & cut_mmep2_FD & cut_mpi01_FD & cut_mpi02_FD & 
                        cut_mmegg1_FD & cut_mmegg2_FD & cut_meepgg1_FD & cut_meepgg2_FD &
                        cut_mpt_FD & cut_recon_FD & cut_mmepgg1_FD & cut_mmepgg2_FD)

        #df_dvpi0p.loc[cut_CDFT, "config"] = 3
        #df_dvpi0p.loc[cut_CD, "config"] = 2
        #df_dvpi0p.loc[cut_FD, "config"] = 1

        #df_dvpi0p = df_dvpi0p[df_dvpi0p.config>0]
    
        self.df_dvpi0p = df_dvpi0p #no need to reduce duplicates of pi0. remove the event if any.

