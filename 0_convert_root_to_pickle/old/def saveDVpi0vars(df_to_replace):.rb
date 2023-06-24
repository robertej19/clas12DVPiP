def saveDVpi0vars(df_to_replace):
    #set up pi0 variables

    # useful objects
    ele = [df_to_replace['Epx'], df_to_replace['Epy'], df_to_replace['Epz']]
    df_to_replace.loc[:, 'Ep'] = mag(ele)
    df_to_replace.loc[:, 'Ee'] = getEnergy(ele, me)
    df_to_replace.loc[:, 'Etheta'] = getTheta(ele)
    df_to_replace.loc[:, 'Ephi'] = getPhi(ele)

    pro = [df_to_replace['Ppx'], df_to_replace['Ppy'], df_to_replace['Ppz']]

    gam = [df_to_replace['Gpx'], df_to_replace['Gpy'], df_to_replace['Gpz']]
    df_to_replace.loc[:, 'Gp'] = mag(gam)
    df_to_replace.loc[:, 'Ge'] = getEnergy(gam, 0)
    df_to_replace.loc[:, 'Gtheta'] = getTheta(gam)
    df_to_replace.loc[:, 'Gphi'] = getPhi(gam)

    gam2 = [df_to_replace['Gpx2'], df_to_replace['Gpy2'], df_to_replace['Gpz2']]
    df_to_replace.loc[:, 'Gp2'] = mag(gam2)
    df_to_replace.loc[:,'Ge2'] = getEnergy(gam2, 0)
    df_to_replace.loc[:, 'Gtheta2'] = getTheta(gam2)
    df_to_replace.loc[:, 'Gphi2'] = getPhi(gam2)

    pi0 = vecAdd(gam, gam2)
    VGS = [-df_to_replace['Epx'], -df_to_replace['Epy'], pbeam - df_to_replace['Epz']]
    v3l = cross(beam, ele)
    v3h = cross(pro, VGS)
    v3g = cross(VGS, gam)
    v3pi0 = cross(VGS, pi0)

    VmissPi0 = [-df_to_replace["Epx"] - df_to_replace["Ppx"], -df_to_replace["Epy"] -
                df_to_replace["Ppy"], pbeam - df_to_replace["Epz"] - df_to_replace["Ppz"]]
    VmissP = [-df_to_replace["Epx"] - df_to_replace["Gpx"] - df_to_replace["Gpx2"], -df_to_replace["Epy"] -
                df_to_replace["Gpy"] - df_to_replace["Gpy2"], pbeam - df_to_replace["Epz"] - df_to_replace["Gpz"] - df_to_replace["Gpz2"]]
    Vmiss = [-df_to_replace["Epx"] - df_to_replace["Ppx"] - df_to_replace["Gpx"] - df_to_replace["Gpx2"],
                -df_to_replace["Epy"] - df_to_replace["Ppy"] - df_to_replace["Gpy"] - df_to_replace["Gpy2"],
                pbeam - df_to_replace["Epz"] - df_to_replace["Ppz"] - df_to_replace["Gpz"] - df_to_replace["Gpz2"]]
    costheta = cosTheta(VGS, gam)

    df_to_replace.loc[:, 'Mpx'], df_to_replace.loc[:, 'Mpy'], df_to_replace.loc[:, 'Mpz'] = Vmiss

    # binning kinematics
    df_to_replace.loc[:,'Q2'] = -((ebeam - df_to_replace['Ee'])**2 - mag2(VGS))
    df_to_replace.loc[:,'nu'] = (ebeam - df_to_replace['Ee'])
    df_to_replace.loc[:,'xB'] = df_to_replace['Q2'] / 2.0 / M / df_to_replace['nu']
    df_to_replace.loc[:,'t1'] = 2 * M * (df_to_replace['Pe'] - M)
    df_to_replace.loc[:,'t2'] = (M * df_to_replace['Q2'] + 2 * M * df_to_replace['nu'] * (df_to_replace['nu'] - np.sqrt(df_to_replace['nu'] * df_to_replace['nu'] + df_to_replace['Q2']) * costheta))\
    / (M + df_to_replace['nu'] - np.sqrt(df_to_replace['nu'] * df_to_replace['nu'] + df_to_replace['Q2']) * costheta)
    df_to_replace.loc[:,'W'] = np.sqrt(np.maximum(0, (ebeam + M - df_to_replace['Ee'])**2 - mag2(VGS)))
    df_to_replace.loc[:,'MPt'] = np.sqrt((df_to_replace["Epx"] + df_to_replace["Ppx"] + df_to_replace["Gpx"] + df_to_replace["Gpx2"])**2 +
                             (df_to_replace["Epy"] + df_to_replace["Ppy"] + df_to_replace["Gpy"] + df_to_replace["Gpy2"])**2)
    # trento angles
    df_to_replace.loc[:,'phi1'] = angle(v3l, v3h)
    df_to_replace.loc[:,'phi1'] = np.where(dot(v3l, pro) > 0, 360.0 -
                              df_to_replace['phi1'], df_to_replace['phi1'])
    df_to_replace.loc[:,'phi2'] = angle(v3l, v3g)
    df_to_replace.loc[:,'phi2'] = np.where(dot(v3l, gam) <
                              0, 360.0 - df_to_replace['phi2'], df_to_replace['phi2'])

    # exclusivity variables
    df_to_replace.loc[:,'MM2_ep'] = (-M - ebeam + df_to_replace["Ee"] +
                         df_to_replace["Pe"])**2 - mag2(VmissPi0)
    df_to_replace.loc[:,'MM2_egg'] = (-M - ebeam + df_to_replace["Ee"] +
                         df_to_replace["Ge"] + df_to_replace["Ge2"])**2 - mag2(VmissP)
    df_to_replace.loc[:,'MM2_epgg'] = (-M - ebeam + df_to_replace["Ee"] + df_to_replace["Pe"] +
                         df_to_replace["Ge"] + df_to_replace["Ge2"])**2 - mag2(Vmiss)
    df_to_replace.loc[:,'ME_epgg'] = (M + ebeam - df_to_replace["Ee"] - df_to_replace["Pe"] - df_to_replace["Ge"] - df_to_replace["Ge2"])
    df_to_replace.loc[:,'Mpi0'] = pi0InvMass(gam, gam2)
    df_to_replace.loc[:,'reconPi'] = angle(VmissPi0, pi0)
    df_to_replace.loc[:,"Pie"] = df_to_replace['Ge'] + df_to_replace['Ge2']
    df_to_replace.loc[:,'coplanarity'] = angle(v3h, v3pi0)
    df_to_replace.loc[:,'coneAngle1'] = angle(ele, gam)
    df_to_replace.loc[:,'coneAngle2'] = angle(ele, gam2)

    df_to_replace.loc[:, "closeness"] = np.abs(df_to_replace.loc[:, "Mpi0"] - .1349766)

    return df_to_replace