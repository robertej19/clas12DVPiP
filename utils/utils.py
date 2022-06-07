import numpy as np

def dot(vec1, vec2):
    # dot product of two 3d vectors
    return vec1[0]*vec2[0]+vec1[1]*vec2[1]+vec1[2]*vec2[2]

def mag(vec1):
    # L2 norm of vector
    return np.sqrt(dot(vec1, vec1))

def mag2(vec1):
    # square of L2 norm
    return  dot(vec1, vec1)

def cosTheta(vec1, vec2):
    # cosine angle between two 3d vectors
    return dot(vec1,vec2)/np.sqrt(mag2(vec1) * mag2(vec2))

def angle(vec1, vec2):
    # angle between two 3d vectors
    return 180/np.pi*np.arccos(np.minimum(1, cosTheta(vec1, vec2)))

def cross(vec1, vec2):
    # cross product of two 3d vectors
    return [vec1[1]*vec2[2]-vec1[2]*vec2[1], vec1[2]*vec2[0]-vec1[0]*vec2[2], vec1[0]*vec2[1]-vec1[1]*vec2[0]]

def vecAdd(gam1, gam2):
    # add two 3d vectors
    return [gam1[0]+gam2[0], gam1[1]+gam2[1], gam1[2]+gam2[2]]

def pi0Energy(gam1, gam2):
    # reconstructed pi0 energy of two 3d photon momenta
    return mag(gam1)+mag(gam2)

def pi0InvMass(gam1, gam2):
    # pi0 invariant mass of two 3d photon momenta
    pi0mass2 = pi0Energy(gam1, gam2)**2-mag2(vecAdd(gam1, gam2))
    pi0mass2 = np.where(pi0mass2 >= 0, pi0mass2, 10**6)
    pi0mass = np.sqrt(pi0mass2)
    pi0mass = np.where(pi0mass > 100, -1000, pi0mass)
    return pi0mass

def getPhi(vec1):
    # azimuthal angle of one 3d vector
    return 180/np.pi*np.arctan2(vec1[1], vec1[0])

def getTheta(vec1):
    # polar angle of one 3d vector
    return 180/np.pi*np.arctan2(np.sqrt(vec1[0]*vec1[0]+vec1[1]*vec1[1]), vec1[2])

def getEnergy(vec1, mass):
    # for taken 3d momenta p and mass m, return energy = sqrt(p**2 + m**2)
    return np.sqrt(mag2(vec1)+mass**2)

def readFile(fname):
    #read root using uproot
    ffile = uproot.open(fname)
    tree = ffile["T"]
    return tree
