from __future__ import division
import sys, time, math, matplotlib
matplotlib.use('PS')
import numpy as np
from numpy import linalg as LA
from scipy import spatial
from scipy.sparse import csr_matrix
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.style as style
#from mpi4py import MPI
#import eigen
style.use('seaborn-whitegrid')

############################################################
################ Parameters Of the Problem #################
############################################################
a = 2.46
i = 1 # to set the rotation angle
#theta = 0
#i = int(sys.argv[1])
theta = 2*np.arcsin(1/(2*np.sqrt(3*i**2+3*i+1)))

### TABLE(1) of SF&EK ###
t1 = -2.8922
t2 = 0.2425
t3 = -0.2656
t4 = 0.0235
t5 = 0.0524
t6 = -0.0209
t7 = -0.0148
t8 = -0.0211

l_0 = 0.3155
l_3 = -0.0688
l_6 = -0.0083
s_0 = 1.7543
s_3 = 3.4692
s_6 = 2.8764
x_3 = 0.5212
x_6 = 1.5206
k_0 = 2.0010
k_6 = 1.5731

step = 20
k_MODE = "along_w_dl"

#INPUT_DIR = '/home/eric/Downloads/TB_tBG/input/'
#OUTPUT_DIR = '/home/eric/Downloads/TB_tBG/output'
INPUT_DIR = '/Users/chirueipan/gSuite/graphene/TB_tBG/input/'
OUTPUT_DIR = '/Users/chirueipan/gSuite/graphene/TB_tBG/output'

############################################################
############################################################
############################################################

#rotate a vector by theta angle counterclockwise
def rotate(inVector,theta):
    RotationOperator = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    outVector = np.dot(RotationOperator,np.transpose(inVector))
    return np.transpose(outVector)

#a = [1, 0]
#b = rotate(a, np.pi/3)
#print(b)

#normalize a vector
def normalize(inVector):
    norm = LA.norm(inVector)
    if norm == 0:
        return inVector
    else:
        return inVector/norm
#calculate angle (degree)  between two vectors
def calcAngleOfTwoVectors(inVec1, inVec2):
    dot_product = np.dot(inVec1, inVec2)
    norm_inVec1 = LA.norm(inVec1)
    norm_inVec2 = LA.norm(inVec2)
    if norm_inVec1 == 0 or norm_inVec2 == 0:
        theta = 0
    else:    
        ratio = dot_product/norm_inVec1/norm_inVec2
        if ratio < -1.0:
            ratio = -1
        if ratio > 1.0:
            ratio = 1.0
        theta = np.arccos(ratio)
    return theta

#construct real space primitive lattice basis vectors
def construct_rPrimLatt_basis(LatticeConstant):
    rPrimLattVec1 = LatticeConstant*np.array([1/2,np.sqrt(3)/2])  #connecting A to A sublattice
    rPrimLattVec2 = LatticeConstant*np.array([-1/2,np.sqrt(3)/2]) #connecting A to A sublattice
    rOtherSublattice = (rPrimLattVec1 + rPrimLattVec2)/3                    #connecting A to B sublattice
    return rPrimLattVec1, rPrimLattVec2, rOtherSublattice

#construct real space super lattice basis vectors
def construct_rSuperLatt_basis(rPrimLattVec1, rPrimLattVec2):
    ### Superlattice unit cell, EQN(1) in PM&MK ###
    rSuperLattVec1 = i*rPrimLattVec1 + (i+1)*rPrimLattVec2         #connecting AA to AA region
    rSuperLattVec2 = -(i+1)*rPrimLattVec1 + (2*i+1)*rPrimLattVec2  #connecting AA to AA region
    rABRegion = (rSuperLattVec1 + rSuperLattVec2)/3             #connecting AA to AB region
    return rSuperLattVec1, rSuperLattVec2, rABRegion

#construct reciprocal lattice basis vectors
def construct_kSpace_basis(rLattVec1, rLattVec2):
    ### Reciprocal lattice, EQN(5.3) in Ashcroft ###
    ### Reciprocal superlattice, EQN(5.3) in Ashcroft ###
    kRecLattVec1 = np.copy(rLattVec2)
    kRecLattVec2 = np.copy(rLattVec1)
    kRecLattVec1[0], kRecLattVec1[1] = rLattVec2[1], -rLattVec2[0] 
    kRecLattVec2[0], kRecLattVec2[1] = rLattVec1[1], -rLattVec1[0]  
    kRecLattVec1  = 2*np.pi*kRecLattVec1 /(np.dot(rLattVec1, kRecLattVec1)) 
    kRecLattVec2  = 2*np.pi*kRecLattVec2 /(np.dot(rLattVec2, kRecLattVec2))
    return kRecLattVec1, kRecLattVec2

def Wigner_cell(some_r_super): #enclosing the 1st BZ of moire cell
    point = some_r_super
    for n in range(6):
        point = np.vstack((point,rotate(some_r_super,(n+1)*np.pi/3)))
    poly_verts = point
    path1 = Path(poly_verts)
    return path1

def Brillouin_zone(Gamma_point,M_point,K_point):
    dM = Gamma_point - M_point
    dK = K_point - M_point
    dM = dM/step/LA.norm(dM)*LA.norm(dK)
    dK = dK/step
    region1 = np.array([M_point-dM*10e-5-dK*10e-5, K_point, Gamma_point, M_point-dM*10e-5-dK*10e-5])
    path1 = Path(region1)
    point1 = K_point
    point2 = -K_point+2*M_point
    point3 = -2*K_point+2*M_point
    point4 = -point1
    point5 = -point2
    point6 = -point3
    region2 = np.vstack([point1,point2,point3,point4,point5,point6,point1])
    path2 = Path(region2)
    return path1, path2

def construct_kVectors(k_MODE):
    if k_MODE == "grid":
        tmp1 = []
        for j in range(step*2):
            n = step*2 - j
            for m in range(n):
                tmp1.append((j,m))
        tmp1 = np.array(tmp1)
        grid_1 = list_vec(tmp1, dK, dM)
        grid_1 += M
        ma = Brillouin1.contains_points(grid_1, radius=10e-10)
        k_vectors = grid_1[np.array(ma)]
        k_vectors = np.vstack((k_vectors, Ga))
        k_vectors = np.vstack((k_vectors, K))
#        if (rank==0):
#            print("Size of k_vectors",len(k_vectors))
    elif k_MODE == "along_w_step":
        dGM = (M-Ga)/step
        dMK = (K-M)/step
        dKG = (Ga-K)/step
        GM = np.array([Ga+dGM*i_step for i_step in range(step+1)])
        MK = np.array([M+dMK*i_step for i_step in range(step+1)])
        KG = np.array([K+dKG*i_step for i_step in range(step+1)])
        k_vectors, indices = np.unique(np.vstack([GM,MK,KG]),axis=0,return_inverse=True)
        k_vectors = k_vectors[indices]
#        if (rank==0):
#            print("Size of k_vectors",len(k_vectors))
    elif k_MODE == "along_w_dl":
        dl = LA.norm((M-Ga)/step)/2; print('dl =',dl)
        dGM = normalize(M-Ga)*dl
        dMK = normalize(K-M)*dl
        dKG = normalize(Ga-K)*dl
        GM = np.array([Ga+dGM*i_step for i_step in range(np.int(LA.norm(M-Ga)/dl)+1)])
        MK = np.array([M+dMK*i_step for i_step in range(np.int(LA.norm(K-M)/dl)+1)])
        KG = np.array([K+dKG*i_step for i_step in range(np.int(LA.norm(Ga-K)/dl)+1)])
        k_vectors, indices = np.unique(np.vstack([GM,MK,KG]),axis=0,return_inverse=True)
        k_vectors = k_vectors[indices]
#        if (rank==0):
#            print("Size of k_vectors",len(k_vectors))
    else:
        print("k_MODE not defined")
    np.save('%s/k_vector'%(OUTPUT_DIR), k_vectors)
    return k_vectors




#======calculate intralayer hopping parameters==========
def calcIntraHopping(coordExtSuperLatt, coordSuperLatt):
    IntraHopVec = coordExtSuperLatt - coordSuperLatt; #print(e)
    IntraHopDist = LA.norm(IntraHopVec)
    IntraHopStrength = 0
    if (abs(IntraHopDist   - 0) <= 10e-12):
        IntraHopStrength = 0
    elif (abs(IntraHopDist - NearestDist_1st) <= 10e-12):
        IntraHopStrength = t1
    elif (abs(IntraHopDist - NearestDist_2nd) <= 10e-12):
        IntraHopStrength = t2
    elif (abs(IntraHopDist - NearestDist_3rd) <= 10e-12):
        IntraHopStrength = t3
    elif (abs(IntraHopDist - NearestDist_4th) <= 10e-12):
        IntraHopStrength = t4
    elif (abs(IntraHopDist - NearestDist_5th) <= 10e-12):
        IntraHopStrength = t5
    elif (abs(IntraHopDist - NearestDist_6th) <= 10e-12):
        IntraHopStrength = t6
    elif (abs(IntraHopDist - NearestDist_7th) <= 10e-12):
        IntraHopStrength = t7
    elif (abs(IntraHopDist - NearestDist_8th) <= 10e-12):
        IntraHopStrength = t8
    else:
        IntraHopStrength = 0
    return IntraHopStrength, IntraHopVec

#======calculate interlayer hopping parameters==========
### Fitting functions for Vi(r), EQN(3) in SF&EK ### For interlayer hoppings
def V0(r):
    return l_0 * np.exp(-s_0*(r/a)**2)*np.cos(k_0*(r/a))
def V3(r):
    return l_3 * (r/a)**2*np.exp( -s_3*(r/a-x_3)**2)
def V6(r):
    return l_6 * np.exp(-s_6*(r/a-x_6)**2)*np.sin(k_6*(r/a))
def t(r, theta1, theta2):
    return V0(r) + V3(r)*(np.cos(3*theta1)+np.cos(3*theta2)) + V6(r)*(np.cos(6*theta1)+np.cos(6*theta2))
def calcInterHopping(coordExtSuperLatt, coordSuperLatt):
    InterHopProjVec = coordExtSuperLatt - coordSuperLatt; #print(e)
    InterHopProjDist = LA.norm(InterHopProjVec)
    theta1 = calcAngleOfTwoVectors(InterHopProjVec, rOtherSublattice)
    theta2 = calcAngleOfTwoVectors(InterHopProjVec, rRotOtherSublattice)
    if (theta1 > np.pi/2):
        theta1 = np.pi - theta1
    if (theta2 > np.pi/2):
        theta2 = np.pi - theta2 
    if (np.isnan(theta1)):
        theta1 = 0
    if (np.isnan(theta2)):
        theta2 = 0
    ### Hopping parameter given by the superposition, EQN(2) in SF&EK, used in function "hop_inter" ###
    InterHopStrength = t(InterHopProjDist, theta1, theta2)
    return InterHopStrength, InterHopProjVec

#======assign matrix element==========
def assignIntraElement(ExtSuperLatt, SuperLatt, orderExtSuperLatt, orderSuperLatt, subExtSuperLatt, subSuperLatt, SearchRange, row, col, stre, dic):
    tree = spatial.cKDTree(ExtSuperLatt)
    index = tree.query_ball_point(SuperLatt, LA.norm(SearchRange)+10e-12) #search neighbor with distance LimitNeighborDist for every point in Superlattice
    for indexSuperLatt in range(len(SuperLatt)):  #run over the whole superlattice and calculate the coupling strength for each atom
        term = len(index[indexSuperLatt])
        for j in range(term):
            indexExtSuperLatt = index[indexSuperLatt][j]
            coordExtSuperLatt = ExtSuperLatt[indexExtSuperLatt]
            coordSuperLatt = SuperLatt[indexSuperLatt]
            t, e = calcIntraHopping(coordExtSuperLatt, coordSuperLatt)
            row.append(orderSuperLatt[indexSuperLatt])  #save the order of superlattice (m)
            col.append(orderExtSuperLatt[indexExtSuperLatt])  #save the order of extendedsuperlattice (n)
            stre.append(t) #save the coupling between m and n atoms thus is the matrix element of Hmn
            dic.append(e)
    return index

def assignInterElement(ExtSuperLatt, SuperLatt, orderExtSuperLatt, orderSuperLatt, subExtSuperLatt, subSuperLatt, SearchRange, row, col, stre, dic):
    tree = spatial.cKDTree(ExtSuperLatt)
    index = tree.query_ball_point(SuperLatt, LA.norm(SearchRange)+10e-12)
    for indexSuperLatt in range(len(SuperLatt)):
        term = len(index[indexSuperLatt])
        for j in range(term):
            indexExtSuperLatt = index[indexSuperLatt][j]
            coordExtSuperLatt = ExtSuperLatt[indexExtSuperLatt]
            coordSuperLatt = SuperLatt[indexSuperLatt]
            t, e = calcInterHopping(coordExtSuperLatt, coordSuperLatt)
            if orderExtSuperLatt[indexExtSuperLatt] == 0 and orderSuperLatt[indexSuperLatt] == 15:
                print("e=",e)
                print("interHoppintStrength", t)
            row.append(orderSuperLatt[indexSuperLatt])
            col.append(orderExtSuperLatt[indexExtSuperLatt])
            stre.append(t)
            dic.append(e)
    return index


############################################################
################ Main Code #################################
############################################################

rPrimLattVec1,  rPrimLattVec2,  rOtherSublattice       = construct_rPrimLatt_basis(a)
rRotPrimLattVec1 = rotate(rPrimLattVec1, theta)
rRotPrimLattVec2 = rotate(rPrimLattVec2, theta)
rRotOtherSublattice = (rRotPrimLattVec1 + rRotPrimLattVec2)/3
rSuperLattVec1, rSuperLattVec2, rABRegion = construct_rSuperLatt_basis(rPrimLattVec1, rPrimLattVec2)

kRecPrimLattVec1,    kRecPrimLattVec2    = construct_kSpace_basis(rPrimLattVec1,    rPrimLattVec2)
kRecRotPrimLattVec1, kRecRotPrimLattVec2 = construct_kSpace_basis(rRotPrimLattVec1, rRotPrimLattVec2)
kRecSuperLattVec1,   kRecSuperLattVec2   = construct_kSpace_basis(rSuperLattVec1,   rSuperLattVec2)

LimitNeighborDist = rOtherSublattice
NearestDist_1st = LA.norm(rOtherSublattice)
NearestDist_2nd = LA.norm(rPrimLattVec1)
NearestDist_3rd = LA.norm(rPrimLattVec1 + rPrimLattVec2 - rOtherSublattice) 
NearestDist_4th = LA.norm(2*rPrimLattVec1 - rOtherSublattice)
NearestDist_5th = LA.norm(rPrimLattVec1 + rPrimLattVec2)
NearestDist_6th = LA.norm(2*rPrimLattVec1)
NearestDist_7th = LA.norm(2*rPrimLattVec1 - rPrimLattVec2 + rOtherSublattice)
NearestDist_8th = LA.norm(rPrimLattVec1 + rPrimLattVec2 + rOtherSublattice)
print("Unrotated layer primitive lattice vector1 (V1)=", rPrimLattVec1)
print("Unrotated layer primitive lattice vector2 (V2)=", rPrimLattVec2)
#print("*Angle between V1 and V2=",   calcAngleOfTwoVectors(rPrimLattVec1, rPrimLattVec2)) 
print("Rotated layer primitive lattice vector1 (V1')=", rRotPrimLattVec1)
print("Rotated layer primitive lattice vector2 (V2')=", rRotPrimLattVec2)
#print("*Angle between V1' and V2'=", calcAngleOfTwoVectors(rRotPrimLattVec1, rRotPrimLattVec2)) 
#print("*Angle between V1 and V1'=",  calcAngleOfTwoVectors(rPrimLattVec1, rRotPrimLattVec1)) 
#print("*Angle between V2 and V2'=",  calcAngleOfTwoVectors(rPrimLattVec2, rRotPrimLattVec2)) 
#print("Super lattice vector1 (V3)=", rSuperLattVec1)
#print("Super lattice vector2 (V4)=", rSuperLattVec2)
#print("*Angle between V3 and V4=", calcAngleOfTwoVectors(rSuperLattVec1, rSuperLattVec2)) 
#print("Unrotated layer reciprocal primitive lattice vector1 (recV1)=",  kRecPrimLattVec1)
#print("Unrotated layer reciprocal primitive lattice vector2 (recV2)=",  kRecPrimLattVec2)
#print("*Angle between recV1 and recV2=",   calcAngleOfTwoVectors(kRecPrimLattVec1, kRecPrimLattVec2)) 
#print("Unrotated layer reciprocal primitive lattice vector1 (recV1')=", kRecRotPrimLattVec1)
#print("Unrotated layer reciprocal primitive lattice vector2 (recV2')=", kRecRotPrimLattVec2)
#print("*Angle between recV1' and recV2'=", calcAngleOfTwoVectors(kRecRotPrimLattVec1, kRecRotPrimLattVec2)) 
#print("*Angle between recV1 and recV1'=",  calcAngleOfTwoVectors(kRecPrimLattVec1, kRecRotPrimLattVec1)) 
#print("*Angle between recV2 and recV2'=",  calcAngleOfTwoVectors(kRecPrimLattVec2, kRecRotPrimLattVec2)) 
#print("V1 dot recV1=", np.dot(rPrimLattVec1, kRecPrimLattVec1))
#print("V1 dot recV2=", np.dot(rPrimLattVec1, kRecPrimLattVec2))
#print("V2 dot recV1=", np.dot(rPrimLattVec2, kRecPrimLattVec1))
#print("V2 dot recV2=", np.dot(rPrimLattVec2, kRecPrimLattVec2))
#print("V1' dot recV1'=", np.dot(rRotPrimLattVec1, kRecRotPrimLattVec1))
#print("V1' dot recV2'=", np.dot(rRotPrimLattVec1, kRecRotPrimLattVec2))
#print("V2' dot recV1'=", np.dot(rRotPrimLattVec2, kRecRotPrimLattVec1))
#print("V2' dot recV2'=", np.dot(rRotPrimLattVec2, kRecRotPrimLattVec2))




Ga = 0*kRecSuperLattVec1
M = (kRecSuperLattVec1+kRecSuperLattVec2)/2
K = (2*M+kRecSuperLattVec1)/3
print("Ga=",Ga)
print("M=",M)
print("K=",K)

Wigner1 = Wigner_cell(rABRegion)
Brillouin1, Brillouin2 = Brillouin_zone(Ga, M, K)
_g_1 = construct_kVectors(k_MODE) #information of Brillouin1 in used here

### Load in array of superlattice  ###
SuperLatt1 = np.load('%s/AtomicPosition_layer1.npy'%INPUT_DIR)
SuperLatt2 = np.load('%s/AtomicPosition_layer2.npy'%INPUT_DIR)
orderSuperLatt1 = np.load('%s/Order_layer1.npy'%INPUT_DIR)
orderSuperLatt2 = np.load('%s/Order_layer2.npy'%INPUT_DIR)
subSuperLatt1 = np.load('%s/Sublattice_layer1.npy'%INPUT_DIR)
subSuperLatt2 = np.load('%s/Sublattice_layer2.npy'%INPUT_DIR)

ExtSuperLatt1 = np.load('%s/AtomicPosition_extendedlayer1.npy'%INPUT_DIR)
ExtSuperLatt2 = np.load('%s/AtomicPosition_extendedlayer2.npy'%INPUT_DIR)
orderExtSuperLatt1 = np.load('%s/Order_extendedlayer1.npy'%INPUT_DIR)
orderExtSuperLatt2 = np.load('%s/Order_extendedlayer2.npy'%INPUT_DIR)
subExtSuperLatt1 = np.load('%s/Sublattice_extendedlayer1.npy'%INPUT_DIR)
subExtSuperLatt2 = np.load('%s/Sublattice_extendedlayer2.npy'%INPUT_DIR)

#print(subSuperlattice1[0])


row11 = []
col11 = []
stre11 = []
dic11 = []

row22 = []
col22 = []
stre22 = []
dic22 = []

row12 = []
col12 = []
stre12 = []
dic12 = []

row21 = []
col21 = []
stre21 = []
dic21 = []


### Write in the value of "row", "col", "stre", "dic" 
## The value of them are inserted when the function is called.
IntraSearchRange = LimitNeighborDist 
InterSearchRange = NearestDist_2nd 
#InterSearchRange = NearestDist_5th
index11 = assignIntraElement(ExtSuperLatt1, SuperLatt1, orderExtSuperLatt1, orderSuperLatt1, IntraSearchRange, subExtSuperLatt1, subSuperLatt1, row11, col11, stre11, dic11)
index22 = assignIntraElement(ExtSuperLatt2, SuperLatt2, orderExtSuperLatt2, orderSuperLatt2, IntraSearchRange, subExtSuperLatt2, subSuperLatt2, row22, col22, stre22, dic22)
index12 = assignInterElement(ExtSuperLatt2, SuperLatt1, orderExtSuperLatt2, orderSuperLatt1, InterSearchRange, subExtSuperLatt2, subSuperLatt1, row12, col12, stre12, dic12)
index21 = assignInterElement(ExtSuperLatt1, SuperLatt2, orderExtSuperLatt1, orderSuperLatt2, InterSearchRange, subExtSuperLatt1, subSuperLatt2, row21, col21, stre21, dic21)
print("here")


row11file = open("row11.dat", "w")
col11file = open("col11.dat", "w")
stre11file = open("stre11.dat", "w")
dic11file = open("dic11.dat", "w")
for i in range(len(row11)):
    row11file.write(str(row11[i])+"\n")
    col11file.write(str(col11[i])+"\n")
    stre11file.write(str(stre11[i])+"\n")
    dic11file.write(str(dic11[i])+"\n")
row11file.close
col11file.close
stre11file.close
dic11file.close

row22file  = open("row22.dat", "w")
col22file  = open("col22.dat", "w")
stre22file = open("stre22.dat", "w")
dic22file  = open("dic22.dat", "w")
for i in range(len(row22)):
    row22file.write(str(row22[i])+"\n")
    col22file.write(str(col22[i])+"\n")
    stre22file.write(str(stre22[i])+"\n")
    dic22file.write(str(dic22[i])+"\n")
row22file.close
col22file.close
stre22file.close
dic22file.close

row12file  = open("row12.dat", "w")
col12file  = open("col12.dat", "w")
stre12file = open("stre12.dat", "w")
dic12file  = open("dic12.dat", "w")
for i in range(len(row12)):
    row12file.write(str(row12[i])+"\n")
    col12file.write(str(col12[i])+"\n")
    stre12file.write(str(stre12[i])+"\n")
    dic12file.write(str(dic12[i])+"\n")
row12file.close
col12file.close
stre12file.close
dic12file.close

row21file = open("row21.dat", "w")
col21file = open("col21.dat", "w")
stre21file = open("stre21.dat", "w")
dic21file = open("dic21.dat", "w")
for i in range(len(row21)):
    row21file.write(str(row21[i])+"\n")
    col21file.write(str(col21[i])+"\n")
    stre21file.write(str(stre21[i])+"\n")
    dic21file.write(str(dic21[i])+"\n")
row21file.close
col21file.close
stre21file.close
stre21file.close
dic12file.close
dic12file.close

print("there")
for k in _g_1:
    data11 = np.array(stre11)*np.exp(-1j*np.inner(np.array(dic11),k))
    data22 = np.array(stre22)*np.exp(-1j*np.inner(np.array(dic22),k))
    data12 = np.array(stre12)*np.exp(-1j*np.inner(np.array(dic12),k))
    data21 = np.array(stre21)*np.exp(-1j*np.inner(np.array(dic21),k))
    all_row = row11 + row22 + row12 + row21
    all_col = col11 + col22 + col12 + col21
    all_data = np.concatenate((data11, data22, data12, data21), axis=None)
    ham = csr_matrix((all_data, (all_row, all_col)), shape=(SuperLatt1.size, SuperLatt1.size))

print(ham[0])


