import numpy as np
from coordinate_maps import rstoab
from simplex import Simplex2DP, GradSimplex2DP
from polynomials import JacobiP


def Vandermonde2D(N, r, s):
    V = np.zeros((len(r), int((N+1)*(N+2)//2)))
    a, b = rstoab(r, s)

    sk = 0
    for i in range(N+1):
        for j in range(N - i + 1):
            temp = Simplex2DP(a, b, i, j).T
            temp.shape = V[:, sk].shape
            V[:, sk] = temp
            sk += 1
    return V


def GradVandermonde2D(N, r, s): 
    Mp = int((N+1)*(N+2)/2)
    V2Dr = np.zeros((len(r), Mp))
    V2Ds = np.zeros((len(r), Mp))

    a, b = rstoab(r, s)
    sk = 0
    for i in range(N+1):
        for j in range(N - i + 1):
            t1, t2 = GradSimplex2DP(a, b, i, j)
            V2Dr[:, sk], V2Ds[:, sk] = t1, t2
            sk += 1
    return V2Dr, V2Ds


def Vandermonde1D(N, r):
    V1D = np.zeros((len(r), N + 1))
    for j in range(N + 1):
        temp = JacobiP(r, 0, 0, j)
        V1D[:, j] = temp.T
    return V1D
