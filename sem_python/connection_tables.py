## Methods for grid generation and connection tables.
## Assumes standard triangular grid.
import numpy as np
from math import gamma


def gen_EToV(elemsx, elemsy): # n1 is x-axis, n2 is y-axis
    n_elements = elemsx*elemsy*2
    EToV = np.zeros((n_elements, 3), dtype='int')

    for i in range(elemsx):
        for j in range(elemsy):
            idx = (j*2)  +  (i*elemsy*2) # (coloumn id) + (row offset)
            EToV[idx, :]     = [j + (i+1)*(elemsy + 1) , j + i*(elemsy + 1), j + (i+1)*(elemsy + 1) + 1] ## Upper triangle
            EToV[idx + 1, :] = [j + i*(elemsy + 1) + 1 , j + (i+1)*(elemsy + 1) + 1, j + i*(elemsy + 1)] ## Lower triangle
    return EToV


def ti_connect2D(elemsx, elemsy): ## EToF is trivial, so not returned
    n_elements = elemsx*elemsy*2
    EToE = np.zeros((n_elements, 3), dtype='int') 

    for i in range(elemsx):
        for j in range(elemsy):

            idx = (j*2)  +  (i*elemsy*2) # (coloumn id) + (row offset)

            ## EToE
            EToE[idx, :]     = [idx - 1, idx + 1, idx + elemsy*2 + 1] ## Upper triangle
            EToE[idx + 1, :] = [idx + 2, idx, idx - elemsy*2] ## Lower triangle

    ## Self rereference at boundaries:
    for i in range(elemsy):
        # left
        EToE[i*2 + 1, 2] = i*2 + 1
        # right
        EToE[i*2 + elemsy*(elemsx -1)*2, 2] = i*2 + elemsy*(elemsx -1)*2

    for i in range(elemsx):
        # bottom
        EToE[(i+1)*elemsy*2 - 1, 0] = (i+1)*elemsy*2 -1
        # top
        EToE[(i)*elemsy*2, 0]       = (i)*elemsy*2

    return EToE


def algo14(P, N, EToV, EToE):
    ## For starters implemented as in book

    gidx = EToV.shape[0]
    Mp = (P+1)*(P+2)/2
    Mpf = P + 1
    Nfaces = 3

    C = np.zeros((N, gidx*Mp))

    for n in range(N):
        for i in range(Nfaces):
            C[n, i] = EToV[n, i] # Assign lowest global index to vertex points.

            if P > 1:
                if EToE[n, i] >= n: # if neighbor has higher index than current index

                    # Assign global index to points on face:
                    list_of_idx = [Nfaces + (i)*(Mpf-2) + j for j in range(Mpf-2)] # range() should fix conversion to 0 index
                    C[n, list_of_idx] = [gidx + k for k in range(Mpf - 2)] 
                    gidx += (Mpf - 2)

                else: # Else, the points have been assigned an index, so this index is copied and reversed.
                    kc = EToE[n,i]
                    # EToF is trivial for chosen grid, ic = i
                    list_of_idx = [Nfaces + (i)*(Mpf-2) + j for j in range(Mpf-2)]
                    C[n, list_of_idx] = C[kc, [Nfaces + (i)*(Mpf-2) + Mpf-3 - j for j in range(Mpf-2)]]

            # Internal points
            if P > 2:
                list_of_idx = [Nfaces + Nfaces*(Mpf - 2) + k  for k in range(Mp - Nfaces - Nfaces*(Mpf - 2))] # Param in range is amount of already numbered points
                C[n, list_of_idx] = [gidx + k for k in range(Mp - Nfaces - Nfaces*(Mpf - 2))]
                gidx += (Mp - Nfaces - Nfaces*(Mpf - 2))
    return C


def global_assembly(C, N, Mp): # combined algo 15 and 16
    A = np.zeros()
    B = np.zeros()
    for n in range(N):
        # Compute kij(n)

        #Compute mij(n)
        for j in range(Mp):

            jj = C[n,j]
            # Coordinates of point j
            # xj = x[jj]
            # yj = y[jj]

            for i in range(Mp):
                if C[n,j] >= C[n,i]:
                    A[C[n,i], C[n,j]] += 0# kij(n)

                ii = C[n,i]
                B[ii] += 0 # mij(n) * f(xj, yj)
    return A, B


def impose_dirichlet_bc(BoundaryNodesidx, A, B):
    for bn in BoundaryNodesidx:
        B += 0 #-f(bx,by)*A[:, bn]
        A[bn, :] = 0
        A[:, bn] = 0
        A[bn, bn] = 1
    for bn in BoundaryNodesidx:
        B[bn] = 0 # d(bx, by)
    return A, B


## Functions from Hesthaven, Warburton - Nodal Discontinuous Galerkin Methods
# Should probably find out exactly what is needed and the math behind these functions.


def Vandermonde2D(N, r, s):
    V = np.zeros((len(r), (N+1)*(N+2)//2))

    a,b = rstoab(r,s)

    sk = 0
    for i in range(N+1):
        for j in range(N - i + 1):
            V[:, sk] = Simplex2DP(a, b, i, j)
            sk += 1
    return V


def rstoab(r,s):
    Np = len(r)
    a = np.zeros(Np)
    for i in range(Np):
        if s[i] != 1:
            a[i] = 2*(1 + r[i])/(1 - s[i]) - 1
        else:
            a[i] = -1
    return a, s


def Simplex2DP(a, b, i, j):
    # Evaluate 2D orthogonal polynomial on simplex at (a,b) of order (id,jd)
    h1 = JacobiP(a,0,0,id)
    h2 = JacobiP(b,2*i+1,0,j)
    return np.sqrt(2.0)*h1*h2*((1 - b)**i)


def JacobiP(x, alpha, beta, N):
    xp = x # Transformation to right dim
    dims = xp.shape ## Size(xp)
    # if dims(2) == 1: xp = xp.T

    PL = np.zeros((N , len(xp)))

    gamma0 = 2**(alpha + beta + 1)/(alpha + beta + 1)*gamma(alpha + 1)*gamma(beta + 1)/gamma(alpha + beta + 1)
    PL[0, :] = 1.0/np.sqrt(gamma0)
    if N == 0:
        return PL.T
    gamma1 = (alpha + 1)*(beta + 1)/(alpha + beta + 3)*gamma0
    PL[1, :] = ((alpha + beta + 2)*xp/2 + (alpha - beta)/2)/np.sqrt(gamma1)
    if N == 1:
        return PL[N, :].T
    
    aold = 2/(2 + alpha + beta)*np.sqrt((alpha + 1)*(beta +1)/(alpha + beta + 3))

    for i in range(0, N-2):
        h1 = 2*i + alpha + beta
        anew = 2/(h1 + 2)*np.sqrt( (i+1)*(i+1+alpha+beta)*(i+1+alpha)*(i+1+beta)/(h1+1)/(h1+3) )
        bnew = -(alpha*alpha - beta*beta)/(h1*(h1 + 2))
        PL[i+2, :] = ( -aold*PL[i, :] + (xp - bnew)*PL[i+1, :])
        aold = anew

    return PL[N, :].T


def GradJacobiP(x, alpha, beta, N):
    # Derivative of Jacobi polynomial
    dp = np.zeros(len(x), 1)
    if N == 0:
        return dp
    else:
        dp = np.sqrt(N*(N + alpha + beta + 1))*JacobiP(x, alpha + 1, beta + 1, N - 1)
    return dp


def GradSimplex2DP(a,b,id,jd):
    fa = JacobiP(a,0,0,id)
    dfa  = GradJacobiP(a,0,0,id)
    gb = JacobiP(b,2*id+1,0,jd)
    dgb = GradJacobiP(b,2*id+1,0,jd)

    dmodedr = dfa*gb
    if id > 0:
        dmodedr = dmodedr*((0.5*(1-b))**(id))

    dmodeds = dfa*(gb*(0.5*(1+a)))
    if id > 0:
        dmodeds = dmodeds*((0.5*(1-b))**(id))


def GradVandermonde2D(N, r, s): # Need to fix indexing 0-index
    V2Dr = np.zeros((len(r), (N+1)*(N+2)/2))
    V2Ds = np.zeros((len(r), (N+1)*(N+2)/2))

    a,b = rstoab(r,s)
    sk = 0
    for i in range(N+1):
        for j in range(N - i + 1):
            V2Dr[:, sk], V2Ds[:, sk] = GradSimplex2DP(a, b, i, j)
            sk += 1
    return V2Dr, V2Ds


def Dmatrices2D(N,r,s,V): 
    Vr, Vs = GradVandermonde2D(N, r, s)
    Dr = Vr/V
    Ds = Vs/V
    return Dr, Ds




if __name__ == "__main__":
    elemsx = 4
    elemsy = 3
    P = 1
    N = elemsx*elemsy*2
    EToV = gen_EToV(elemsx, elemsy)
    EToE = ti_connect2D(elemsx, elemsy)
    print("EToV:\n", EToV)
    print("EToE:\n", EToE)



