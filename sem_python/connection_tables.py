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


def gen_node_coordinates(elemsx, elemsy, min_x=-1, max_x=1, min_y=-1, max_y=1):
    n_nodes_x = elemsx + 1
    n_nodes_y = elemsy + 1
    x = np.zeros((n_nodes_x * n_nodes_y, 1))
    y = np.zeros((n_nodes_x * n_nodes_y, 1))

    for i in range(n_nodes_x):
        for j in range(n_nodes_y):
            idx = j + i*n_nodes_y
            x[idx] = min_x + (max_x - min_x) * i / elemsx
            y[idx] = min_y + (max_y - min_y) * j / elemsy
    return x, y


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
    Mp = int((P+1)*(P+2)/2)
    Mpf = P + 1
    Nfaces = 3

    C = np.zeros((N, Mp))

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


def global_assembly(C, N, P, x, y, q): # combined algo 15 and 16
    Mp =  int((P+1)*(P+2)/2)
    A = np.zeros((N,N))
    B = np.zeros(N)

    r,s = xytors(x,y)

    # Compute kij(n) (4.46) - precompute 
    V = Vandermonde2D(P, r, s)
    Dr, Ds = Dmatrices2D(N,r,s,V)
    rx, sx, ry, sy, J = geometric_factors(x, y, Dr, Ds)
    M = (V @ V.T).inv() # Mass matrix

    tmp1 = (rx*Dr + sx*Ds).T @ M @ (rx*Dr + sx*Ds)
    tmp2 = (ry*Dr + sy*Ds).T @ M @ (ry*Dr + sy*Ds)
    tmp = tmp1 + tmp2
    

    for n in range(N):
        # Compute kij(n) (4.46) - value
        kij = J(n)*tmp
        


        mij = J(n)*M*q(n)
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
    V = np.zeros((len(r), int((N+1)*(N+2)//2)))
    print("size of V:", V.shape)

    a,b = rstoab(r,s)

    sk = 0
    for i in range(N+1):
        for j in range(N - i + 1):
            print("------------- printing V[:, sk] ------------")
            print(V[:, sk])
            temp = Simplex2DP(a, b, i, j)
            print("------------- printing Simplex2DP output ------------")
            print(temp.T)
            V[:, sk] = temp.T
            sk += 1
    return V


def xytors(x,y):
    L1 = (np.sqrt(3)*y + 1)/3
    L2 = (-3*x - np.sqrt(3)*y + 2)/6
    L3 = (3*x - np.sqrt(3)*y + 2)/6
    r = -L2 + L3 - L1
    s = -L2 - L3 + L1
    return r, s


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
    h1 = JacobiP(a, 0, 0, i)
    h2 = JacobiP(b, 2*i+1, 0, j)
    return np.sqrt(2.0)*h1*h2*((1 - b)**i)


def JacobiP(x, alpha, beta, N):
    xp = x # Transformation to right dim
    dims = xp.shape ## Size(xp)
    # if dims(2) == 1: xp = xp.T
    print("------------- printing xp ------------")
    print(xp)
    print(dims)
    print(len(xp))

    PL = np.zeros((N+1 , len(xp)))
    print("size of PL:", PL.shape)

    gamma0 = 2**(alpha + beta + 1)/(alpha + beta + 1)*gamma(alpha + 1)*gamma(beta + 1)/gamma(alpha + beta + 1)
    PL[0, :] = 1.0/np.sqrt(gamma0)
    if N == 0:
        return PL.T
    gamma1 = (alpha + 1)*(beta + 1)/(alpha + beta + 3)*gamma0
    PL[1, :] = (((alpha + beta + 2)*xp/2 + (alpha - beta)/2)/np.sqrt(gamma1)).T # Transposed to fix dims
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


def GradSimplex2DP(a,b,id,jd): # Check functionality, check ids
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

    tmp = dgb*(0.5*(1 - b))**id ## Make sure this is an elementwise multiplication
    if id > 0:
        tmp = tmp -0.5*id*gb*((0.5*(1 - b))**(id - 1))
    dmodeds = dmodeds + fa*tmp # elementwise mult again

    dmodedr = 2**(id + 0.5)*dmodedr
    dmodeds = 2**(id + 0.5)*dmodeds
    return dmodedr, dmodeds
    


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


def geometric_factors(x, y, Dr, Ds):
    xr = Dr*x
    xs = Ds*x
    yr = Dr*y
    ys = Ds*y
    J = xr*ys - xs*yr
    rx =  ys/J
    sx = -yr/J
    ry = -xs/J
    sy =  xr/J
    return rx, sx, ry, sy, J


def Warpfactor(N, rout, Tol=1e-10):
    LGLr = JacobiGL(0,0,N)
    req = np.linspace(-1,1,N+1)

    Veq = Vandermonde1D(N, LGLr)

    Nr = len(rout)
    Pmat = np.zeros((N + 1, Nr))
    for i in range(N + 1):
        Pmat[i, :] = JacobiP(rout, 0, 0, i)

    Lmat = np.linalg.solve(Veq.T, Pmat)

    warp = Lmat.T @ (LGLr - req)
    zerof = (abs(rout) < Tol)               ## Boolean array ? does it work?
    sf = 1.0 - (zerof*rout)**2
    warp = warp/sf + warp * (zerof - 1)
    return warp


def JacobiGL(alpha, beta, N):
    x = np.zeros((N+1, 1))
    if N == 1:
        x[0] = -1
        x[1] = 1
        return x
    xint, w = JacobiGQ(alpha + 1, beta + 1, N - 2)
    x[0] = -1
    x[1:N] = xint
    x[N] = 1
    return x

def Vandermonde1D(N, r):
    V1D = np.zeros((len(r), N + 1))
    for j in range(N + 1):
        V1D[:, j] = JacobiP(r, 0, 0, j)
    return V1D


def JacobiGQ(alpha, beta, N): ##  Needs to verify correctness
    x = np.zeros((N + 1, 1))
    w = np.zeros((N + 1, 1))
    if N == 0:
        x[0] = (alpha - beta)/(alpha + beta + 2)
        w[0] = 2
        return x, w

    J = np.zeros((N + 1, N + 1))
    h1 = 2*np.arange(0, N + 1) + alpha + beta ## Np arrange?????
    
    J = np.diag(-1/2*(alpha**2 - beta**2)/(h1 + 2)/h1) + \
        np.diag(2/(h1[0:N] +2) *np.sqrt( np.arange(1, N +1)*(np.arange(1, N +1) + alpha + beta)* \
        (np.arange(1, N +1) + alpha)*(np.arange(1, N +1) + beta)/(h1[0:N] +1)/(h1[0:N] +3) ), 1)
    J = J + J.T
    V, D = np.linalg.eig(J)
    x = np.diag(D)
    w = (V[0,:].T)**2 * 2**(alpha + beta + 1)/(alpha + beta + 1)*gamma(alpha + 1)*gamma(beta + 1)/gamma(alpha + beta + 1)
    return x, w


def Nodes2D(N):
    ## Generates x and y vectors. See p 179.
    alpopt = [0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999, 1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258]
    Np = int((N+1)*(N+2)/2)

    if N <= 14:
        alpha = alpopt[N]
    else:
        alpha = 5/3
    
    L1 = np.zeros((Np, 1))
    L2 = np.zeros((Np, 1))
    L3 = np.zeros((Np, 1))

    sk = 0
    for n in range(N+1):
        for m in range(N - n + 1): # subtract one less due to diff indexing of n
            L1[sk] = n / N
            L3[sk] = m / N
            sk += 1
    L2 = 1.0 - L1 - L3
    x = - L2 + L3
    y = (-L2 - L3 + 2*L1)/np.sqrt(3)

    blend1 = 4*L2*L3
    blend2 = 4*L1*L3
    blend3 = 4*L1*L2
    warpf1 = Warpfactor(N, L3-L2)
    warpf2 = Warpfactor(N, L1-L3)
    warpf3 = Warpfactor(N, L2-L1)

    warp1 = blend1*warpf1*(1 + (alpha*L1)**2)
    warp2 = blend2*warpf2*(1 + (alpha*L2)**2)
    warp3 = blend3*warpf3*(1 + (alpha*L3)**2)

    x = x + 1*warp1 + np.cos(2*np.pi/3)*warp2 + np.cos(4*np.pi/3)*warp3
    y = y + 0*warp1 + np.sin(2*np.pi/3)*warp2 + np.sin(4*np.pi/3)*warp3
    return x, y


if __name__ == "__main__":
    elemsx = 4
    elemsy = 3
    P = 1
    N = elemsx*elemsy*2
    EToV = gen_EToV(elemsx, elemsy)
    EToE = ti_connect2D(elemsx, elemsy)

    x,y = Nodes2D(P) # Local coords
    r,s = xytors(x,y)
    v1 = EToV[:, 0].T
    v2 = EToV[:, 1].T
    v3 = EToV[:, 2].T

    vx,vy = gen_node_coordinates(elemsx, elemsy)

    # x and y will have size (Mp x N)?
    x = 0.5*(-(r+s) @ vx[v1] + (1 + r) @ vx[v2] + (1 + s) @ vx[v3]) 
    y = 0.5*(-(r+s) @ vy[v1] + (1 + r) @ vy[v2] + (1 + s) @ vy[v3])

    tol = 0.2/P**2
    Mp = len(s)

    ## Implement reordering of nodes here




    print("EToV:\n", EToV)
    print("EToE:\n", EToE)
    print("Node coordinates (x):\n", x)
    print("Node coordinates (y):\n", y)
    C = algo14(P, N, EToV, EToE)
    print("Connection table C:\n", C)
    print(C.shape)


    q = lambda n: 1 # Dummy q

    A, B = global_assembly(C, N, P, x, y, q)
    print("Global matrix A:\n", A)
    print("Global vector B:\n", B)


