## Methods for grid generation and connection tables.
## Assumes standard triangular grid.
import numpy as np
from polynomials import *


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
    return C.astype('int')


def global_assembly(C, N, P, x, y, q, V, Dr, Ds): # combined algo 15 and 16
    Mp =  int((P+1)*(P+2)/2)
    A = np.zeros((N*Mp,N*Mp))
    B = np.zeros(N*Mp,)

    r,s = xytors(x,y)
    print("r in global assembly:", r)
    print("s in global assembly:", s)

    # Compute kij(n) (4.46) - precompute 
    # V = Vandermonde2D(P, r, s)
    # Dr, Ds = Dmatrices2D(N,r,s,V)
    rx, sx, ry, sy, J = geometric_factors(x, y, Dr, Ds)
    M = np.linalg.inv(V @ V.T) # Mass matrix

    for n in range(N):
        # Compute kij(n) (4.46) - value
        print("shape rx" , rx.shape)
        Dx = np.diag(rx[:, n]) @ Dr + np.diag(sx[:, n]) @ Ds
        Dy = np.diag(ry[:, n]) @ Dr + np.diag(sy[:, n]) @ Ds
        print("Dx:", Dx)
        tmp1 = Dx.T @ M @ Dx
        tmp2 = Dy.T @ M @ Dy
        tmp = tmp1 + tmp2
        #print("tmp print", tmp)
        kij = J.T[n]*tmp
        #print("kij:", kij)
        print(np.diag(J[:, n]) @ tmp)

        mij = np.diag(J[:, n]) @ M*q(n)
        for j in range(Mp):

            jj = C[n,j]
            # Coordinates of point j
            # xj = x[jj]
            # yj = y[jj]

            for i in range(Mp):
                if C[n,j] >= C[n,i]:
                    A[C[n,i], C[n,j]] += kij[i,j]

                ii = C[n,i]
                print("B shape" , B.shape)
                print(ii)
                B[ii] +=  mij[i,j] #* f(xj, yj)
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
            # print("------------- printing V[:, sk] ------------")
            # print(V[:, sk])
            # print("calling params:", "a:", a, "b:", b, "i:", i, "j:", j)
            temp = Simplex2DP(a, b, i, j).T
            # print("------------- printing Simplex2DP output ------------")
            temp.shape = V[:, sk].shape
            V[:, sk] = temp
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
    a = np.zeros((Np,))
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
    return np.sqrt(2.0)*h1.T*h2.T*((1 - b)**i)


def GradSimplex2DP(a,b,id,jd): # Check functionality, check ids
    fa = JacobiP(a,0,0,id)
    dfa  = GradJacobiP(a,0,0,id)
    gb = JacobiP(b,2*id+1,0,jd)
    dgb = GradJacobiP(b,2*id+1,0,jd)
    # print("Calling params in GradSimplex2DP:")
    # print("a:", a)
    # print("b:", b)
    # print("id:", id)
    # print("jd:", jd)

    dmodedr = dfa*gb
    # print()
    # print("Initial dmodedr:", dmodedr)
    # print(dmodedr.shape)
    # print("a", a.shape)
    # print("dfa" , dfa.shape)
    # print("gb", gb.shape)
    if id > 0:
        dmodedr = dmodedr*((0.5*(1-b))**(id - 1))

    dmodeds = dfa*(gb*(0.5*(1+a)))

    if id > 0:
        dmodeds = dmodeds*((0.5*(1-b))**(id - 1))

    tmp = dgb*(0.5*(1 - b))**id ## Make sure this is an elementwise multiplication
    if id > 0:
        tmp = tmp -0.5*id*gb*((0.5*(1 - b))**(id - 1))
    dmodeds = dmodeds + fa*tmp # elementwise mult again

    dmodedr = 2**(id + 0.5)*dmodedr
    dmodeds = 2**(id + 0.5)*dmodeds
    # print("Final dmodedr:", dmodedr)
    # print("Final dmodeds:", dmodeds)
    return dmodedr, dmodeds
    


def GradVandermonde2D(N, r, s): # Need to fix indexing 0-index
    Mp = int((N+1)*(N+2)/2)
    V2Dr = np.zeros((len(r), Mp))
    V2Ds = np.zeros((len(r), Mp))

    a,b = rstoab(r,s)
    # print("Shapes:")
    # print(a.shape)
    # print(b.shape)
    sk = 0
    for i in range(N+1):
        for j in range(N - i + 1):
            # print("------------- printing GradSimplex2DP output ------------")
            # print("calling params:", "a:", a, "b:", b, "i:", i, "j:", j)
            # print("Shapes", "a:", a.shape, "b:", b.shape)
            t1, t2 = GradSimplex2DP(a, b, i, j)
            # print("t1:", t1)
            # print("t2:", t2)
            # print(V2Dr[:, sk].shape)
            # print(V2Ds[:, sk].shape)
            # t1.shape = V2Dr[:, sk].shape
            # t2.shape = V2Ds[:, sk].shape
            V2Dr[:, sk], V2Ds[:, sk] = t1, t2
            # print("V2DR" , V2Dr)
            # print("V2DS" , V2Ds)
            sk += 1
    return V2Dr, V2Ds


def Dmatrices2D(N,r,s,V): 
    Vr, Vs = GradVandermonde2D(N, r, s)
    print("Vr", Vr)
    print("Vs", Vs)
    print("V", V)
    Dr = Vr @ np.linalg.inv(V)
    Ds = Vs @ np.linalg.inv(V)
    
    return Dr, Ds


def geometric_factors(x, y, Dr, Ds):
    xr = Dr @ x
    xs = Ds @ x
    yr = Dr @ y
    ys = Ds @ y
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
        Pmat[i, :] = JacobiP(rout, 0, 0, i).T

    Lmat = np.linalg.solve(Veq.T, Pmat)

    warp = Lmat.T @ (LGLr - req)
    zerof = (abs(rout) < Tol)               ## Boolean array ? does it work?
    sf = 1.0 - (zerof*rout)**2
    warp = warp/sf + warp * (zerof - 1)
    return warp


def Vandermonde1D(N, r):
    V1D = np.zeros((len(r), N + 1))
    for j in range(N + 1):
        #print(r, 0, 0, j)
        temp = JacobiP(r, 0, 0, j)
        V1D[:, j] = temp.T
    return V1D


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

    # print("Initial x:", x)
    # print("Initial y:", y)

    blend1 = 4*L2*L3
    blend2 = 4*L1*L3
    blend3 = 4*L1*L2
    warpf1 = Warpfactor(N, L3-L2)
    warpf2 = Warpfactor(N, L1-L3)
    warpf3 = Warpfactor(N, L2-L1)
    warpf1.shape = blend1.shape
    warpf2.shape = blend2.shape
    warpf3.shape = blend3.shape
    # print("Warpf1:", warpf1)
    # print("Warpf2:", warpf2)

    # print("blend1:", blend1)
    # print("blend2:", blend2)

    warp1 = blend1*warpf1*(1 + (alpha*L1)**2)
    warp2 = blend2*warpf2*(1 + (alpha*L2)**2)
    warp3 = blend3*warpf3*(1 + (alpha*L3)**2)
    test = np.multiply(blend1, (1 + (alpha*L1)**2))
    # print("debug",test)
    
    # print("debug 2", np.multiply(test, warpf1))
    # print( "warp1:", warp1)
    # print( "warp2:", warp2)
    # print( "warp3:", warp3)
    # print("x before warp:", x)
    # print("y before warp:", y)

    x = x + 1*warp1 + np.cos(2*np.pi/3)*warp2 + np.cos(4*np.pi/3)*warp3
    y = y + 0*warp1 + np.sin(2*np.pi/3)*warp2 + np.sin(4*np.pi/3)*warp3

    # print(1*warp1.T)

    # print("x after warp:", x)
    # print("y after warp:", y)
    return x, y


if __name__ == "__main__":
    elemsx = 4
    elemsy = 3
    P = 2
    N = elemsx*elemsy*2
    EToV = gen_EToV(elemsx, elemsy)
    EToE = ti_connect2D(elemsx, elemsy)

    x,y = Nodes2D(P) # Local coords
    r,s = xytors(x,y)

    # Constants:
    Nfp = P + 1
    Nfaces = 3
    Mp = int((P+1)*(P+2)/2)
    NODETOL = 1e-12


    V = Vandermonde2D(P, r, s)
    print("Vandermonde", V)
    Vr, Vs = GradVandermonde2D(P,r,s)
    print("Vr:", Vr)
    print("Vs:", Vs)

    Dr, Ds = Dmatrices2D(P, r, s, V)
    invV = np.linalg.inv(V)

    print("DR:", Dr)
    print("DS:", Ds)

    v1 = EToV[:, 0].T
    v2 = EToV[:, 1].T
    v3 = EToV[:, 2].T

    vx,vy = gen_node_coordinates(elemsx, elemsy)
    print("vx:", vx)
    print("vy:", vy)


    # x and y will have size (Mp x N)?
    # print(-(r+s))
    # print(vx[v1])
    #print(vx)
    #print(v1)
    # (-(r+s)).shape = (len(r),)
    # vx.shape = (len(vx),1)


    x = 0.5*(-(r.T+s.T) * vx[v1] + (1 + r.T) * vx[v2] + (1 + s.T) * vx[v3]) 
    y = 0.5*(-(r.T+s.T) * vy[v1] + (1 + r.T) * vy[v2] + (1 + s.T) * vy[v3])
    x = x.T
    y = y.T

    tol = 0.2/P**2

    ## Implement reordering of nodes here
    # Not done yet

    print("x after mapping:", x)
    print("y after mapping:", y)

    # print("EToV:\n", EToV)
    # print("EToE:\n", EToE)
    # print("Node coordinates (x):\n", x)
    # print("Node coordinates (y):\n", y)
    C = algo14(P, N, EToV, EToE)
    print("Connection table C:\n", C)
    print(C.shape)


    q = lambda n: 1 # Dummy q

    A, B = global_assembly(C, N, P, x, y, q, V, Dr, Ds)
    print("Global matrix A:\n", A)
    print("Global vector B:\n", B)


