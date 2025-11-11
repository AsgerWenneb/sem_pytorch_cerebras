## Methods for grid generation and connection tables.
## Assumes standard triangular grid.
import numpy as np
from global_nodes import gen_node_coordinates
from local_nodes import Nodes2D
from coordinate_maps import xytors
from connection_tables import algo14, gen_EToV, ti_connect2D, convert_coords_to_vec
from vandermonde import Vandermonde2D, GradVandermonde2D




def global_assembly(C, N, Ne, P, x, y, q, V, Dr, Ds): # combined algo 15 and 16
    Mp =  int((P+1)*(P+2)/2)
    A = np.zeros((Ne,Ne))
    B = np.zeros(Ne,)

    rx, sx, ry, sy, J = geometric_factors(x, y, Dr, Ds)
    M = np.linalg.inv(V @ V.T) # Mass matrix

    for n in range(N):
        # Compute kij(n) (4.46) - value
        Dx = np.diag(rx[:, n]) @ Dr + np.diag(sx[:, n]) @ Ds
        Dy = np.diag(ry[:, n]) @ Dr + np.diag(sy[:, n]) @ Ds
        tmp1 = Dx.T @ M @ Dx
        tmp2 = Dy.T @ M @ Dy
        tmp = tmp1 + tmp2

        kij = J.T[n]*tmp # same as np.diag(J[:, n]) @ tmp 

        mij = (np.diag(J[:, n]) @ M)
        for j in range(Mp):
            # Coordinates of local idx j in N- different from algo due to x/y data structure
            # Could pass xv and yv and use jj as index.
            # jj = C[n,j]
            xj = x[j, n]
            yj = y[j, n]
            print("Global idx:",C[n,j], "xj:", xj, "yj:", yj)

            for i in range(Mp):
                if C[n,j] >= C[n,i]:
                    A[C[n,i], C[n,j]] += kij[i,j]

                ii = C[n,i]
                #print("B shape" , B.shape)
                #print(ii)
                f = -q(xj, yj)
                B[ii] +=  mij[i,j]* f
    return A, B


def impose_dirichlet_bc(BoundaryNodesidx, A, B, f, xv, yv, boundary_f):
    for bn in BoundaryNodesidx:
        xn = xv[bn]
        yn = yv[bn]
        B += -f(xn,yn)*A[:, bn]
        A[bn, :] = 0
        A[:, bn] = 0
        A[bn, bn] = 1
    for bn in BoundaryNodesidx:
        xn = xv[bn]
        yn = yv[bn]
        B[bn] = boundary_f(xn, yn)
    return A, B
    

def Dmatrices2D(N,r,s,V): 
    Vr, Vs = GradVandermonde2D(N, r, s)
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


if __name__ == "__main__":
    elemsx = 4
    elemsy = 3
    P = 2
    N = elemsx*elemsy*2
    EToV = gen_EToV(elemsx, elemsy)
    EToE = ti_connect2D(elemsx, elemsy)

    # Local coords
    x,y = Nodes2D(P) 
    print(x.shape)
    r,s = xytors(x,y)

    # Constants:
    Nfp = P + 1
    Nfaces = 3
    Mp = int((P+1)*(P+2)/2)
    NODETOL = 1e-12


    # Vandermonde and D matrices
    V = Vandermonde2D(P, r, s)
    Vr, Vs = GradVandermonde2D(P,r,s)
    Dr, Ds = Dmatrices2D(P, r, s, V)
    invV = np.linalg.inv(V)


    v1 = EToV[:, 0].T
    v2 = EToV[:, 1].T
    v3 = EToV[:, 2].T

    # Global corner node coordinates
    vx,vy = gen_node_coordinates(elemsx, elemsy, -2,4,-2,4)

    # Map local into global coordinates
    x = 0.5*(-(r.T+s.T) * vx[v1] + (1 + r.T) * vx[v2] + (1 + s.T) * vx[v3]) 
    y = 0.5*(-(r.T+s.T) * vy[v1] + (1 + r.T) * vy[v2] + (1 + s.T) * vy[v3])
    x = x.T
    y = y.T



    C, Ne = algo14(P, N, EToV, EToE) 
    print("Number of elements:", Ne)
    print("Connection table C:\n", C)

    print("r, ", r)
    print("s, ", s)

    tol = 0.2/P**2

    ## Implement reordering of nodes here
    # Copied exactly from FEM book

    #% Faces
    fid1 = np.nonzero( abs(s + 1) < tol)[0]
    fid2 = np.nonzero( abs(r+s) < tol)[0]
    fid3 = np.nonzero( abs(r + 1) < tol)[0]
    # Fids is used to compute outer normals 
    # Fids = np.concatenate([fid1, fid2, fid3])
    
    #% Interior
    fint = np.setdiff1d(np.arange(0, Mp), np.concatenate([fid1, fid2, fid3]))
    print("Interior nodes indices:", fint)
    Mpf = P + 1
    print(fid1[1:Mpf-2])
    print(fid2[1:Mpf-2])
    print(fid3[Mpf-2:1])
    print(fid1)
    print(fid2)
    print(fid3)
    Local_reorder = np.concatenate([0, Mpf - 1, Mp - 1, fid1[1:Mpf-2], fid2[1:Mpf-2], fid3[Mpf-2:1], fint]).T



    xv, yv = convert_coords_to_vec(C, Ne, x, y)


    # Global assembly params
    q = lambda x,y: 1 # Dummy q

    A, B = global_assembly(C, N, Ne, P, x, y, q, V, Dr, Ds)

    print("Global matrix A:\n", A)
    print("Global vector B:\n", B)

