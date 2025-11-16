# Methods for grid generation and connection tables.
# Assumes standard triangular grid.
import numpy as np
from global_nodes import gen_node_coordinates
from local_nodes import Nodes2D
from coordinate_maps import xytors
from connection_tables import algo14, gen_EToV, ti_connect2D, convert_coords_to_vec, boundary_nodes_from_grid
from vandermonde import Vandermonde2D, GradVandermonde2D
from data_export import export_etov, export_data_header, export_solution


def global_assembly(C, N, Ne, P, x, y, q, V, Dr, Ds):  # combined algo 15 and 16
    Mp = int((P+1)*(P+2)/2)
    print("Global assembly called with Ne =", Ne, "and N =", N)
    A = np.zeros((Ne, Ne))
    B = np.zeros(Ne,)

    rx, sx, ry, sy, J = geometric_factors(x, y, Dr, Ds)
    M = np.linalg.inv(V @ V.T)  # Mass matrix

    for n in range(N):
        # Compute kij(n) (4.46) - value
        Dx = np.diag(rx[:, n]) @ Dr + np.diag(sx[:, n]) @ Ds
        Dy = np.diag(ry[:, n]) @ Dr + np.diag(sy[:, n]) @ Ds
        tmp1 = Dx.T @ M @ Dx
        tmp2 = Dy.T @ M @ Dy
        tmp = tmp1 + tmp2

        kij = J.T[n]*tmp  # same as np.diag(J[:, n]) @ tmp

        mij = (np.diag(J[:, n]) @ M)
        for j in range(Mp):
            # Coordinates of local idx j in N- different from algo due to x/y data structure
            # Could pass xv and yv and use jj as index.
            # jj = C[n,j]
            xj = x[j, n]
            yj = y[j, n]

            for i in range(Mp):
                # if C[n,j] >= C[n,i]: ## Results in symmetric assembly
                A[C[n, i], C[n, j]] += kij[i, j]

                ii = C[n, i]
                # print("B shape" , B.shape)
                # print(ii)
                f = -q(xj, yj)
                B[ii] += mij[i, j] * f
    return A, B


def impose_dirichlet_bc(BoundaryNodesidx, A, B, f, xv, yv, boundary_f):
    for bn in BoundaryNodesidx:
        xn = xv[bn]
        yn = yv[bn]
        B += -f(xn, yn)*A[:, bn]
        A[bn, :] = 0
        A[:, bn] = 0
        A[bn, bn] = 1
    for bn in BoundaryNodesidx:
        xn = xv[bn]
        yn = yv[bn]
        B[bn] = boundary_f(xn, yn)
    return A, B


def Dmatrices2D(N, r, s, V):
    print("Calling params:")
    print("N = ", N)
    print("r = ", r)
    print("s = ", s)
    print("V = ", V)
    Vr, Vs = GradVandermonde2D(N, r, s)
    Dr = Vr @ np.linalg.inv(V)
    Ds = Vs @ np.linalg.inv(V)
    return Dr, Ds


def geometric_factors(x, y, Dr, Ds):
    print("x = ", x)
    print("y = ", y)
    print("Dr = ", Dr)
    print("Ds = ", Ds)
    xr = Dr @ x
    xs = Ds @ x
    yr = Dr @ y
    ys = Ds @ y
    print("xr =", xr)
    print("xs =", xs)
    print("yr =", yr)
    print("ys =", ys)
    J = xr*ys - xs*yr
    print("J =", J)
    rx = ys/J
    sx = -yr/J
    ry = -xs/J
    sy = xr/J
    return rx, sx, ry, sy, J


if __name__ == "__main__":
    elemsx = 4
    elemsy = 3
    P = 3
    N = elemsx*elemsy*2
    EToV = gen_EToV(elemsx, elemsy)
    EToE = ti_connect2D(elemsx, elemsy)

    # Local coords
    x, y = Nodes2D(P)
    print(x.shape)
    r, s = xytors(x, y)

    # Constants:
    Nfp = P + 1
    Nfaces = 3
    Mp = int((P+1)*(P+2)/2)
    NODETOL = 1e-12

    # Vandermonde and D matrices
    V = Vandermonde2D(P, r, s)
    Vr, Vs = GradVandermonde2D(P, r, s)
    Dr, Ds = Dmatrices2D(P, r, s, V)
    invV = np.linalg.inv(V)

    v1 = EToV[:, 0]
    v2 = EToV[:, 1]
    v3 = EToV[:, 2]

    # Global corner node coordinates
    vx, vy = gen_node_coordinates(elemsx, elemsy, -2, 4, -2, 4)

    # Map local into global coordinates
    x = 0.5*(np.outer(-(r+s), vx[v1]) + np.outer((1 + r),
             vx[v2]) + np.outer((1 + s), vx[v3]))
    y = 0.5*(np.outer(-(r+s), vy[v1]) + np.outer((1 + r),
             vy[v2]) + np.outer((1 + s), vy[v3]))

    x = x
    y = y
    print(x)

    C, Ne = algo14(P, N, EToV, EToE, elemsx, elemsy)
    print("Number of unique nodes:", Ne)
    print("Connection table C:\n", C)

    # print("r, ", r)
    # print("s, ", s)

    tol = 0.2/P**2

    # Implement reordering of nodes here
    # Copied exactly from FEM book

    # % Faces
    fid1 = np.nonzero(abs(s + 1) < tol)[0]
    fid2 = np.nonzero(abs(r+s) < tol)[0]
    fid3 = np.nonzero(abs(r + 1) < tol)[0]
    # Fids is used to compute outer normals
    # Fids = np.concatenate([fid1, fid2, fid3])

    # % Interior
    fint = np.setdiff1d(np.arange(0, Mp), np.concatenate([fid1, fid2, fid3]))
    # print("Interior nodes indices:", fint)
    Mpf = P + 1
    # print(fid1[1:Mpf-1])
    # print(fid2[1:Mpf-1])
    # print(np.flip(fid3[1:Mpf-1]))
    # print(fid1)
    # print(fid2)
    # print(fid3)
    Local_reorder = np.concatenate([np.array([0]), np.array(
        [Mpf - 1]), np.array([Mp - 1]), fid1[1:Mpf-1], fid2[1:Mpf-1], np.flip(fid3[1:Mpf-1]), fint])

    # Use the reordering:
    x = x[Local_reorder, :]
    y = y[Local_reorder, :]

    print("reordering:", Local_reorder)

    # Vector style x and y
    xv, yv = convert_coords_to_vec(C, Ne, x.T, y.T)

    # Global assembly params
    def q(x, y): return 0  # Dummy q
    def test_bc(x, y): return 1

    print("x =", x)
    print("y =", y)
    print("Dr =", Dr)
    print("Ds =", Ds)

    A, B = global_assembly(C, N, Ne, P, x, y, q, V, Dr, Ds)

    b_nodes = boundary_nodes_from_grid(elemsx, elemsy, P, C)
    print("Boundary nodes indices:", b_nodes)
    A, B = impose_dirichlet_bc(b_nodes, A, B, q, xv, yv, test_bc)

    print("Global matrix A:\n", A)
    print("Global vector B:\n", B)

    print("Size of A:", A.shape)
    print("Size of B:", B.shape)

    sol = np.linalg.solve(A, B)

    print("Solution:", sol)
    print("Solution max value:", np.max(sol))
    print("Solution min value:", np.min(sol))

    # Export data
    print(xv)

    n_corner_nodes = (elemsx + 1) * (elemsy + 1)
    export_data_header("output_data.dat", n_corner_nodes, xv, yv)
    export_solution("output_data.dat", sol)

    export_etov("output_etov.csv", C)
    print(xv.dtype)
