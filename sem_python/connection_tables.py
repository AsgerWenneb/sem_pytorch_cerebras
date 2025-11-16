import numpy as np

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


def algo14(P, N, EToV, EToE, elemsx, elemsy):
    ## For starters implemented as in book

    gidx = (elemsx + 1)*(elemsy + 1)
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
    return C.astype('int'), gidx


def convert_coords_to_vec(C, Ne, x, y): ## Quite inefficient, double work for corner and edge nodes
    #print("Called with Ne =", Ne, "and C shape", C.shape, "x shape", x.shape, "y shape", y.shape)
    xv = np.zeros(Ne, )
    yv = np.zeros(Ne, )

    for n in range(C.shape[0]):
        for i in range(C.shape[1]):
            idx = C[n, i]
            xv[idx] = x[n, i]
            yv[idx] = y[n, i]
    return xv, yv

def boundary_nodes_from_grid(elemsx, elemsy, P, C):
    idx = 0
    num_bnodes = (2*(elemsy +1) +  2*(elemsx - 1)) + (P-1)*(2*(elemsx + elemsy))  # First order nodes + higher order nodes
    print(num_bnodes)
    boundary_nodes = np.zeros([num_bnodes], dtype=int)

    # Add 1st order nodes first:
    for j in range(elemsy + 1):
        boundary_nodes[idx] = j
        boundary_nodes[idx + 1] = j + (elemsy + 1)*elemsx
        idx += 2
    for i in range(1,elemsx):
        boundary_nodes[idx] = i*(elemsy + 1)
        boundary_nodes[idx + 1] = (i+1)*(elemsy + 1) - 1
        idx += 2

    # Higher order nodes:
    # Will be located at index 3:(P-1) in C, or offset depending on face id
    # For y-axis: fid 3.
    # For x-axis: fid 1.
    if P > 1:
        left_boundary_id = range(1, elemsy*2, 2)
        right_boundary_id = range(2*elemsx*elemsy - elemsy*2, 2*elemsx*elemsy, 2)

        top_boundary_id = range(0, 2*elemsx*elemsy - elemsy*2 + 1, 2*elemsy)
        bottom_boundary_id = range(elemsy*2 -1, 2*elemsx*elemsy, 2*elemsy)

        # print("Left boundary element IDs:", list(left_boundary_id))
        # print("Right boundary element IDs:", list(right_boundary_id))
        # print("Top boundary element IDs:", list(top_boundary_id))
        # print("Bottom boundary element IDs:", list(bottom_boundary_id))

        for eid in left_boundary_id:
            for p in range(1, P):
                boundary_nodes[idx] = C[eid, 2 + p]
                idx += 1
        for eid in right_boundary_id:
            for p in range(1, P):
                boundary_nodes[idx] = C[eid, 2 + p]
                idx += 1
        for eid in top_boundary_id:
            for p in range(1, P):
                boundary_nodes[idx] = C[eid, 2 + 2*(P-1) + p]
                idx += 1
        for eid in bottom_boundary_id:
            for p in range(1, P):
                boundary_nodes[idx] = C[eid, 2 + 2*(P-1) + p]
                idx += 1
    return boundary_nodes

