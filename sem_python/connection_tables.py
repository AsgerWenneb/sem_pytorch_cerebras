## Methods for grid generation and connection tables.
## Assumes standard triangular grid.
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


def algo14(P, N, EToV, EToE):
    gidx = EToV.shape[0]
    Mp = (P+1)*(P+2)/2
    Mpf = P + 1
    Nfaces = 3

    C = np.zeros((gidx*Mp, gidx*Mp))
    for n in range(N):
        for i in range(Nfaces):
            C[n, i] = EToV[n, i]
            if EToE[n, i] >= n: # if neighbor has higher index than current index:
                pass # Assign global index to points on face
            else: # Else, the points have been assigned an index, so this index is copied and reversed.
                pass

    # Error in lidx definition, should have been as below instead of just Mp
    # Internal nodes = Mp - nfaces - nfaces*(P-2) (for 1 indexed) 

    # 








if __name__ == "__main__":
    elemsx = 4
    elemsy = 3
    EToV = gen_EToV(elemsx, elemsy)
    EToE = ti_connect2D(elemsx, elemsy)
    print("EToV:\n", EToV)
    print("EToE:\n", EToE)



