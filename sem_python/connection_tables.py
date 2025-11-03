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