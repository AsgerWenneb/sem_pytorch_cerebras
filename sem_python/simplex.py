import numpy as np
from polynomials import JacobiP, GradJacobiP


def Simplex2DP(a, b, i, j):
    # Evaluate 2D orthogonal polynomial on simplex at (a,b) of order (id,jd)
    h1 = JacobiP(a, 0, 0, i)
    h2 = JacobiP(b, 2*i+1, 0, j)
    return np.sqrt(2.0)*h1.T*h2.T*((1 - b)**i)


def GradSimplex2DP(a, b, id, jd):  # Check functionality, check ids
    fa = JacobiP(a, 0, 0, id)
    dfa = GradJacobiP(a, 0, 0, id)
    gb = JacobiP(b, 2*id+1, 0, jd)
    dgb = GradJacobiP(b, 2*id+1, 0, jd)

    dmodedr = dfa*gb
    if id > 0:
        dmodedr = dmodedr*((0.5*(1-b))**(id - 1))

    dmodeds = dfa*(gb*(0.5*(1+a)))

    if id > 0:
        dmodeds = dmodeds*((0.5*(1-b))**(id - 1))

    tmp = dgb*(0.5*(1 - b))**id  # Elementwise mult
    if id > 0:
        tmp = tmp - 0.5*id*gb*((0.5*(1 - b))**(id - 1))
    dmodeds = dmodeds + fa*tmp  # elementwise mult

    dmodedr = 2**(id + 0.5)*dmodedr
    dmodeds = 2**(id + 0.5)*dmodeds
    return dmodedr, dmodeds
