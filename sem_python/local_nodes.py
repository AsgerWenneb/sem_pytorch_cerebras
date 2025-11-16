import numpy as np
from vandermonde import Vandermonde1D
from polynomials import JacobiGL, JacobiP


def Warpfactor(N, rout, Tol=1e-10):
    LGLr = JacobiGL(0, 0, N)
    req = np.linspace(-1, 1, N+1)

    Veq = Vandermonde1D(N, LGLr)

    Nr = len(rout)
    Pmat = np.zeros((N + 1, Nr))
    for i in range(N + 1):
        Pmat[i, :] = JacobiP(rout, 0, 0, i).T

    Lmat = np.linalg.solve(Veq.T, Pmat)

    warp = Lmat.T @ (LGLr - req)
    zerof = (abs(rout) < Tol)  # Boolean array ? does it work?
    sf = 1.0 - (zerof*rout)**2
    warp = warp/sf + warp * (zerof - 1)
    return warp


def Nodes2D(N):
    # Generates x and y vectors. See p 179.
    alpopt = [0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999,
              1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258]
    Np = int((N+1)*(N+2)/2)

    if N <= 14:
        alpha = alpopt[N]
    else:
        alpha = 5/3

    L1 = np.zeros((Np,))
    L2 = np.zeros((Np,))
    L3 = np.zeros((Np,))

    sk = 0
    for n in range(N+1):
        for m in range(N - n + 1):  # subtract one less due to diff indexing of n
            L1[sk] = n / N
            L3[sk] = m / N
            sk += 1
    L2 = 1.0 - L1 - L3
    x = - L2 + L3
    y = (-L2 - L3 + 2*L1)/np.sqrt(3)

    blend1 = 4*L2*L3
    blend2 = 4*L1*L3
    blend3 = 4*L1*L2
    print("Debugging warpfactor:")
    print(L3-L2)
    warpf1 = Warpfactor(N, L3-L2)
    warpf2 = Warpfactor(N, L1-L3)
    warpf3 = Warpfactor(N, L2-L1)
    warpf1.shape = blend1.shape
    warpf2.shape = blend2.shape
    warpf3.shape = blend3.shape

    warp1 = blend1*warpf1*(1 + (alpha*L1)**2)
    warp2 = blend2*warpf2*(1 + (alpha*L2)**2)
    warp3 = blend3*warpf3*(1 + (alpha*L3)**2)
    test = np.multiply(blend1, (1 + (alpha*L1)**2))

    x = x + 1*warp1 + np.cos(2*np.pi/3)*warp2 + np.cos(4*np.pi/3)*warp3
    y = y + 0*warp1 + np.sin(2*np.pi/3)*warp2 + np.sin(4*np.pi/3)*warp3

    return x, y


if __name__ == "__main__":
    N = 4
    x, y = Nodes2D(N)
    print("x:", x)
    print("y:", y)
