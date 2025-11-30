import numpy as np
from math import gamma


def JacobiP(x, alpha, beta, N):  
    xp = x  # Transformation to right dim - trivial in python.
    PL = np.zeros((N+1, len(xp)))

    gamma0 = 2**(alpha + beta + 1)/(alpha + beta + 1) * \
        gamma(alpha + 1)*gamma(beta + 1)/gamma(alpha + beta + 1)
    PL[0, :] = 1.0/np.sqrt(gamma0)
    if N == 0:
        return PL[N, :].T
    gamma1 = (alpha + 1)*(beta + 1)/(alpha + beta + 3)*gamma0
    PL[1, :] = (((alpha + beta + 2)*xp/2 + (alpha - beta)/2) /
                np.sqrt(gamma1)).T  # Transposed to fix dims
    if N == 1:
        return PL[N, :].T

    aold = 2/(2 + alpha + beta)*np.sqrt((alpha + 1)
                                        * (beta + 1)/(alpha + beta + 3))

    for i in range(1, N):
        h1 = 2*i + alpha + beta
        anew = 2/(h1 + 2)*np.sqrt((i+1)*(i+1+alpha+beta)
                                  * (i+1+alpha)*(i+1+beta)/(h1+1)/(h1+3))
        bnew = -(alpha*alpha - beta*beta)/(h1*(h1 + 2))
        PL[i+1, :] = 1/anew*(-aold*PL[i-1, :] + (xp - bnew)*PL[i, :])
        aold = anew
    return PL[N, :].T


def GradJacobiP(x, alpha, beta, N):
    dp = np.zeros((len(x),))
    if N == 0:
        return dp
    else:
        dp = np.sqrt(N*(N + alpha + beta + 1)) * \
            JacobiP(x, alpha + 1, beta + 1, N - 1)
    return dp


def JacobiGQ(alpha, beta, N):  
    x = np.zeros((N + 1,))
    w = np.zeros((N + 1,))
    if N == 0:
        x[0] = (alpha - beta)/(alpha + beta + 2)
        w[0] = 2
        return x, w

    J = np.zeros((N + 1, N + 1))
    h1 = 2*np.arange(0, N + 1) + alpha + beta

    J = np.diag(-1/2*(alpha**2 - beta**2)/(h1 + 2)/h1) + \
        np.diag(2/(h1[0:N] + 2) * np.sqrt(np.arange(1, N + 1)*(np.arange(1, N + 1) + alpha + beta) *
                                          (np.arange(1, N + 1) + alpha)*(np.arange(1, N + 1) + beta)/(h1[0:N] + 1)/(h1[0:N] + 3)), 1)
    J = J + J.T
    
    # D should be the diagonal matrix of eigenvalues (it is not in python, only matlab), V the eigenvectors
    D, V = np.linalg.eig(J)
    x = D  # Kept this way to reflect original matlab code
    w = (V[0, :].T)**2 * 2**(alpha + beta + 1)/(alpha + beta + 1) * \
        gamma(alpha + 1)*gamma(beta + 1)/gamma(alpha + beta + 1)
    return x, w


def JacobiGL(alpha, beta, N):
    x = np.zeros((N+1,))
    if N == 1:
        x[0] = -1
        x[1] = 1
        return x
    xint, w = JacobiGQ(alpha + 1, beta + 1, N - 2)
    x[0] = -1
    x[1:N] = xint
    x[N] = 1
    return x


if __name__ == "__main__":
    N = 5
    x = np.linspace(-1, 1, 3)
    print(x.shape)

    print("Test case for polynomials.py")
    print("x:", x)
    print("N:", N)

    P = JacobiP(x, 0, 0, N)
    print("P:", P)
    print("Shape of P", P.shape)
    # dP = GradJacobiP(x, 0, 0, N)
    # print("dP:", dP)
    # xgq, wgq = JacobiGQ(0, 0, N)
    # print("xgq:", xgq)
    # print("wgq:", wgq)
    # xgl = JacobiGL(0, 0, N)
    # print("xgl:", xgl)
