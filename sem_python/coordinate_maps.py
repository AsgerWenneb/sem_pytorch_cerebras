import numpy as np


def xytors(x, y):
    # print("x =", x)
    # print("y =", y)
    L1 = (np.sqrt(3)*y + 1)/3
    L2 = (-3*x - np.sqrt(3)*y + 2)/6
    L3 = (3*x - np.sqrt(3)*y + 2)/6
    r = -L2 + L3 - L1
    s = -L2 - L3 + L1
    return r, s


def rstoab(r, s):
    Np = len(r)
    a = np.zeros((Np,))
    for i in range(Np):
        if s[i] != 1:
            a[i] = 2*(1 + r[i])/(1 - s[i]) - 1
        else:
            a[i] = -1
    return a, s
