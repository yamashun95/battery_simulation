import numpy as np
import matplotlib.pyplot as plt
from utils import *


def simulate_electrochemical_system(x, t, Uapp, r0, a, F_RT, D):
    Nx = len(x)
    Ntau = len(t)
    tau = t * D / r0**2
    dx = x[1] - x[0]
    dtau = tau[1] - tau[0]
    alpha0 = dtau / (2 * dx**2)
    alpha1 = dtau / (2 * dx)
    y = np.ones(Nx) * 0.994
    js = np.zeros(Ntau)
    A = np.zeros((Nx, Nx))
    B = np.zeros((Nx, Nx))
    for i in range(1, Nx - 1):
        A[i, i - 1] = -alpha0 + alpha1 / x[i]
        A[i, i] = 1 + 2 * alpha0
        A[i, i + 1] = -alpha0 - alpha1 / x[i]
        B[i, i - 1] = alpha0 - alpha1 / x[i]
        B[i, i] = 1 - 2 * alpha0
        B[i, i + 1] = alpha0 + alpha1 / x[i]
    g0 = 0
    A[0, 0] = A[-1, -1] = 1
    B[0, 0] = B[-1, -1] = 1
    for itau in range(Ntau):
        gN = (
            -a
            * np.sqrt((1.0 - y[-2]) * (y[-2]))
            * 2
            * np.sinh(0.5 * F_RT * (Uapp[itau] - ocv_LMO(y[-2])))
        )
        y[0] = y[1]
        y[-1] = y[-2] + gN * dx
        b = np.dot(B, y)

        yn = np.linalg.solve(A, b)
        y = yn
        print(y[-10:])
        js[itau] = (
            a
            * np.sqrt((1.0 - y[-1]) * (y[-1]))
            * 2
            * np.sinh(0.5 * F_RT * (Uapp[itau] - ocv_LMO(y[-1])))
        )

    return y, js, tau, Uapp
