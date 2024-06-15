import numpy as np
import matplotlib.pyplot as plt
from utils import *


def simulate_electrochemical_system(r0, nu, a, U0, F_RT, D, Nx, Ntau, t_max):
    x = np.linspace(0, 1, Nx)
    t = np.linspace(0, t_max, Ntau)
    tau = t * D / r0**2
    dx = x[1] - x[0]
    dtau = tau[1] - tau[0]
    alpha0 = dtau / dx**2
    alpha1 = 2 * dtau / dx
    Uapp = U0 + nu * t
    y = np.ones(Nx) * 0.99
    js = np.zeros(Ntau)

    for itau in range(Ntau):
        yn = np.zeros(Nx)
        print(f"ddx:{alpha0*(y[Nx - 1] - 2 * y[Nx - 2] + y[Nx - 3])}")
        for i in range(1, Nx - 1):
            yn[i] = (
                y[i]
                + alpha0 * (y[i + 1] - 2 * y[i] + y[i - 1])
                # + alpha1 / x[i] * (y[i + 1] - y[i])
            )
        yn[0] = yn[1]
        print(
            -dx
            * a
            * np.sqrt((1.0 - yn[-2]) * (yn[-2]))
            * 2
            * np.sinh(0.5 * F_RT * (Uapp[itau] - ocv_LMO(yn[-2])))
        )
        yn[-1] = yn[-2] - dx * a * np.sqrt((1.0 - yn[-2]) * (yn[-2])) * 2 * np.sinh(
            0.5 * F_RT * (Uapp[itau] - ocv_LMO(yn[-2]))
        )
        print(yn[-10:])
        y = yn
        js[itau] = (
            a
            * np.sqrt((1.0 - y[-1]) * (y[-1]))
            * 2
            * np.sinh(0.5 * F_RT * (Uapp[itau] - ocv_LMO(y[-1])))
        )
        print(y[-1])

    return y, js, tau


def plot_results(tau, js):
    plt.figure()
    plt.plot(tau, js)
    plt.xlabel("Tau")
    plt.ylabel("Current Density (j)")
    plt.title("Current Density vs. Time")
    plt.show()
