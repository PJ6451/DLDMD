import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
import numpy as np

from pydmd import DMD

def f1(x, t):
    return 1.0 / np.cosh(x + 3) * np.exp(2.3j * t)


def f2(x, t):
    return 2.0 / np.cosh(x) * np.tanh(x) * np.exp(2.8j * t)


x = np.linspace(-5, 5, 4)
t = np.linspace(0, 4 * np.pi, 129)

xgrid, tgrid = np.meshgrid(x, t)

X1 = f1(xgrid, tgrid)
X2 = f2(xgrid, tgrid)
X = X1 + X2

dmd = DMD(svd_rank=2)
dmd.fit(X.T)

for mode in dmd.modes.T:
    plt.plot(x, mode.real)
    plt.title("Modes")
plt.show()

for dynamic in dmd.dynamics:
    plt.plot(t, dynamic.real)
    plt.title("Dynamics")
plt.show()