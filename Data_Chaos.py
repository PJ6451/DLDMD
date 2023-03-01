"""
    Created by:
        Opal Issan

    Modified:
        17 Nov 2020 - Jay Lago
"""
import numpy as np
import pickle
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ==============================================================================
# Dynamical System Functions
# ==============================================================================
def lorenz63(lhs):
    sigma, rval, bval = 10., 28., 8./3.
    y1, y2, y3 = lhs[0], lhs[1], lhs[2]
    rhs = np.zeros(3, dtype=np.float64)
    rhs[0] = sigma*(y2-y1)
    rhs[1] = rval*y1-y2-y1*y3
    rhs[2] = -bval*y3 + y1*y2
    return rhs


def rossler(lhs):
    aval, bval, cval = .1, .1, 5.7
    y1, y2, y3 = lhs[0], lhs[1], lhs[2]
    rhs = np.zeros(3, dtype=np.float64)
    rhs[0] = -y2 - y3
    rhs[1] = y1 + aval*y2
    rhs[2] = bval + y3 * (y1 - cval)
    return rhs


def lorenz96(lhs):
    F = 8.
    rhs = -lhs + F + ( np.roll(lhs,-1) - np.roll(lhs,2) ) * np.roll(lhs,1)
    return rhs


# ==============================================================================
# Solver Functions
# ==============================================================================
def rk4(x0, f, dt):
    k1 = dt*f(x0)
    k2 = dt*f(x0 + k1/2.)
    k3 = dt*f(x0 + k2/2.)
    k4 = dt*f(x0 + k3)
    return x0 + (k1 + 2.*k2 + 2.*k3 + k4)/6.

# Time stepping scheme for solving x' = f(x) for t0<=t<=tf with time step dt.
def timestepper(x0,t0,tf,dt,f):
    ndim = np.size(x0)
    nsteps = int((tf-t0)/dt)
    solpath = np.zeros((ndim,nsteps),dtype=np.float64)
    solpath[:,0] = x0
    for jj in range(1, nsteps):
        solpath[:, jj] = rk4(solpath[:, jj-1], f, dt)
    return solpath


def data_builder(n_ic, dim, x0, tf, dt, dyn_sys):
    nsteps = int(tf / dt)
    n_ic = int(n_ic)
    # Generate initial conditions
    initconds = np.zeros((n_ic, dim), dtype=np.float64)
    rawdata = np.zeros([n_ic, dim, nsteps], dtype=np.float64)
    for ll in range(n_ic):
        initconds[ll,:-1] = 5*np.random.uniform(-1,1,2)
        initconds[ll,:] = initconds[ll,:] + np.random.uniform(-.5,.5,dim)
        rawdata[ll,:,:] = timestepper(initconds[ll,:], 0, tf, dt, dyn_sys)

    return np.transpose(rawdata, [0, 2, 1])


# ==============================================================================
# Test program
# ==============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ax = plt.axes(projection='3d')
    create_lorenz63 = False
    create_rossler = True
    create_lorenz96 = False

    if create_lorenz63:
        # Generate the data
        data_fname = 'lorenz63_data.pkl'
        x0 = np.array([3.1, 3.1, 27.])
        data = data_builder(150, 3, x0, 30., 0.05, lorenz63)
        pickle.dump(data, open(data_fname, 'wb'))

    if create_rossler:
        # Generate the data
        data_fname = 'rossler_data.pkl'
        x0 = np.array([.1, .1, .1])
        data = data_builder(150, 3, x0, 10., 0.05, rossler)
        #pickle.dump(data, open(data_fname, 'wb'))
        for ii in range(data.shape[0]):
            ax.plot3D(data[ii, :, 0], data[ii, :, 1], data[ii, :, 2], '-')
        ax.set_xlabel("x1", fontsize=18)
        ax.set_ylabel("X2", fontsize=18)
        ax.set_zlabel("X3", fontsize=18)
        plt.show()


    if create_lorenz96:
        # Generate the data
        data_fname = 'lorenz96_data.pkl'
        x0 = np.array([8.1, 8., 8., 8., 8.])
        data = data_builder(150, 3, x0, 30., 0.05, lorenz96)
        pickle.dump(data, open(data_fname, 'wb'))