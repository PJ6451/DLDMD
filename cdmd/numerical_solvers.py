import numpy as np

# ==============================================================================
# Dynamical System Functions
# ==============================================================================
def spiral(lhs):
    x1, x2 = lhs[0], lhs[1]
    rhs = np.zeros(2, dtype=np.float64)
    rhs[0] = -0.2*x1 + x2
    rhs[1] = -x1 - 0.2*x2
    return rhs

def center(lhs):
    x1, x2 = lhs[0], lhs[1]
    rhs = np.zeros(2, dtype=np.float64)
    rhs[0] = 0*x1 - 1*x2
    rhs[1] = 1*x1 - 0*x2
    return rhs

def saddle(lhs):
    x1, x2 = lhs[0], lhs[1]
    rhs = np.zeros(2, dtype=np.float64)
    rhs[0] = x1 + 4*x2
    rhs[1] = 2*x1 - x2
    return rhs

def harmonic(lhs):
    x1, x2 = lhs[0], lhs[1]
    rhs = np.zeros(2, dtype=np.float64)
    rhs[0] = x2
    rhs[1] = -np.sin(x1)
    return rhs

def duffing(lhs):
    x1, x2 = lhs[0], lhs[1]
    rhs = np.zeros(2, dtype=np.float64)
    rhs[0] = x2
    rhs[1] = x1 - x1**3.
    return rhs

def vanderpol(lhs,mu):
    x1, x2 = lhs[0], lhs[1]
    rhs = np.zeros(2, dtype=np.float64)
    rhs[0] = x2
    rhs[1] = mu*(1.-x1**2.)*x2 - x1
    return rhs

def lorenz63(lhs):
    sigma, rval, bval = 10., 28., 8./3.
    y1, y2, y3 = lhs[0], lhs[1], lhs[2]
    rhs = np.zeros(3, dtype=np.float64)
    rhs[0] = sigma*(y2-y1)
    rhs[1] = rval*y1-y2-y1*y3
    rhs[2] = -bval*y3 + y1*y2
    return rhs

def rossler(lhs):
    aval, bval, cval = .1, .1, 14.
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
        initconds[ll,:] = 8.*(np.random.rand(dim) - .5) + x0
        rawdata[ll,:,:] = timestepper(initconds[ll,:], 0, tf, dt, dyn_sys)

    return rawdata

def data_builder_ross(n_ic, dim, x0, tf, dt, dyn_sys):
    nsteps = int(tf / dt)
    n_ic = int(n_ic)
    # Generate initial conditions
    initconds = np.zeros((n_ic, dim), dtype=np.float64)
    rawdata = np.zeros([n_ic, dim, nsteps], dtype=np.float64)
    for ll in range(n_ic):
        initconds[ll,:-1] = 8*np.random.uniform(-1.,1.,2)
        rawdata[ll,:,:] = timestepper(initconds[ll,:], 0, tf, dt, dyn_sys)

    return rawdata