import numpy as np
import matplotlib.pyplot as plt
from numerical_solvers import *
from dmd import *

def make_plots(rawdata, cm_data, evals, lbl, Phi = None, Psi = None, tt = None, x = None):
    fig = plt.figure()

    #plot cm_data with recon
    ax = fig.add_subplot(111)
    for i in range(20):
        ax.plot(rawdata[i,0,:], rawdata[i,1,:],linewidth=2,linestyle='-',color='b')
        ax.plot(cm_data[i,0,:], cm_data[i,1,:],linewidth=2,linestyle='dashed', color='r')
    ax.plot(rawdata[i,0,:], rawdata[i,1,:],linewidth=2,linestyle='-',color='b',label='RK4')
    ax.plot(cm_data[i,0,:], cm_data[i,1,:],linewidth=2,linestyle='dashed', color='r',label='CM')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    title = 'Phase Plane'
    plt.title(title)
    ax.legend()

    fig.savefig("dmd_project_" + lbl, dpi=200)

    #Eigenvalues
    fig = plt.figure()
    ax = fig.add_subplot(111)
    t = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(t), np.sin(t), linewidth=1)
    ax.scatter(np.real(evals), np.imag(evals),alpha = 0.25)
    ax.set_xlabel("Real$(\lambda)$")
    ax.set_ylabel("Imag$(\lambda)$")
    ax.set_title("Eigenvalues")
    fig.savefig("dmd_eig_"+lbl, dpi=200)

    if Psi.any():
        #modes
        fig, ([ax1,ax2]) = plt.subplots(1,2)
        ax1.plot(x, Phi.real[:60,:3])
        ax2.plot(x, Phi.real[60:,:3])
        ax1.set_title('Modes in $x_1$')
        ax1.set_xlabel('$x_1$')
        ax2.set_title('Modes in $x_2$')
        ax2.set_xlabel('$x_2$')
        ax1.legend(['$1^{st}$','$2^{nd}$','$3^{rd}$'])
        ax2.legend(['$1^{st}$','$2^{nd}$','$3^{rd}$'])
        fig.tight_layout()
        fig.savefig("dmd_modes_"+lbl, dpi=200)
        #dynamics
        fig, ax = plt.subplots(1)
        for dynamic in Psi[:3,:]:
            ax.plot(tt, dynamic.real)
            ax.set_title('Dynamics')
        ax.legend(['$1^{st}$','$2^{nd}$','$3^{rd}$'])
        ax.set_xlabel('t')
        fig.tight_layout()
        fig.savefig("dmd_dynamics_"+lbl, dpi=200)

def raw_proc_data(rawdata, numiconds, numdim, NT, tvals, dt):
    #stack data
    stacked_data = np.zeros([numiconds*numdim, NT])
    for i in range(numdim):
        stacked_data[(i)*numiconds:(i+1)*numiconds,:] = rawdata[:,i,:]

    #dmd
    thrshhld = 15
    cm_recon, evals, Phi, Psi = cm_dmd(stacked_data, NT, thrshhld)

    #unstack data
    cm_data = np.zeros([numiconds, numdim, NT])
    for i in range(numdim):
        cm_data[:,i,:] = np.real(cm_recon[(i)*numiconds:(i+1)*numiconds,:])

    return cm_data, evals, Phi, Psi

def cent():
    dt = .05
    t0 = 0.
    tf = 10.
    NT = int((tf-t0)/dt)
    xvals = np.linspace(-5,5,60)
    tvals = np.linspace(t0,tf,NT)
    numiconds = 60
    initconds = np.zeros([numiconds,2])
    numdim = 2
    rawdata = np.zeros([numiconds, numdim, NT], dtype=np.float64)
    fhandle = lambda x: center(x)
    for ll in range(numiconds):
        initconds[ll,:] = np.random.uniform(-5,5,2)
        rawdata[ll,:,:] = timestepper(initconds[ll,:], t0, tf, dt, fhandle)

    #dmd
    cm_data, evals, Phi, Psi = raw_proc_data(rawdata, numiconds, numdim, NT, tvals, dt)

    #plot
    make_plots(rawdata, cm_data, evals, 'Center', Phi=Phi, Psi=Psi, tt=tvals, x=xvals)

def spir():
    dt = .05
    t0 = 0.
    tf = 10.
    NT = int((tf-t0)/dt)
    tvals = np.linspace(t0,tf,NT)
    xvals = np.linspace(-5,5,60)
    numiconds = 60
    initconds = np.zeros([numiconds,2])
    numdim = 2
    rawdata = np.zeros([numiconds, numdim, NT], dtype=np.float64)
    fhandle = lambda x: spiral(x)
    for ll in range(numiconds):
        initconds[ll,:] = np.random.uniform(-5,5,2)
        rawdata[ll,:,:] = timestepper(initconds[ll,:], t0, tf, dt, fhandle)

    #dmd
    cm_data, evals, Phi, Psi = raw_proc_data(rawdata, numiconds, numdim, NT, tvals, dt)

    #plot
    make_plots(rawdata, cm_data, evals, 'Spiral_Attractor', Phi=Phi, Psi=Psi, tt=tvals, x=xvals)

def harm():
    dt = .01
    t0 = 0.
    tf = 3
    NT = int((tf-t0)/dt)
    tvals = np.linspace(t0,tf,NT)
    numiconds = 60
    xvals = np.linspace(-3,3,60)
    numdim = 2
    initconds = np.zeros((numiconds,numdim), dtype=np.float64)
    rawdata = np.zeros([numiconds, numdim, NT], dtype=np.float64)
    fhandle = lambda x: harmonic(x)
    for ll in range(numiconds):
        initconds[ll,:] = np.random.uniform(-3.,3.,numdim)
        rawdata[ll,:,:] = timestepper(initconds[ll,:], t0, tf, dt, fhandle)

    #dmd
    cm_data, evals,Phi, Psi = raw_proc_data(rawdata, numiconds, numdim, NT, tvals, dt)

    #plot
    make_plots(rawdata, cm_data, evals, 'Harmonic_Oscilator',Phi=Phi, Psi=Psi, tt=tvals, x=xvals)

def duff():
    dt = .025
    t0 = 0.
    tf = 20.
    NT = int(tf/dt)
    tvals = np.linspace(t0,tf,NT)
    numdim = 2
    fhandle = lambda x: duffing(x)
    numiconds = 10
    initconds = np.zeros((numiconds, numdim), dtype=np.float64)
    rawdata = np.zeros([numiconds, numdim, NT], dtype=np.float64)
    for ll in range(numiconds):
        initconds[ll,:] = 1.5*np.random.uniform(-1.,1.,numdim) - .5
        rawdata[ll] = timestepper(initconds[ll,:], t0, tf, dt, fhandle)

    for ii in range(rawdata.shape[0]):
        plt.plot(rawdata[ii, 0, :], rawdata[ii, 1, :], '-')
    plt.show()

    #dmd
    cm_data, evals,_,_ = raw_proc_data(rawdata, numiconds, numdim, NT, tvals, dt)

    #plot
    make_plots(rawdata, cm_data, evals, 4, 'Duffing_Oscilator')

def vdp():
    mu = .1
    dt = .05
    t0 = 0.
    tf = 50.
    NT = int(tf/dt)
    tvals = np.linspace(t0,tf,NT)
    numdim = 2
    fhandle = lambda x: vanderpol(x,mu)
    numiconds = 10
    initconds = np.zeros((numiconds,2), dtype=np.float64)
    rawdata = np.zeros([numiconds, numdim, NT], dtype=np.float64)
    for ll in range(numiconds):
        initconds[ll,:] = np.random.uniform(-1.,1.,2)
        rawdata[ll] = timestepper(initconds[ll,:], t0, tf, dt, fhandle)

    #dmd
    cm_data, evals,_,_ = raw_proc_data(rawdata, numiconds, numdim, NT, tvals, dt)

    #plot
    make_plots(rawdata, cm_data, evals, 'VDP_Oscilator_mu_small_tf_big')

def lorenz():
    # Generate the data
    numiconds = 1
    numdim = 3
    t0 = 0
    tf = 30.
    dt = 0.05
    NT = int(tf/dt)
    tvals = np.linspace(t0,tf,NT)
    x0 = np.array([3.1, 3.1, 27.])
    rawdata = data_builder(numiconds, numdim, x0, tf, dt, lorenz63)
    
    #dmd
    cm_data, evals,_,_ = raw_proc_data(rawdata, numiconds, numdim, NT, tvals, dt)

    make_plots_3D(rawdata, cm_data, evals, 'l63')

    fig = plt.figure()
    traj = rawdata[0,:,:]
    #RK4
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(traj[0,:], traj[1,:], traj[2,:],linewidth=1,label='rk4')
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.set_title("Phase Plane")
    fig.savefig("l63", dpi=200)

def ross():
    # Generate the data
    numiconds = 1
    numdim = 3
    t0 = 0
    tf = 30.
    dt = 0.05
    NT = int(tf/dt)
    tvals = np.linspace(t0,tf,NT)
    x0 = np.array([.1, .1, .1])
    rawdata = data_builder_ross(numiconds, numdim, x0, tf, dt, rossler)
    
    #dmd
    #cm_data, evals,_,_  = raw_proc_data(rawdata, numiconds, numdim, NT, tvals, dt)

    #make_plots_3D(rawdata, cm_data, evals, 'ross')

    fig = plt.figure()
    traj = rawdata[0,:,:]
    #RK4
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(traj[0,:], traj[1,:], traj[2,:],linewidth=1,label='rk4')
    ax.set_xlabel("$y_1$")
    ax.set_ylabel("$y_2$")
    ax.set_zlabel("$y_3$")
    ax.set_title("Phase Plane")
    fig.savefig("rossler", dpi=200)

def lorenz_96():
    # Generate the data
    numiconds = 80
    numdim = 5
    t0 = 0
    tf = 30.
    dt = 0.05
    NT = int(tf/dt)
    tvals = np.linspace(t0,tf,NT)
    x0 = np.array([8.1, 8., 8., 8., 8.])
    rawdata = data_builder(numiconds, numdim, x0, tf, dt, lorenz96)
    
    #dmd
    cm_data, evals,_,_ = raw_proc_data(rawdata, numiconds, numdim, NT, tvals, dt)

    make_plots_3D(rawdata, cm_data, evals, 'l96')

def make_plots_3D(rawdata,cm_data,evals,hdle):
    fig = plt.figure()
    traj = rawdata[0,:,:]
    #RK4
    ax = fig.add_subplot(121, projection='3d')
    ax.plot3D(traj[0,:], traj[1,:], traj[2,:],linewidth=1,label='rk4')
    ax.legend()
    #DMD
    ax = fig.add_subplot(122, projection='3d')
    ax.plot3D(cm_data[0,0,:], cm_data[0,1,:], cm_data[0,2,:], linewidth=1,label='CDMD')
    ax.legend()
    fig.savefig("dmd_recon_" + hdle, dpi=200)
    #Eigenvalues
    fig = plt.figure()
    ax = fig.add_subplot(111)
    t = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(t), np.sin(t), linewidth=1)
    ax.scatter(np.real(evals), np.imag(evals),alpha = 0.25)
    ax.set_xlabel("Real$(\lambda)$")
    ax.set_ylabel("Imag$(\lambda)$")
    ax.set_title("Eigenvalues")
    fig.savefig("dmd_eig_"+hdle, dpi=200)

if __name__ == '__main__':
    cent()
    spir()
    harm()
    #duff()
    #vdp()
    #lorenz()
    #ross()
    #lorenz_96()