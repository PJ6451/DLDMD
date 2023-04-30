import numpy as np
import matplotlib.pyplot as plt
from numerical_solvers import *
from dmd import *

def make_plots(rawdata, cm_data, evals, lbl, Phi = None, Psi = None, amps = None, tt = None, x = None, plt_modes = False):
    fig = plt.figure()

    #plot cm_data with recon
    ax = fig.add_subplot(111)
    for i in range(10):
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
    ax.scatter(np.real(evals[0,:]), np.imag(evals[0,:]),alpha = 0.25)
    ax.set_xlabel("Real$(\lambda)$")
    ax.set_ylabel("Imag$(\lambda)$")
    ax.set_title("Eigenvalues")
    fig.savefig("dmd_eig_"+lbl, dpi=200)
    print(lbl + ':  (' + str(round(rawdata[0,0,0],2)) + ', ' + str(round(rawdata[0,1,0],2)) + ')')
    
    if plt_modes == True:
        #modes
        fig, ax = plt.subplots(1)
        norms = np.zeros(Phi.shape[2])
        for i in range(Phi.shape[2]):
            norms[i] = np.linalg.norm(Phi[0,:,i])
        ind = np.argsort(-norms,kind='quicksort')
        Phi = Phi[0,:,ind].T
        amps = amps[0,:,0]
        ind = np.argsort(-amps,kind='quicksort')
        amps = amps[ind]
        x = np.arange(1,len(x)+1)
        ax.plot(x, np.log10(np.abs(Phi[0,:])))
        ax.plot(x, np.log10(np.abs(Phi[1,:])))
        ax.plot(x, np.log10(np.abs(amps)))
        ax.set_title('Modes and Amplitudes')
        ax.set_xlabel('$Index$')
        ax.set_ylabel('$\log_{10}(||)$')
        ax.legend(['$x_1$','$x_2$','$a_m$'])
        fig.tight_layout()
        fig.savefig("dmd_modes_"+lbl, dpi=200)
        #dynamics
        fig, ax = plt.subplots(1)
        norms = np.zeros(Psi.shape[0])
        for i in range(Psi.shape[0]):
            norms[i] = np.linalg.norm(Psi[i,:])
        ind = np.argsort(-norms,kind='quicksort')
        Psi = Psi[ind,:]
        for dynamic in Psi[:2,:]:
            ax.plot(tt, dynamic.real)
            ax.set_title('Dynamics')
        ax.legend(['$1^{st}$','$2^{nd}$'])
        ax.set_xlabel('Time')
        ax.set_ylabel('$Re(\mathbb{V}_m^{-1}A)$')
        fig.tight_layout()
        fig.savefig("dmd_dynamics_"+lbl, dpi=200)

def make_plots_stacked(rawdata, cm_data, evals, lbl, Phi = None, Psi = None, amps = None, tt = None, x = None, plt_modes = False):
    fig = plt.figure()

    #plot reconstruction
    ax = fig.add_subplot(111)
    for i in range(1):
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

    if plt_modes == True:
        #modes
        fig, ax = plt.subplots(1)
        ax.plot(x, np.log10(np.abs(Phi[0,:])))
        ax.plot(x, np.log10(np.abs(amps)))
        ax.set_title('Modes')
        ax.set_xlabel('$Index$')
        ax.set_ylabel('$\log(|K_M|)$')
        ax.legend(['$1^{st}$','$2^{nd}$'])
        fig.tight_layout()
        fig.savefig("dmd_modes_"+lbl, dpi=200)
        #dynamics
        fig, ax = plt.subplots(1)
        for dynamic in Psi[:3,:]:
            ax.plot(tt, dynamic.real)
            ax.set_title('First three Dynamics')
        ax.legend(['$1^{st}$','$2^{nd}$','$3^{rd}$'])
        ax.set_xlabel('Time')
        ax.set_ylabel('$Re(\mathbb{V}_m^{-1}A)$')
        fig.tight_layout()
        fig.savefig("dmd_dynamics_"+lbl, dpi=200)

def cent():
    dt = .05
    t0 = 0.
    tf = 10.
    NT = int((tf-t0)/dt)
    xvals = np.linspace(-5,5,199)
    tvals = np.linspace(t0,tf,NT)
    numiconds = 60
    initconds = np.zeros([numiconds,2])
    numdim = 2
    rawdata = np.zeros([numiconds, numdim, NT], dtype=np.float64)
    #rawdata = np.zeros([numdim, NT], dtype=np.float64)
    fhandle = lambda x: center(x)
    for ll in range(numiconds):
        initconds[ll,:] = np.random.uniform(-5,5,2)
        rawdata[ll,:,:] = timestepper(initconds[ll,:], t0, tf, dt, fhandle)

    #cdmd
    cm_recon, evals, amps, Phi = cdmd(rawdata, NT)
    Psi = np.vander(evals[0,:].numpy(), N = NT, increasing=True) * amps[0,:,:]
    make_plots(rawdata, cm_recon, evals, 'Center', Phi=Phi.numpy(), Psi=Psi.numpy(), amps = amps.numpy(), tt=tvals, x=xvals, plt_modes = True)

    #cdmd_stacked
    #cm_recon, evals, Phi, Psi, amps = cdmd_stacked(rawdata, NT)
    #make_plots_stacked(rawdata, cm_recon, evals, 'Center_stacked', Phi=Phi, Psi=Psi, amps = amps, tt=tvals, x=xvals, plt_modes = False)

def spir():
    dt = .05
    t0 = 0.
    tf = 10.
    NT = int((tf-t0)/dt)
    tvals = np.linspace(t0,tf,NT)
    xvals = np.linspace(-5,5,199)
    numiconds = 60
    initconds = np.zeros([numiconds,2])
    numdim = 2
    rawdata = np.zeros([numiconds, numdim, NT], dtype=np.float64)
    fhandle = lambda x: spiral(x)
    for ll in range(numiconds):
        initconds[ll,:] = np.random.uniform(-5,5,2)
        rawdata[ll,:,:] = timestepper(initconds[ll,:], t0, tf, dt, fhandle)

    #cdmd
    cm_recon, evals, amps, Phi = cdmd(rawdata, NT)
    Psi = np.vander(evals[0,:].numpy(), N = NT, increasing=True) * amps[0,:,:]
    make_plots(rawdata, cm_recon, evals, 'Spiral_Attractor', Phi=Phi.numpy(), Psi=Psi.numpy(), amps = amps.numpy(), tt=tvals, x=xvals, plt_modes = True)

    #cdmd_stacked
    #cm_recon, evals, Phi, Psi, amps = cdmd_stacked(rawdata, NT)
    #make_plots_stacked(rawdata, cm_recon, evals, 'Spiral_Attractor_stacked', Phi=Phi, Psi=Psi, amps = amps, tt=tvals, x=xvals, plt_modes = False)

def harm():
    dt = .01
    t0 = 0.
    tf = 3
    NT = int((tf-t0)/dt)
    tvals = np.linspace(t0,tf,NT)
    numiconds = 60
    xvals = np.linspace(-3,3,299)
    numdim = 2
    initconds = np.zeros((numiconds,numdim), dtype=np.float64)
    rawdata = np.zeros([numiconds, numdim, NT], dtype=np.float64)
    fhandle = lambda x: harmonic(x)
    for ll in range(numiconds):
        initconds[ll,:] = np.random.uniform(-3.,3.,numdim)
        rawdata[ll,:,:] = timestepper(initconds[ll,:], t0, tf, dt, fhandle)

    #cdmd
    cm_recon, evals, amps, Phi = cdmd(rawdata, NT)
    Psi = np.vander(evals[0,:].numpy(), N = NT, increasing=True) * amps[0,:,:]
    make_plots(rawdata, cm_recon, evals, 'Harm_Osc', Phi=Phi.numpy(), Psi=Psi.numpy(), amps = amps.numpy(), tt=tvals, x=xvals, plt_modes = True)

    #cdmd_stacked
    #cm_recon, evals, Phi, Psi, amps = cdmd_stacked(rawdata, NT)
    #make_plots_stacked(rawdata, cm_recon, evals, 'Harm_Osc_stacked_foc', Phi=Phi, Psi=Psi, amps = amps, tt=tvals, x=xvals, plt_modes = False)

def vdp(mu, tf, dt, lbl):
    t0 = 0.
    NT = int(tf/dt)
    tvals = np.linspace(t0,tf,NT)
    xvals = np.linspace(-1,1,NT-1)
    numdim = 2
    fhandle = lambda x: vanderpol(x,mu)
    numiconds = 2
    initconds = np.zeros((numiconds,2), dtype=np.float64)
    rawdata = np.zeros([numiconds, numdim, NT], dtype=np.float64)
    for ll in range(numiconds):
        initconds[ll,:] = np.random.uniform(-1.,1.,2)
        rawdata[ll] = timestepper(initconds[ll,:], t0, tf, dt, fhandle)

    #cdmd
    #cm_recon, evals, _, _ = cdmd(rawdata, NT)
    #Psi = np.vander(evals[0,:].numpy(), N = NT, increasing=True) * amps[0,:,:]
    #make_plots(rawdata, cm_recon, evals, lbl)

    #cdmd_stacked
    cm_recon, evals, _, _, _ = cdmd_stacked(rawdata, NT)
    make_plots_stacked(rawdata, cm_recon, evals, lbl)

def lorenz():
    # Generate the data
    numiconds = 60
    numdim = 3
    t0 = 0
    tf = 30.
    dt = 0.05
    NT = int(tf/dt)
    tvals = np.linspace(t0,tf,NT)
    x0 = np.array([3.1, 3.1, 27.])
    rawdata = data_builder(numiconds, numdim, x0, tf, dt, lorenz63)
    
    #dmd
    #cm_data, evals, _, _ = cdmd(rawdata, NT)
    #make_plots_3D(rawdata, cm_data, evals, 'l63')

    cm_data, evals, _, _, _ = cdmd_stacked(rawdata, NT)
    make_plots_3D(rawdata, cm_data, evals, 'l63_stacked')

def ross():
    # Generate the data
    numiconds = 2
    numdim = 3
    t0 = 0
    tf = 30.
    dt = 0.05
    NT = int(tf/dt)
    tvals = np.linspace(t0,tf,NT)
    x0 = np.array([.1, .1, .1])
    rawdata = data_builder_ross(numiconds, numdim, x0, tf, dt, rossler)
    
    #dmd
    cm_data, evals, _, _ = cdmd(rawdata, NT)
    make_plots_3D(rawdata, cm_data, evals, 'ross')

    #cm_data, evals, _, _, _ = cdmd_stacked(rawdata, NT)
    #make_plots_3D(rawdata, cm_data, evals, 'ross_stacked')

def lorenz_96():
    # Generate the data
    numiconds = 60
    numdim = 5
    t0 = 0
    tf = 30.
    dt = 0.05
    NT = int(tf/dt)
    tvals = np.linspace(t0,tf,NT)
    x0 = np.array([8.1, 8., 8., 8., 8.])
    rawdata = data_builder(numiconds, numdim, x0, tf, dt, lorenz96)
    
    #dmd
    cm_data, evals, Phi, Psi = cdmd(rawdata, NT)
    make_plots_3D(rawdata, cm_data, evals, 'l96')

    #cm_data, evals, _, _, _ = cdmd_stacked(rawdata, NT)
    #make_plots_3D(rawdata, cm_data, evals, 'l96_stacked')

def make_plots_3D(rawdata,cm_data,evals,hdle):
    fig = plt.figure()
    #DMD
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(rawdata[0,0,:], rawdata[0,1,:], rawdata[0,2,:], linewidth=1,label='RK4')
    ax.plot3D(cm_data[0,0,:], cm_data[0,1,:], cm_data[0,2,:], linewidth=1,label='CDMD')
    ax.legend()
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    fig.savefig("dmd_recon_" + hdle, dpi=200)
    #Eigenvalues
    fig = plt.figure()
    ax = fig.add_subplot(111)
    t = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(t), np.sin(t), linewidth=1)
    ax.scatter(np.real(evals[0,:]), np.imag(evals[0,:]),alpha = 0.25)
    ax.set_xlabel("Real$(\lambda)$")
    ax.set_ylabel("Imag$(\lambda)$")
    ax.set_title("Eigenvalues")
    fig.savefig("dmd_eig_"+hdle, dpi=200)

if __name__ == '__main__':
    #cent()
    #spir()
    harm()
    #vdp(0.1, 50, 0.05, 'VDP_mu_1e-1')
    #vdp(1., 20, 0.05, 'VDP_mu_1')
    #vdp(1., 20, 0.025, 'VDP_mu_1_dt_small')
    #vdp(10., 20, 0.05, 'VDP_mu_10')
    #lorenz()
    #ross()
    #lorenz_96()