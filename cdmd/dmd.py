import numpy as np
import tensorflow as tf

def dmd(data, NT, thrshhld):
    x_0 = data[:,0]
    X = data[:,:-1]
    Y = data[:,1:]

    u, s ,vh = np.linalg.svd(X, full_matrices = False)
    sm = np.max(s)
    indskp = np.log10(s / sm) > -thrshhld
    sr = s[indskp]
    ur = u[:, indskp]
    v = np.conj(vh.T)
    vr = v[:, indskp]
    
    #Solve for Operator
    K = np.conj(ur.T) @ Y @ vr @ np.diag(1./sr)

    #Eigen decomp
    evals, evecs = np.linalg.eig(K)

    #modes
    Phi = ur.dot(evecs)
    amps = np.linalg.pinv(Phi) @ x_0

    #reconstruction
    Psi = np.vander(evals, N = NT, increasing=True) * amps[...,None]

    recon = Phi.dot(Psi)

    return recon

def cm_dmd(data, NT):
    # assign values for regression
    x_0 = data[:, :, 0]
    y = data[:, :, -1]
    X = data[:, :, :-1]

    #svd and building companion matrix
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    comp_mat = np.array(data.shape[0]*[np.diag(np.array([1.]*(NT-2)), k = -1)])

    sigr_inv = tf.linalg.diag(1.0 / s)
    Uh = tf.linalg.adjoint(u)
    V = tf.linalg.adjoint(vh)

    c = V @ sigr_inv @ Uh @ y[...,None]
    comp_mat[:, :, -1] = c[:,:,0]

    #calculating eigenvalues/vectors/modes
    evals, evecs = np.linalg.eig(comp_mat)
    Phi = X @ evecs
    amps = np.linalg.pinv(Phi) @ x_0[...,None]

    #reconstruction
    Psi = np.zeros((evals.shape[0],evals.shape[1],NT),dtype='complex64')
    for i in range(evals.shape[0]):
        Psi[i,:,:] = np.vander(evals[i,:], N = NT, increasing=True) * amps[i,:]

    recon = Phi @ Psi

    return recon.real, evals, Phi, Psi