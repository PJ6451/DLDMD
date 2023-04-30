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

def cdmd(data, NT):
    # assign values for regression
    x_0 = data[:, :, 0]
    y = data[:, :, -1]
    X = data[:, :, :-1]

    #svd and building companion matrix
    sig, U, V = tf.linalg.svd(X, compute_uv=True, full_matrices=False)
    sigr_inv = tf.linalg.diag(1.0 / sig)
    Uh = tf.linalg.adjoint(U)
    comp_mat = np.array(data.shape[0]*[np.diag(np.array([1.]*(NT-2)), k = -1)])
    c = V @ sigr_inv @ Uh @ y[...,None]
    comp_mat[:, :, -1] = c[:,:,0]
    comp_mat = tf.convert_to_tensor(comp_mat)

    #calculating eigenvalues/vectors/modes
    evals, evecs = tf.linalg.eig(comp_mat)
    Phi = tf.cast(X, dtype='complex128') @ evecs
    amps = np.linalg.pinv(Phi.numpy()) @ x_0[...,None]
    amps = tf.cast(amps,dtype='complex128')

    recon = tf.TensorArray('complex128', size=NT)
    recon = recon.write(0, Phi @ amps)
    evals_k = tf.identity(evals)
    for ii in tf.range(1, NT):
        tmp = Phi @ (tf.linalg.diag(evals_k) @ amps)
        recon = recon.write(ii, tmp)
        evals_k = evals_k * evals
    recon = tf.math.real(tf.transpose(tf.squeeze(recon.stack()), perm=[1, 2, 0]))
    return recon, evals, amps, Phi

def cdmd_stacked(data, NT):
    #stack data
    stacked_data = np.zeros([data.shape[0]*data.shape[1], NT])
    for i in range(data.shape[1]):
        stacked_data[(i)*data.shape[0]:(i+1)*data.shape[0],:] = data[:,i,:]
    
    # assign values for regression
    x_0 = stacked_data[:,0]
    y = stacked_data[:,-1]
    X = stacked_data[:,:-1]

    #svd and building companion matrix
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    sm = np.max(s)
    indskp = np.log10(s / sm) > -16
    sr = s[indskp]
    ur = u[:, indskp]
    v = np.conj(vh.T)
    vr = v[:, indskp]
        
    c = vr @ np.diag(1. / sr) @ np.conj(ur.T) @ y
    comp_mat = np.diag(np.array([1.]*(NT-2)), k = -1)
    comp_mat[:,-1] = c

    #calculating eigenvalues/vectors/modes
    evals, evecs = np.linalg.eig(comp_mat)
    Phi = X.dot(evecs)
    amps = np.linalg.pinv(Phi) @ x_0

    #reconstruction
    Psi = np.vander(evals, N = NT, increasing=True) * amps[...,None]

    recon = Phi.dot(Psi)

    #unstack data
    cm_data = np.zeros([data.shape[0], data.shape[1], NT])
    for i in range(data.shape[1]):
        cm_data[:,i,:] = np.real(recon[(i)*data.shape[0]:(i+1)*data.shape[0],:])

    return cm_data, evals, Phi, Psi, amps