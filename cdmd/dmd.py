import numpy as np

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
    uh = np.zeros([u.shape[0],u.shape[2],u.shape[1]])
    v = np.zeros([vh.shape[0],vh.shape[2],vh.shape[1]])
    sr = np.zeros([s.shape[0],s.shape[1],s.shape[1]])
    comp_mat = np.zeros([data.shape[0],X.shape[2],X.shape[2]])
    for i in range(data.shape[0]):
        uh[i,:,:] = u[i,:,:].conj().T
        v[i,:,:] = vh[i,:,:].conj().T
        sr[i,:,:] = np.diag(1./s[i,:])
        comp_mat[i,:,:] = np.diag(np.array([1.]*(NT-2)), k = -1)

    c = v @ sr @ uh @ y[...,None]
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