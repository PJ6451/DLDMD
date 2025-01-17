"""
    Author:
        Jay Lago, NIWC/SDSU, 2021
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
import numpy as np


class DLDMD(keras.Model):
    def __init__(self, hyp_params, **kwargs):
        super(DLDMD, self).__init__(**kwargs)

        # Parameters
        self.batch_size = hyp_params['batch_size']
        self.phys_dim = hyp_params['phys_dim']
        self.latent_dim = hyp_params['latent_dim']
        self.num_time_steps = int(hyp_params['num_time_steps'])
        self.num_pred_steps = int(hyp_params['num_pred_steps'])
        self.num_recon_steps = int(hyp_params['num_recon_steps'])
        self.time_final = hyp_params['time_final']
        self.num_en_layers = hyp_params['num_en_layers']
        self.num_neurons = hyp_params['num_en_neurons']
        self.delta_t = hyp_params['delta_t']
        self.enc_input = (self.num_time_steps, self.phys_dim)
        self.dec_input = (self.num_time_steps, self.latent_dim)
        self.precision = hyp_params['precision']
        if self.precision == 'float32':
            self.precision_complex = tf.complex64
        else:
            self.precision_complex = tf.complex128

        # Construct the ENCODER network
        self.encoder = keras.Sequential(name="encoder")
        self.encoder.add(Dense(self.num_neurons,
                               input_shape=self.enc_input,
                               activation=hyp_params['hidden_activation'],
                               kernel_initializer=hyp_params['kernel_init_enc'],
                               bias_initializer=hyp_params['bias_initializer'],
                               trainable=True, name='enc_in'))
        for ii in range(self.num_en_layers):
            self.encoder.add(Dense(self.num_neurons,
                                   activation=hyp_params['hidden_activation'],
                                   kernel_initializer=hyp_params['kernel_init_enc'],
                                   bias_initializer=hyp_params['bias_initializer'],
                                   trainable=True, name='enc_' + str(ii)))
        self.encoder.add(Dense(self.latent_dim,
                               activation=hyp_params['ae_output_activation'],
                               kernel_initializer=hyp_params['kernel_init_enc'],
                               bias_initializer=hyp_params['bias_initializer'],
                               trainable=True, name='enc_out'))

        # Construct the DECODER network
        self.decoder = keras.Sequential(name="decoder")
        self.decoder.add(Dense(self.num_neurons,
                               input_shape=self.dec_input,
                               activation=hyp_params['hidden_activation'],
                               kernel_initializer=hyp_params['kernel_init_enc'],
                               bias_initializer=hyp_params['bias_initializer'],
                               trainable=True, name='dec_in'))
        for ii in range(self.num_en_layers):
            self.decoder.add(Dense(self.num_neurons,
                                   activation=hyp_params['hidden_activation'],
                                   kernel_initializer=hyp_params['kernel_init_dec'],
                                   bias_initializer=hyp_params['bias_initializer'],
                                   trainable=True, name='dec_' + str(ii)))
        self.decoder.add(Dense(self.phys_dim,
                               activation=hyp_params['ae_output_activation'],
                               kernel_initializer=hyp_params['kernel_init_dec'],
                               bias_initializer=hyp_params['bias_initializer'],
                               trainable=True, name='dec_out'))

    def call(self, x):
        # Encode the entire time series
        y = self.encoder(x)
        x_ae = self.decoder(y)

        # Reshape for DMD step
        yt = tf.transpose(y, [0, 2, 1])

        # Generate latent time series using DMD prediction
        y_adv, evals, evecs, modes = self.cdmd_stacked(yt[:,:,:self.num_recon_steps])

        # Decode the latent trajectories
        x_adv = self.decoder(y_adv)

        # Model weights
        weights = self.trainable_weights

        return [y, x_ae, x_adv, y_adv, weights, evals, evecs, modes]

    def edmd(self, Y):
        Y_m = Y[:, :, :-1]
        Y_p = Y[:, :, 1:]

        sig, U, V = tf.linalg.svd(Y_m, compute_uv=True, full_matrices=False)
        sigr_inv = tf.linalg.diag(1.0 / sig)
        Uh = tf.linalg.adjoint(U)

        A = Y_p @ V @ sigr_inv @ Uh
        evals, evecs = tf.linalg.eig(A)
        phi = tf.linalg.solve(evecs, tf.cast(Y_m, dtype=self.precision_complex))
        y0 = phi[:, :, 0]
        y0 = y0[:, :, tf.newaxis]

        recon = tf.TensorArray(self.precision_complex, size=self.num_pred_steps)
        recon = recon.write(0, evecs @ y0)
        evals_k = tf.identity(evals)
        for ii in tf.range(1, self.num_pred_steps):
            tmp = evecs @ (tf.linalg.diag(evals_k) @ y0)
            recon = recon.write(ii, tmp)
            evals_k = evals_k * evals
        recon = tf.math.real(tf.transpose(tf.squeeze(recon.stack()), perm=[1, 0, 2]))
        return recon, evals, evecs, phi

    def cdmd(self, data):
        # assign values for regression
        x_0 = data[:, :, 0]
        y = data[:, :, -1]
        X = data[:, :, :-1]

        #svd and building companion matrix
        sig, U, V = tf.linalg.svd(X, compute_uv=True, full_matrices=False)
        sigr_inv = tf.linalg.diag(1.0 / sig)
        Uh = tf.linalg.adjoint(U)
        comp_mat = np.array(data.shape[0]*[np.diag(np.array([1.]*(self.num_recon_steps-2)), k = -1)])

        c = V @ sigr_inv @ Uh @ y[...,None]
        comp_mat[:, :, -1] = c[:,:,0]
        comp_mat = tf.convert_to_tensor(comp_mat)

        #calculating eigenvalues/vectors/modes
        evals, evecs = tf.linalg.eig(comp_mat)
        Phi = tf.cast(X, dtype=self.precision_complex) @ evecs
        amps = np.linalg.pinv(Phi.numpy()) @ x_0[...,None]
        amps = tf.cast(amps,dtype=self.precision_complex)

        recon = tf.TensorArray(self.precision_complex, size=self.num_pred_steps)
        recon = recon.write(0, Phi @ amps)
        evals_k = tf.identity(evals)
        for ii in tf.range(1, self.num_pred_steps):
            tmp = Phi @ (tf.linalg.diag(evals_k) @ amps)
            recon = recon.write(ii, tmp)
            evals_k = evals_k * evals
        recon = tf.math.real(tf.transpose(tf.squeeze(recon.stack()), perm=[1, 0, 2]))
        return recon, evals, evecs, Phi
    
    def cdmd_stacked(self, tfdata):
        # transfer to numpy
        data = tfdata.numpy()

        # stack data
        stacked_data = np.row_stack(data)

        # assign values for auto-regression
        y0 = stacked_data[:,0]
        y = stacked_data[:,-1]
        X = stacked_data[:,:-1]

        # compute svd and build companion matrix
        u, s, vh = np.linalg.svd(X, full_matrices=False)

        comp_mat = np.diag(np.array([1.]*(self.num_recon_steps-2)), k = -1)
        comp_mat[:,-1] = np.conj(vh.T) @ np.diag(1. / s) @ np.conj(u.T) @ y

        # calculate eigenvalues/vectors/modes
        evals, evecs = np.linalg.eig(comp_mat)
        modes = X.dot(evecs)
        amps = np.linalg.pinv(modes) @ y0
        
        # Reconstruction, forecasting
        Psi = np.vander(evals, N = self.num_pred_steps, increasing=True) * amps[...,None]
        recon = modes.dot(Psi)

        # unstack data
        unstacked_data = np.resize(recon, data.shape)

        recon = tf.convert_to_tensor(unstacked_data, dtype = 'float64')

        return recon, evals, evecs, modes

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                'encoder': self.encoder,
                'decoder': self.decoder}
