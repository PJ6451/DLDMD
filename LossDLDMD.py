"""
    Author:
        Jay Lago, NIWC/SDSU, 2021
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import MSE
import numpy as np


class LossDLDMD(keras.losses.Loss):
    def __init__(self, hyp_params, **kwargs):
        super(LossDLDMD, self).__init__(**kwargs)

        # Parameters
        self.a1 = hyp_params['a1']
        self.a2 = hyp_params['a2']
        self.a3 = hyp_params['a3']
        self.a4 = hyp_params['a4']
        self.precision = hyp_params['precision']

        # Loss components
        self.loss_recon = tf.constant(0.0, dtype=self.precision)
        self.loss_pred = tf.constant(0.0, dtype=self.precision)
        self.loss_dmd = tf.constant(0.0, dtype=self.precision)
        self.loss_reg = tf.constant(0.0, dtype=self.precision)
        self.total_loss = tf.constant(0.0, dtype=self.precision)
        self.num_recon_steps = int(hyp_params['num_recon_steps'])

    def call(self, model, obs):
        """
            model = [y, x_ae, x_adv, y_adv, weights, evals, evecs, phi]
        """
        y = tf.identity(model[0])
        x_ae = tf.identity(model[1])
        x_adv = tf.identity(model[2])
        y_adv = tf.identity(model[3])
        weights = model[4]

        # Autoencoder reconstruction
        self.loss_recon = tf.reduce_mean(MSE(obs, x_ae))

        # DMD reconstruction in the latent space
        self.loss_dmd = self.stacked_cdmd_loss(y)

        # Future state prediction
        self.loss_pred = tf.reduce_mean(MSE(obs, x_adv))

        # Regularization on weights
        self.loss_reg = tf.add_n([tf.nn.l2_loss(w) for w in weights])

        # Total loss
        self.total_loss = self.a1 * self.loss_recon + self.a2 * self.loss_dmd + \
                          self.a3 * self.loss_pred + self.a4 * self.loss_reg

        return self.total_loss

    @tf.function
    def dmdloss(self, y):
        y_m = tf.transpose(y, perm=[0, 2, 1])[:, :, :-1]
        y_p = tf.transpose(y, perm=[0, 2, 1])[:, :, 1:]
        [_, _, V] = tf.linalg.svd(y_m, compute_uv=True, full_matrices=False)
        VVh = V @ tf.linalg.adjoint(V)
        eye_mat = tf.eye(VVh.shape[-1], batch_shape=[VVh.shape[0]], dtype=self.precision)
        return tf.reduce_mean(tf.norm(y_p @ (eye_mat - VVh), ord='fro', axis=[-2, -1]))

    def stacked_cdmd_loss(self, y):
        Y = np.row_stack(y.numpy())
        u, s, vh = np.linalg.svd(Y[:,self.num_recon_steps,:], full_matrices=False)
        f_nr = Y[:,self.num_recon_steps-1,:]
        loss = (Y @ np.conj(vh.T) @ np.diag(1. / s) @ np.conj(u.T) - np.eye(Y.shape[0])) * f_nr
        return tf.reduce_mean(np.linalg.norm(loss, ord='fro'))
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                'loss_recon': self.loss_recon,
                'loss_pred': self.loss_pred,
                'loss_dmd': self.loss_dmd,
                'loss_inf': self.loss_inf,
                'loss_reg': self.loss_reg,
                'total_loss': self.total_loss}
