"""
    Author:
        Jay Lago, NIWC/SDSU, 2021
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import numpy as np

font = {'family': 'DejaVu Sans', 'size': 24}
matplotlib.rc('font', **font)


def diagnostic_plot(y_pred, y_true, hyp_params, epoch, save_path, loss_comps, val_loss):
    if hyp_params['experiment'] == 'harmonic':
        plot_2D(y_pred, y_true, hyp_params, epoch, save_path, loss_comps, val_loss)
    elif hyp_params['experiment'] == 'rossler' or \
            hyp_params['experiment'] == 'lorenz96' or \
                hyp_params['experiment'] == 'lorenz':
        plot_3d_latent(y_pred, y_true, hyp_params, epoch, save_path, loss_comps, val_loss)
    else:
        print("[ERROR] unknown experiment, create new diagnostic plots...")


def plot_2D(y_pred, y_true, hyp_params, epoch, save_path, loss_comps, val_loss):
    enc_adv_dec = y_pred[2].numpy()
    evals = y_pred[5]

    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=(40, 20), facecolor='white')
    ax = ax.flat
    skip = 8

    # Encoded-advanced-decoded time series
    for ii in np.arange(0, enc_adv_dec.shape[0], skip):
        ax[0].plot(enc_adv_dec[ii, :, 0], enc_adv_dec[ii, :, 1], '-')
    ax[0].scatter(enc_adv_dec[::skip, 0, 0], enc_adv_dec[::skip, 0, 1])
    ax[0].grid()
    ax[0].set_xlabel("x1")
    ax[0].set_ylabel("x2")
    ax[0].set_title("Encoded-Advanced-Decoded (x_adv))")

    # evals
    t = np.linspace(0, 2*np.pi, 300)
    ax[1].plot(np.cos(t), np.sin(t), linewidth=1)
    ax[1].scatter(np.real(evals[0,:]), np.imag(evals[0,:]))
    ax[1].set_xlabel("Real$(\lambda)$")
    ax[1].set_ylabel("Imag$(\lambda)$")
    ax[1].set_title("Eigenvalues")

    # Loss components
    lw = 3
    loss_comps = np.asarray(loss_comps)
    ax[2].plot(val_loss, color='k', linewidth=lw, label='total')
    ax[2].set_title("Total Loss")
    ax[2].grid()
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("$log_{10}(L)$")
    ax[2].legend(loc="upper right")

    ax[3].plot(loss_comps[:, 0], color='r', linewidth=lw, label='recon')
    ax[3].set_title("Recon Loss")
    ax[3].grid()
    ax[3].set_xlabel("Epoch")
    ax[3].set_ylabel("$log_{10}(L_{recon})$")
    ax[3].legend(loc="upper right")

    ax[4].plot(loss_comps[:, 1], color='b', linewidth=lw, label='pred')
    ax[4].set_title("Prediction Loss")
    ax[4].grid()
    ax[4].set_xlabel("Epoch")
    ax[4].set_ylabel("$log_{10}(L_{pred})$")
    ax[4].legend(loc="upper right")

    ax[5].plot(loss_comps[:, 2], color='g', linewidth=lw, label='dmd')
    ax[5].set_title("DMD")
    ax[5].grid()
    ax[5].set_xlabel("Epoch")
    ax[5].set_ylabel("$log_{10}(L_{dmd})$")
    ax[5].legend(loc="upper right")

    fig.suptitle(
        "Epoch: {cur_epoch}/{max_epoch}, Learn Rate: {lr:.5f}, Val. Loss: {loss:.3f}".format(
            cur_epoch=epoch,
            max_epoch=hyp_params['max_epochs'],
            lr=hyp_params['lr'],
            loss=val_loss[-1]))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_3d_latent(y_pred, y_true, hyp_params, epoch, save_path, loss_comps, val_loss):
    enc_adv_dec = y_pred[2].numpy()
    evals = y_pred[5]

    font = {'family': 'DejaVu Sans', 'size': 24}
    matplotlib.rc('font', **font)

    skip = 8
    fig = plt.figure(figsize=(40, 20),facecolor='white')

    # Encoded-advanced-decoded time series
    ax = fig.add_subplot(2, 3, 1, projection='3d')
    for ii in np.arange(0, enc_adv_dec.shape[0], skip):
        ii = int(ii)
        x1 = enc_adv_dec[ii, :, 0]
        x2 = enc_adv_dec[ii, :, 1]
        x3 = enc_adv_dec[ii, :, 2]
        ax.plot3D(x1, x2, x3)
    ax.set_xlabel("$x_{1}$")
    ax.set_ylabel("$x_{2}$")
    ax.set_zlabel("$x_{3}$")
    ax.set_title("Encoded-Advanced-Decoded (x_adv))")

    #Eigenvalues
    ax = fig.add_subplot(2, 3, 2)
    t = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(t), np.sin(t), linewidth=1)
    ax.scatter(np.real(evals), np.imag(evals))
    ax.set_xlabel("Real$(\lambda)$")
    ax.set_ylabel("Imag$(\lambda)$")
    ax.set_title("Eigenvalues")

    # Loss components
    lw = 3
    loss_comps = np.asarray(loss_comps)
    ax = fig.add_subplot(2, 3, 3)
    ax.plot(val_loss, color='k', linewidth=lw, label='total')
    ax.set_title("Total Loss")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$log_{10}(L)$")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(2, 3, 4)
    ax.plot(loss_comps[:, 0], color='r', linewidth=lw, label='recon')
    ax.set_title("Recon Loss")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$log_{10}(L_{recon})$")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(2, 3, 5)
    ax.plot(loss_comps[:, 1], color='b', linewidth=lw, label='pred')
    ax.set_title("Prediction Loss")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$log_{10}(L_{pred})$")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(2, 3, 6)
    ax.plot(loss_comps[:, 2], color='g', linewidth=lw, label='dmd')
    ax.set_title("DMD")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$log_{10}(L_{dmd})$")
    ax.legend(loc="upper right")

    fig.suptitle(
        "Epoch: {cur_epoch}/{max_epoch}, Learn Rate: {lr:.5f}, Val. Loss: {loss:.3f}".format(
            cur_epoch=epoch,
            max_epoch=hyp_params['max_epochs'],
            lr=hyp_params['lr'],
            loss=val_loss[-1]))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
