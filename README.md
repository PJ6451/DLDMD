# DLDMD

This is an extentsion of Deep Learning Dynamic Mode Decomposition, which utilizes the companion matrix formula in order to improve results with respect to chaotic systems.

# The Basics

For the uninitiated, Dynamic Mode Decomposition (DMD) utilizes a data-centric approach to calculate finite approximations of the _Koopman Operator_. At a high level, time-series data is often presented as a collection of things we can actually measure such as temperature, velocity, pressure, prices, etc. What Bernard Koopman proposed in 1931 is that the dynamics of whatever system we're interested in can be thought of in terms of a linear operator. Although this was specifically applied to Hamiltonian systems, _Koopmanism_ has spread to many subjects, and is an up-and-coming algorithm for time-series analysis.

Companion Matrix DMD (CDMD) works by finding a finite-dimensional matrix approximation of this operator utilizing the Companion Matrix.  Let $X_m = (f_1, f_2, \dots, f_m)$ be defined as our data matrix, where the columns are the values of our observables for some time-step $dt$. The rows here represent trajectories for initial conditions, and often represent as many trajectories as we can generate (without creating linear dependence due to the power iteration of the subspace) to over-determine the system. The data can be from data generation software, data sampled from experiments, or even as data generated from governing equations using a fourth order Runge-Kuta method.

We assume that the columns of $X_m$ span a Krylov subspace 
$$K_m(K_a,f_1) = span(f_1, f_2, \dots, f_m)$$
such that there exists a matrix $K_a$ such that
$$f_2 = K_a f_1, \quad f_3 = K_a^2 f_1, \quad ... \quad f_m = K_a^{m-1} f_1.$$ 
In this sense, $K_a$ serves as the finite-dimensional approximation of the infinite-dimensional Koopman operator. We also assume that $X_m$ is of full rank, which is to say that the initial conditions are all unique. We can therefore say that the operation of the approximation $K_a$ on $X_m$ is defined as 
$$K_a X_m = X_m C_m + E_{m+1}$$
where $C_m$ is the Frobenius companion matrix given by
```math
C_m = \begin{pmatrix} 0 & 0 & \cdots & 0 & c_1 \\ 1 & 0 & \cdots & 0 & c_2 \\ 0 & 1 & \cdots & 0 & c_3 \\ \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & \cdots & 1 & c_m \\ \end{pmatrix}
```
and 
$$E_{m+1} = r_{m+1}e_m^T$$
where $e_m = (0, 0, ..., 1)$ comes from the canonical basis of $\mathbb{R}^n$. The residual $r_{m+1}$ measures the error of this approximation, and is given by
$$r_{m+1} = X_{m} \mathbf{c} - f_{m+1}$$
where $\mathbf{c}$ is the final column of $C_m$. The error is then minimized through calculating The constants $c_j$ via the least-squares problem 
$$||X_{m} \mathbf{c} - f_{m+1}||_2.$$
This effectively reformulates the final snapshot as a linear combination of the previous steps, making the algorithm auto-regressive and more akin to ARMA or ARIMA methods.

The spectral decomposition of $C_m$ is then given by
$$C_m = \mathbb{V}_m \Delta_m \mathbb{V}_m^{-1}$$
where the eigenvalues are 
```math
\Delta_m = 
	\begin{pmatrix} 
        \lambda_1 &  & \\
        & \ddots & \\
         &  & \lambda_m\\
      \end{pmatrix},
```
the left eigenvectors are given row-wise by the Vandermonde matrix
```math
\mathbb{V}_m^{-1} = 
    \begin{bmatrix} 
        1 & \lambda_1 & \cdots & \lambda_1^{m-1} \\
        1 & \lambda_2 & \cdots & \lambda_2^{m-1} \\
        1 & \lambda_3 & \cdots & \lambda_3^{m-1} \\
        \vdots & \vdots & \cdots & \vdots \\
        1 & \lambda_m & \cdots & \lambda_m^{m-1} \\
    \end{bmatrix}
```
and the right eigenvectors are given column-wise by $\mathbb{V}_m$. From here we can define the modes as $K_M = X_m \mathbb{V}_m$, and define scaling amplitudes for reconstruction as
```math
A = K_M^{\dagger} f_1.
```
where $K_M^{\dagger}$ is the Moore-Penrose pseudo-inverse of $K_M$. This gives us the reconstruction formula
$$f_i = \sum_{k = 0}^{m} w_k a_k \lambda_{k}^{i-1}.$$
where $\lambda_k \in \Delta, w_k \in K_M, a_k \in A$. For a more rigorous treatment, please refer to [this paper.](https://arxiv.org/abs/2009.05883.)

# Extended and Deep Learning DMD
CDMD on its own does not handle certain time-series problems very well, so further methods have been developed. Extended DMD basically first encodes the data through a series of non-linear dictionary functions, and DLDMD takes this a step further by letting an auto-encoder find an optimal dictionary through unsupervised learning ([see here](https://arxiv.org/abs/2108.04433)). The main thing to note here is that both of these methods utilize a one-step method to approximating the Koopman operator, the other option being to utilize Krylov subspaces and the [Frobenious Companion Matrix](https://arxiv.org/abs/1808.09557). The second method would be more optimal from a mathematics standpoint, due to its similarity with ARMA/ARIMA based methods; however, it hasn't been used due to eigenvalues approximations being too high.

While DLDMD has been shown to work for certain systems, the one step approach was not very good at handling chaotic systems like Lorenz 63. The approach of this project was to see if utilizing CDMD would help with these systems. To that end the following algorithm was built:

![Image](https://github.com/PJ6451/DLDMD/blob/main/algorithm.png)

The variables given are defined as the maximum number of epochs $E_{max}$, as well as the number of reconstruction steps $N_R$. To test the forecasting ability, we typically let $N_R < m$, where $m$ is the total nubmer of time-steps in our series. We begin with encoding and decoding networks, in affect creating an auto-encoder such that
```math
\mathcal{E}: \mathbb{R}^{N_s} \to \mathbb{R}^{N_o}, \quad \mathcal{E}(x) = \Tilde{x}
```
and
```math
\mathcal{D}: \mathbb{R}^{N_o} \to \mathbb{R}^{N_s}, \quad \mathcal{D}(\Tilde{x}) = x.
```
The neural network finds an optimal encoding space based on the loss function $\mathcal{L}$, given by
```math
\mathcal{L} = \alpha_1 \mathcal{L}_{recon} + \alpha_2 \mathcal{L}_{pred} + \alpha_3 \mathcal{L}_{dmd} + \alpha_4 ||\mathbf{W}_g||_2^2
```
where $\{ \alpha_i \}_{i=1}^4 > 0$ and
```math
\mathcal{L}_{recon} = ||X - \mathcal{D}(\mathcal{E}(X))||_{MSE},
```
```math
\mathcal{L}_{pred} = ||X - \Tilde{X}||_{MSE},
```
```math
\mathcal{L}_{dmd} = ||Y - \Tilde{Y}||_{MSE}
```
(the final term is to regularize the weights of the nodes, to ensure they don't grow too big). In the latent (encoded) space, we execute the CDMD algorithm, the results then decoded and tested with the raw data. This was tested on a Harmonic Oscillator, the Lorenz 63 and 96 equations, and for the Rossler system, with various predictions steps.


# Results
The results can be found in the examples folders, but to summarize: DLDMD was now able to handle chaotic systems and forecast for multiple time-steps. As an example, refer to the image below, showing the results of DLDMD utilized on the Lorenz 96 system. For this experiment, $t_f = 30, dt = 0.05$ and the system is set to forecast 100 time-steps. Reconstruction is from $0 \to 25$, prediction is from $25 \to 30$. From left to right, the top row shows the validation data, the reconstructed and forecast data utilizing CDMD, and the encoded-decoded raw data, The middle row shows the encoded-advanced data, the eigenvalues of the Companion matrix, and a semi-log plot of total loss calculated per epoch. The bottom row shows the semi-log plots for the reconstruction loss, the prediction loss, and DMD loss.

![Image](https://github.com/PJ6451/DLDMD/blob/main/examples/lorenz96/lorenz96.png)

Overall, we see that for quite a few time-steps we're able to get pretty decent results. We don't expect to reasonably predict quite this many steps based on the leading Lyapunov exponent, but overall we see some that the autoencoder was able to a decent transformation to improve the base CDMD method.
