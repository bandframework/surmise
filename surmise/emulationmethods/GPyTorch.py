import numpy as np
import torch
import gpytorch
from surmise.emulationsupport.LBFGS import FullBatchLBFGS
from gpytorch.constraints.constraints import GreaterThan


# Standard ExactGP model for exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit(fitinfo, x, theta, f, ignore_nan=True, max_iter=500, **kwargs):
    # This module assumes x, theta, and f are in torch Tensor format
    if x is not None and isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if theta is not None and isinstance(theta, np.ndarray):
        theta = torch.from_numpy(theta)
    if f is not None and isinstance(f, np.ndarray):
        f = torch.from_numpy(f)

    if x is None:
        # Train GP on those realizations
        f = f.T

        if ignore_nan:
            fisnan = torch.isnan(f)
            f = f[~fisnan]
            xtheta = theta[~fisnan]

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=GreaterThan(1e-5)
        )
        model = ExactGPModel(xtheta, f, likelihood)

    else:
        # untensorized format
        row_flat = x.shape[0] * theta.shape[0]
        col_flat = x.shape[1] + theta.shape[1]

        f_flat = f.flatten()

        xtheta = torch.from_numpy(
            np.array([(*x_item, *t_item)
                      for t_item in theta
                      for x_item in x]).reshape(row_flat, col_flat)
        )

        if ignore_nan:
            fisnan = torch.isnan(f_flat).squeeze()
            f_flat = f_flat[~fisnan]
            xtheta = xtheta[~fisnan]

        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(xtheta, f_flat, likelihood)

    # use double precision
    model.double()

    # turn on training mode
    model.train()
    likelihood.train()

    # Use L-BFGS optimizer
    tol = 1e-3
    line_search = 'Armijo'
    max_ls = 100
    # history_size = 10

    optimizer = FullBatchLBFGS(model.parameters(), lr=1, line_search=line_search)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1)

    # Loss for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def closure():
        optimizer.zero_grad()
        output = model(xtheta)
        l0 = -mll(output, f_flat)
        return l0

    loss = closure()
    loss.backward()
    old_loss = loss

    for i in range(max_iter):
        options = {'closure': closure, 'current_loss': loss, 'max_ls': max_ls}
        loss, lr, _, _, _, fail = optimizer.step(options)

        loss.backward()
        grad = optimizer._gather_flat_grad()

        grad_norm = torch.norm(grad)
        loss_dist = torch.abs(loss - old_loss)/torch.max(torch.tensor(1, dtype=torch.float), torch.abs(old_loss))
        print(
            'Iter %d/%d - Loss: %.3f - LR: %.3f - Log-Lengthscale: %.3f - Log_Noise: %.3f' % (
                i + 1, max_iter, loss.item(), lr,
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
        if grad_norm < tol or loss_dist < 1e-7:
            break

        old_loss.copy_(loss)

    fitinfo['model'] = model
    fitinfo['likelihood'] = likelihood
    return


def predict(predinfo, fitinfo, x, theta, **kwargs):
    if x is not None and isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if theta is not None and isinstance(theta, np.ndarray):
        theta = torch.from_numpy(theta)

    model = fitinfo['model']
    likelihood = fitinfo['likelihood']

    model.eval()
    likelihood.eval()

    if x is None:

        p = model(theta)
        _mean = p.mean.detach().numpy()
        _var = p.variance.detach().numpy()
    else:
        row_flat = x.shape[0] * theta.shape[0]
        col_flat = x.shape[1] + theta.shape[1]

        xtheta = torch.from_numpy(
            np.array([(*x_item, *t_item)
                      for t_item in theta
                      for x_item in x]).reshape(row_flat, col_flat)
        )

        p = model(xtheta)
        _mean = p.mean.detach().numpy().reshape(-1, x.shape[0]).T
        _var = p.variance.detach().numpy().reshape(-1, x.shape[0]).T

    predinfo['mean'] = _mean
    predinfo['var'] = _var

    return
