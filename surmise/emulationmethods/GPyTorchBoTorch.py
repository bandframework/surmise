import numpy as np
import torch
import gpytorch
from gpytorch.constraints import GreaterThan
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim.fit import fit_gpytorch_torch


def fit(fitinfo, x, theta, f, ignore_nan=True, max_iter=500, **kwargs):
    # This module assumes x, theta, and f are in torch Tensor format
    if x is not None and isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if theta is not None and isinstance(theta, np.ndarray):
        theta = torch.from_numpy(theta)
    if f is not None and isinstance(f, np.ndarray):
        f = torch.from_numpy(f)

    if x is None:
        col_no = theta.shape[1]
        # Train GP on those realizations
        f = f.T

        if ignore_nan:
            fisnan = torch.isnan(f)
            f = f[~fisnan]
            xtheta = theta[~fisnan]

        model = SingleTaskGP(xtheta, f)

    else:
        # untensorized format
        row_flat = x.shape[0] * theta.shape[0]
        col_flat = x.shape[1] + theta.shape[1]

        f_flat = f.flatten().reshape(row_flat, 1)

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
        model = SingleTaskGP(xtheta, f_flat)
    model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))

    #
    # # use double precision
    # model.double()
    #
    # # turn on training mode
    # model.train()
    # likelihood.train()

    # Loss for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    # mll.train()
    # options = {'maxiter': 400, 'lr': 0.1}
    fit_gpytorch_model(mll)  #, options=options)


    # optimizer = FullBatchLBFGS(model.parameters(), lr=1, line_search=line_search)


    #
    # def closure():
    #     optimizer.zero_grad()
    #     output = model(xtheta)
    #     l = -mll(output, f_flat)
    #     return l
    #
    # loss = closure()
    # loss.backward()
    # old_loss = loss
    #
    # for i in range(max_iter):
    #     options = {'closure': closure, 'current_loss': loss, 'max_ls': max_ls}
    #     loss, lr, _, _, _, fail = optimizer.step(options)
    #
    #     loss.backward()
    #     grad = optimizer._gather_flat_grad()
    #
    #     grad_norm = torch.norm(grad)
    #     loss_dist = torch.abs(loss - old_loss)/torch.max(torch.tensor(1, dtype=torch.float), torch.abs(old_loss))
    #     print(
    #         'Iter %d/%d - Loss: %.3f - LR: %.3f - Log-Lengthscale: %.3f - Log_Noise: %.3f' % (
    #             i + 1, max_iter, loss.item(), lr,
    #             model.covar_module.base_kernel.lengthscale.item(),
    #             model.likelihood.noise.item()
    #         ))
    #     if grad_norm < tol or loss_dist < 1e-7:
    #         break
    #
    #     old_loss.copy_(loss)

    fitinfo['model'] = model
    fitinfo['likelihood'] = model.likelihood
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
