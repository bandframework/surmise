import numpy as np
import torch
import gpytorch
from gpytorch.constraints import GreaterThan
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP


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
    fit_gpytorch_model(mll)  # , options=options)

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
