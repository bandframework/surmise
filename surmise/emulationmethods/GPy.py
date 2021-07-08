import numpy as np
import GPy


def fit(fitinfo, x, theta, f, ignore_nan=True, **kwargs):

    if x is None:
        col_no = theta.shape[1]
        # Train GP on those realizations
        f = f.T

        if ignore_nan:
            fisnan = np.isnan(f).squeeze()
            f = f[~fisnan]
            theta = theta[~fisnan]

        kernel = GPy.kern.RBF(input_dim=col_no, variance=1., lengthscale=1.)  # generalize for user input
        white_kern = GPy.kern.White(1, variance=0.1)  # generalize for user input

        kernel = (kernel + white_kern)

        emulator = GPy.models.GPRegression(theta, f, kernel)
        emulator.optimize()

    else:
        # untensorized format
        row_flat = x.shape[0]*theta.shape[0]
        col_flat = x.shape[1] + theta.shape[1]

        f_flat = f.flatten('F').reshape(row_flat, 1)

        xtheta = np.array([(*x_item, *t_item)
                           for t_item in theta
                           for x_item in x]).reshape(row_flat, col_flat)
        if ignore_nan:
            fisnan = np.isnan(f_flat).squeeze()
            f_flat = f_flat[~fisnan]
            xtheta = xtheta[~fisnan]

        # Train GP on those realizations
        kernel = GPy.kern.RBF(input_dim=col_flat, variance=1, lengthscale=1)  # generalize for user input
        emulator = GPy.models.GPRegression(xtheta, f_flat, kernel)
        emulator.optimize()

        # raise some warning or error in this case
        if np.isnan(emulator.objective_function()):
            print('Warning: GPy emulator did not converge.')

    fitinfo['emu'] = emulator
    return


def predict(predinfo, fitinfo, x, theta, **kwargs):

    emulator = fitinfo['emu']

    if x is None:
        p = emulator.predict(theta)
        _mean = p[0].T
        _var = (p[1]**2).T
    else:
        row_flat = x.shape[0]*theta.shape[0]
        col_flat = x.shape[1] + theta.shape[1]

        xtheta = np.array([(*x_item, *t_item)
                           for t_item in theta
                           for x_item in x]).reshape(row_flat, col_flat)
        p = emulator.predict(xtheta)

        _mean = p[0].reshape(-1, x.shape[0]).T
        _var = (p[1]**2).reshape(-1, x.shape[0]).T

    predinfo['mean'] = _mean
    predinfo['var'] = _var

    return
