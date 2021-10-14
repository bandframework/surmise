import numpy as np
from missing_utils import MNAR_mask_quantiles, MNAR_mask_logistic

_dict = {
    'function': 'Piston',
    'xdim':     4,
    'thetadim': 3,
}


def query_func_meta():
    return _dict


def Piston_failmodel(x, theta, p):
    """Given x and theta, return matrix of [row x] times [row theta] of values."""
    f = Piston_model(x, theta)
    mask = MNAR_mask_logistic(f, p, p_params=0.1, exclude_inputs=True)
    f[mask] = np.nan

    return f


def Piston_failmodel_random(x, theta, p):
    f = Piston_model(x, theta)
    wheretoobig = np.where(np.random.choice([0, 1], f.shape, replace=True, p=[1-p, p]))
    f[wheretoobig[0], wheretoobig[1]] = np.nan
    return f


def Piston_model(x, theta):
    """Given x and theta, return matrix of [row x] times [row theta] of values."""

    theta = tstd2theta(theta)
    x = xstd2x(x)
    p = x.shape[0]
    n = theta.shape[0]

    theta_stacked = np.repeat(theta, repeats=p, axis=0)
    x_stacked = np.tile(x.astype(float), (n, 1))

    f = Piston_vec(x_stacked, theta_stacked).reshape((n, p))
    return f.T


def Piston_true(x):
    """Given x, return matrix of [row x] times 1 of values."""
    # assume true theta is [0.5]^d
    theta0 = np.atleast_2d(np.array([0.5] * _dict['thetadim']))
    f0 = Piston_model(x, theta0)

    return f0


def Piston_vec(x, theta):
    (M,S,V0,T0) = np.split(x, x.shape[1], axis=1)
    (k,P0,Ta) = np.split(theta, theta.shape[1], axis=1)

    Aterm1 = P0 * S
    Aterm2 = 19.62 * M
    Aterm3 = -k*V0 / S
    A = Aterm1 + Aterm2 + Aterm3

    Vfact1 = S / (2*k)
    Vfact2 = (A**2 + 4*k*(P0*V0/T0)*Ta) ** 0.5
    V = Vfact1 * (Vfact2 - A)

    fact1 = M
    fact2 = k + (S**2)*(P0*V0/T0)*(Ta/(V**2))

    C = (2 * np.pi * np.sqrt(fact1/fact2)).reshape(-1)
    return C


def xstd2x(xstd):
    if xstd.ndim < 1.5:
        xstd = xstd[:,None].T
    (Ms,Ss,V0s,T0s) = np.split(xstd, xstd.shape[1], axis=1)

    M = 30 + Ms * (60 - 30)
    S = 0.005 + Ss * (0.02 - 0.005)
    V0 = 0.002 + V0s * (0.01 - 0.002)
    T0 = 340 + T0s * (360 - 340)

    x = np.hstack((M, S, V0, T0))
    return x


def tstd2theta(tstd):
    """Given standardized theta in [0, 1]^d, return non-standardized theta."""
    if tstd.ndim < 1.5:
        tstd = tstd[:, None].T
    (ks,P0s,Tas) = np.split(tstd, tstd.shape[1], axis=1)

    k = 1000 + ks * (5000 - 1000)
    P0 = 90000 + P0s * (110000 - 90000)
    Ta = 290 + Tas * (296 - 290)

    theta = np.hstack((k, P0, Ta))
    return theta
