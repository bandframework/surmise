import numpy as np

_dict = {
    'function':     'Wingweight',
    'xdim':         6,
    'thetadim':     4,
    'xbounds':      [[150, 200],
                     [220, 300],
                     [0.08, 0.18],
                     [2.5, 6],
                     [1700, 2500],
                     [0.025, 0.08]],
    'thetabounds':  [[6, 10],
                     [-10, 10],
                     [16, 45],
                     [0.5, 1]]
}


def query_func_meta():
    return _dict


def Wingweight_failmodel(x, theta):
    """Given x and theta, return matrix of [row x] times [row theta] of values."""

    f = Wingweight_model(x, theta)
    wherextoobig = np.where(np.min(x, axis=1) < 0.05)
    wherethetatoobig = np.where(np.linalg.norm(theta, axis=1, ord=np.inf) > 0.8)
    faillist = np.array([(i, j) for i in wherextoobig[0] for j in wherethetatoobig[0]]).T
    f[faillist[0], faillist[1]] = np.nan

    return f


def Wingweight_failmodel_random(x, theta, p=0.2):
    """    """
    f = Wingweight_model(x, theta)
    wheretoobig = np.where(np.random.choice([0, 1], f.shape, replace=True, p=[1-p, p]))
    f[wheretoobig[0], wheretoobig[1]] = np.nan

    return f


def Wingweight_model(x, theta):
    """Given x and theta, return matrix of [row x] times [row theta] of values."""

    theta = tstd2theta(theta)
    x = xstd2x(x)
    p = x.shape[0]
    n = theta.shape[0]

    theta_stacked = np.repeat(theta, repeats=p, axis=0)
    x_stacked = np.tile(x.astype(float), (n, 1))

    f = Wingweight_vec(x_stacked, theta_stacked).reshape((n, p))
    return f.T


def Wingweight_true(x):
    """Given x, return matrix of [row x] times 1 of values."""
    # assume true theta is [0.5]^d
    theta0 = np.atleast_2d(np.array([0.5] * _dict['thetadim']))
    f0 = Wingweight_model(x, theta0)

    return f0


def Wingweight_vec(x, theta):
    (Sw, Wfw, tc, Nz, Wdg, Wp) = np.split(x, x.shape[1], axis=1)
    (A, LamCaps, q, lam) = np.split(theta, theta.shape[1], axis=1)

    LamCaps *= np.pi / 180.

    fact1 = 0.036 * Sw ** 0.758 * Wfw ** 0.0035
    fact2 = (A / (np.cos(LamCaps) ** 2)) ** 0.6
    fact3 = q ** 0.006 * lam ** 0.04
    fact4 = (100 * tc / np.cos(LamCaps)) ** (-0.3)
    fact5 = (Nz * Wdg) ** 0.49

    term1 = Sw * Wp

    W = (fact1 * fact2 * fact3 * fact4 * fact5 + term1).reshape(-1)
    return W


def xstd2x(xstd):
    if xstd.ndim < 1.5:
        xstd = xstd[:, None].T
    # (Sws, Wfws, tcs, Nzs, Wdgs, Wps) = np.split(xstd, xstd.shape[1], axis=1)

    bounds = np.array([[150, 200],
                       [220, 300],
                       [0.08, 0.18],
                       [2.5, 6],
                       [1700, 2500],
                       [0.025, 0.08]])

    x = bounds[:, 0] + xstd * (bounds[:, 1] - bounds[:, 0])
    return x


def tstd2theta(tstd):
    """Given standardized theta in [0, 1]^d, return non-standardized theta."""
    if tstd.ndim < 1.5:
        tstd = tstd[:, None].T
    # (A, LamCaps, q, lam) = np.split(tstd, tstd.shape[1], axis=1)

    bounds = np.array([[6, 10],
                       [-10, 10],
                       [16, 45],
                       [0.5, 1]])

    theta = bounds[:, 0] + tstd * (bounds[:, 1] - bounds[:, 0])
    return theta
