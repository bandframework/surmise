"""OTL circuit function. (xdim = 4, thetadim=2)."""
import numpy as np
from missing_utils import MNAR_mask_quantiles, MNAR_mask_logistic

_dict = {
    'function': 'OTLcircuit',
    'xdim':     4,
    'thetadim': 2
}


def query_func_meta():
    return _dict


def OTLcircuit_failmodel(x, theta, p):
    """Given x and theta, return matrix of [row x] times [row theta] of values."""
    f = OTLcircuit_model(x, theta)
    mask = MNAR_mask_logistic(f, p, p_params=0.1, exclude_inputs=True)
    f[mask] = np.nan
    return f


def OTLcircuit_failmodel_random(x, theta, p):
    f = OTLcircuit_model(x, theta)
    wheretoobig = np.where(np.random.choice([0, 1], f.shape, replace=True, p=[1-p, p]))
    f[wheretoobig[0], wheretoobig[1]] = np.nan
    return f


def OTLcircuit_model(x, theta):
    """Given x and theta, return matrix of [row x] times [row theta] of values."""

    theta = tstd2theta(theta)
    x = xstd2x(x)
    p = x.shape[0]
    n = theta.shape[0]

    theta_stacked = np.repeat(theta, repeats=p, axis=0)
    x_stacked = np.tile(x.astype(float), (n, 1))

    f = OTLcircuit_vec(x_stacked, theta_stacked).reshape((n, p))
    return f.T


def OTLcircuit_true(x):
    """Given x, return matrix of [row x] times 1 of values."""
    # assume true theta is [0.5]^d
    theta0 = np.atleast_2d(np.array([0.5] * 2))
    f0 = OTLcircuit_model(x, theta0)

    return f0


def OTLcircuit_vec(x, theta):
    """
    The OTL Circuit function models an output transformerless push-pull circuit. The response Vm is midpoint voltage.
    OTL_circuit(x) evaluates the function for n-by-6 input matrix x, and
    returns its function value.

    Args:
        x : matrix of (n, 6), where n is the number of inputs.
            Valid input x is generated from gen_OTLcircuit_input().

    Returns:
        Vm : vector of (n, ). OTL circuit function evaluated at input x.

    """

    (Rb1,Rb2,Rc1,Rc2) = np.split(x, x.shape[1], axis=1)
    (Rf,beta) = np.split(theta, theta.shape[1], axis=1)

    const1 = 0.74

    Vb1 = 12*Rb2 / (Rb1+Rb2)
    term1a = (Vb1+const1) * beta * (Rc2+9)
    term1b = beta*(Rc2+9) + Rf
    term1 = term1a / term1b

    term2a = 11.35 * Rf
    term2b = beta*(Rc2+9) + Rf
    term2 = term2a / term2b

    term3a = const1 * Rf * beta * (Rc2+9)
    term3b = (beta*(Rc2+9)+Rf) * Rc1
    term3 = term3a / term3b

    Vm = (term1 + term2 + term3).reshape(-1)

    return Vm


def tstd2theta(tstd):
    """Given standardized theta in [0, 1]^d, return non-standardized theta."""
    if tstd.ndim < 1.5:
        tstd = tstd[:, None].T
    (Rfs, betas) = np.split(tstd, tstd.shape[1], axis=1)

    Rf = 0.5 + Rfs * (3 - 0.5)
    beta = 50 + betas * (300 - 50)

    theta = np.hstack((Rf, beta))
    return theta


def xstd2x(xstd):
    if xstd.ndim < 1.5:
        xstd = xstd[:,None].T
    (Rb1s, Rb2s, Rc1s, Rc2s) = np.split(xstd, xstd.shape[1], axis=1)

    Rb1 = 50 + Rb1s * (150 - 50)
    Rb2 = 25 + Rb2s * (75 - 25)
    Rc1 = 1.2 + Rc1s * (2.5 - 1.2)
    Rc2 = 0.25 + Rc2s * (1.2 - 0.25)

    x = np.hstack((Rb1, Rb2, Rc1, Rc2))
    return x
