"""Shared test scenarios for surmise tests.

Importable at collection time so arrays can be used inside
@pytest.mark.parametrize.
"""
import numpy as np
import scipy.stats as sps

_rng = np.random.default_rng(111848137687551512331846058163015350939)


# ----------------------------------------------------------------------
# Scenario A: linear ball-drop (used by 14 emu/cal test files)
# ----------------------------------------------------------------------
def balldropmodel_linear(x, theta):
    f = np.zeros((theta.shape[0], x.shape[0]))
    for k in range(0, theta.shape[0]):
        t = x[:, 0]
        h0 = x[:, 1] + theta[k, 0]
        vter = theta[k, 1]
        f[k, :] = h0 - vter * t
    return f.T


def balldroptrue(x):
    def logcosh(x):
        s = np.sign(x) * x
        p = np.exp(-2 * s)
        return s + np.log1p(p) - np.log(2)

    t = x[:, 0]
    h0 = x[:, 1]
    vter = 20
    g = 9.81
    return h0 - (vter ** 2) / g * logcosh(g * t / vter)


class priorphys_lin:
    """Prior class for the linear ball-drop scenario."""

    def lpdf(theta):
        return (sps.norm.logpdf(theta[:, 0], 0, 5) +
                sps.gamma.logpdf(theta[:, 1], 2, 0, 10)
                ).reshape((len(theta), 1))

    def rnd(n):
        return np.vstack((
            sps.norm.rvs(0, 5, size=n, random_state=_rng),
            sps.gamma.rvs(2, 0, 10, size=n, random_state=_rng))).T


x_lin = np.array(
    [[0.1, 25.], [0.2, 25.], [0.3, 25.], [0.4, 25.], [0.5, 25.],
     [0.6, 25.], [0.7, 25.], [0.9, 25.], [1.1, 25.], [1.3, 25.],
     [2.0, 25.], [2.4, 25.],
     [0.1, 50.], [0.2, 50.], [0.3, 50.], [0.4, 50.], [0.5, 50.],
     [0.6, 50.], [0.7, 50.], [0.8, 50.], [0.9, 50.], [1.0, 50.],
     [1.2, 50.], [2.6, 50.], [2.9, 50.], [3.1, 50.], [3.3, 50.],
     [3.5, 50.], [3.7, 50.]]).astype('object')
xv_lin = x_lin.astype('float')

theta_lin = priorphys_lin.rnd(50)
f_lin = balldropmodel_linear(xv_lin, theta_lin)
y_lin = balldroptrue(xv_lin)
obsvar_lin = 4 * np.ones(x_lin.shape[0])


# ----------------------------------------------------------------------
# Scenario B: timedrop (test_cal_directbayes, test_cal_saveload,
# test_new_cal)
# ----------------------------------------------------------------------
def timedrop(x, theta, hr, gr):
    """Computer implementation of the mathematical model."""
    min_g = min(gr)
    range_g = max(gr) - min(gr)
    min_h = min(hr)
    range_h = max(hr) - min_h
    f = np.zeros((theta.shape[0], x.shape[0]))
    for k in range(0, theta.shape[0]):
        g = range_g * theta[k] + min_g
        h = range_h * x + min_h
        f[k, :] = np.sqrt(2 * h / g).reshape(x.shape[0])
    return f.T


class prior_balldrop:
    """Prior class for the timedrop scenario."""

    def lpdf(theta):
        return sps.uniform.logpdf(theta[:, 0], 0, 1
                                  ).reshape((len(theta), 1))

    def rnd(n):
        return np.vstack((sps.uniform.rvs(0, 1, size=n,
                                          random_state=_rng)))


x_td = np.array([[0.178, 0.356, 0.534, 0.712, 0.89, 1.068, 1.246,
                  1.424, 1.602, 1.78, 1.958, 2.67, 2.848, 3.026,
                  3.204, 3.382, 3.56, 3.738, 3.916, 4.094, 4.272]]).T
y_td = np.array([[0.27, 0.22, 0.27, 0.43, 0.41, 0.49, 0.46, 0.6,
                  0.65, 0.62, 0.7, 0.81, 0.69, 0.81, 0.89, 0.86,
                  0.89, 1.1, 1.05, 0.99, 1.05]]).T
obsvar_td = np.maximum(0.2 * y_td, 0.1)

theta_range = np.array([1, 30])
x_range = np.array([min(x_td), max(x_td)])
x_std = (x_td - min(x_td)) / (max(x_td) - min(x_td))


def set_RNG_in_tests(rng):
    import surmise
    surmise.set_RNG(rng)
