"""Shared test scenarios for surmise tests.

Importable at collection time so arrays can be used inside
@pytest.mark.parametrize.
"""
import numpy as np
import scipy.stats as sps
from .._RandomNumberGenerator import RandomNumberGenerator

RNG_SEED = 111848137687551512331846058163015350393
# local data generator, surmise RNG does not advance.
_datagen = np.random.default_rng(111848137687551512331846058163015350939)

DEFAULT_MH_SPECS = {
    "nSamples": 2000,
    "nBurnSamples": 1000,
    "theta0": None,
    "stepType": "Normal",
    "stepParam": None,
    "verbose": False
}

DEFAULT_PTLMC_SPECS = {
    'samplesPerChain': 100,
    'nSamples': 800,
    'theta0': np.array([[0, 9]]),
    'verbose': False,
    'nChains': 8,
    'maxTemperature': 30,
    'nTemperatures': 4
}

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
        _rng = RandomNumberGenerator().scipy_stats_RNG
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
theta_lin = np.vstack((sps.norm.rvs(0, 5, size=50, random_state=_datagen),
                       sps.gamma.rvs(2, 0, 10, size=50, random_state=_datagen))).T
f_lin = balldropmodel_linear(xv_lin, theta_lin)
y_lin = balldroptrue(xv_lin)
obsvar_lin = 4 * np.ones(x_lin.shape[0])

theta_new_lin = np.vstack((sps.norm.rvs(0, 5, size=10, random_state=_datagen),
                           sps.gamma.rvs(2, 0, 10, size=10, random_state=_datagen))).T
f_new_lin = balldropmodel_linear(xv_lin, theta_new_lin)

theta_test_lin = np.vstack((sps.norm.rvs(0, 5, size=50, random_state=_datagen),
                            sps.gamma.rvs(2, 0, 10, size=50, random_state=_datagen))).T


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
        _rng = RandomNumberGenerator().scipy_stats_RNG
        return np.vstack((sps.uniform.rvs(0, 1, size=n,
                                          random_state=_rng)))


x_td = np.array([[0.178, 0.356, 0.534, 0.712, 0.89, 1.068, 1.246,
                  1.424, 1.602, 1.78, 1.958, 2.67, 2.848, 3.026,
                  3.204, 3.382, 3.56, 3.738, 3.916, 4.094, 4.272]]).T
y_td = np.array([[0.27, 0.22, 0.27, 0.43, 0.41, 0.49, 0.46, 0.6,
                  0.65, 0.62, 0.7, 0.81, 0.69, 0.81, 0.89, 0.86,
                  0.89, 1.1, 1.05, 0.99, 1.05]]).T
obsvar_td = np.maximum(0.2 * y_td, 0.1)

n = 100
theta_ball = np.vstack((sps.uniform.rvs(0, 1, size=n, random_state=_datagen))).reshape(n, 1)
theta_range = np.array([1, 30])
x_range = np.array([min(x_td), max(x_td)])
x_std = (x_td - min(x_td)) / (max(x_td) - min(x_td))


# ----------------------------------------------------------------------
# Scenario C: borehole function
# ----------------------------------------------------------------------
def borehole_model(x, theta):
    """Given x and theta,
    return matrix of [row x] times [row theta] of values."""
    theta = tstd2theta(theta)
    x = xstd2x(x)
    p = x.shape[0]
    n = theta.shape[0]
    theta_stacked = np.repeat(theta, repeats=p, axis=0)
    x_stacked = np.tile(x.astype(float), (n, 1))
    f = borehole_vec(x_stacked, theta_stacked).reshape((n, p))
    return f.T


def borehole_true(x):
    """Given x, return matrix of [row x] times 1 of values."""
    # assume true theta is [0.5]^d
    theta0 = np.atleast_2d(np.array([0.5] * 4))
    f0 = borehole_model(x, theta0)
    return f0


def borehole_vec(x, theta):
    """Given x and theta, return vector of values."""
    (Hu, Ld_Kw, Treff, powparam) = np.split(theta, theta.shape[1], axis=1)
    (rw,  Hl) = np.split(x[:, :-1], 2, axis=1)
    numer = 2 * np.pi * (Hu - Hl)
    denom1 = 2 * Ld_Kw / rw ** 2
    denom2 = Treff
    f = ((numer / ((denom1 + denom2))) * np.exp(powparam * rw)).reshape(-1)
    return f


def tstd2theta(tstd):
    """Given standardized theta in [0, 1]^d, return non-standardized theta."""
    if tstd.ndim < 1.5:
        tstd = tstd[:, None].T
    (Treffs, Hus, LdKw, powparams) = np.split(tstd, tstd.shape[1], axis=1)

    Treff = (0.5-0.05) * Treffs + 0.05
    Hu = Hus * (1110 - 990) + 990
    Ld_Kw = LdKw * (1680 / 1500 - 1120 / 15000) + 1120 / 15000

    powparam = powparams * (0.5 - (- 0.5)) + (-0.5)

    theta = np.hstack((Hu, Ld_Kw, Treff, powparam))
    return theta


def xstd2x(xstd):
    """Given standardized x in [0, 1]^2 x {0, 1}, return non-standardized x."""
    if xstd.ndim < 1.5:
        xstd = xstd[:, None].T
    (rws, Hls, labels) = np.split(xstd, xstd.shape[1], axis=1)

    rw = rws * (np.log(0.5) - np.log(0.05)) + np.log(0.05)
    rw = np.exp(rw)
    Hl = Hls * (820 - 700) + 700

    x = np.hstack((rw, Hl, labels))
    return x


class thetaprior_bh:
    """ This defines the class instance of priors provided to the methods. """
    # def lpdf(theta):
    #     if theta.ndim > 1.5:
    #         return np.squeeze(np.sum(sps.norm.logpdf(theta, 1, 0.5), 1))
    #     else:
    #         return np.squeeze(np.sum(sps.norm.logpdf(theta, 1, 0.5)))

    def rnd(n):
        return np.vstack((sps.norm.rvs(1, 0.5, size=(n, 4),
                                       random_state=_datagen)))


x_bh = sps.uniform.rvs(0, 1, [50, 3], random_state=_datagen)
x_bh[:, 2] = x_bh[:, 2] > 0.5
yt_bh = np.squeeze(borehole_true(x_bh))
yvar_bh = (10 ** (-2)) * np.ones(yt_bh.shape)
thetatot_bh = (thetaprior_bh.rnd(15))
y_bh = yt_bh + sps.norm.rvs(0, np.sqrt(yvar_bh))
