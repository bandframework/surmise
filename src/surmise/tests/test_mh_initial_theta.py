import numpy as np
import pytest

from surmise.utilitiesmethods.metropolis_hastings import sampler
SPECS = {"nSamples": 50, "nBurnSamples": 10, "theta0": None,
         "stepType": "normal", "stepParam": np.array([0.1]),
         "verbose": False}


def logpost_unit_interval(theta, return_grad=False):
    """Uniform(0, 1) log density, one row per theta."""
    inside = np.all((theta > 0) & (theta < 1), axis=1)
    return np.where(inside, 0.0, -np.inf).reshape(-1, 1)


def test_first_finite_candidate_is_used():
    rng = np.random.default_rng(0)
    n_bad = 37
    calls = []

    def draw_func(n):
        calls.append(n)
        return np.vstack((np.full((n_bad, 1), -1.0),
                          np.linspace(0.2, 0.8, n - n_bad)[:, None]))

    out = sampler(logpost_unit_interval, draw_func, rng, dict(SPECS))
    assert out["theta"].shape == (SPECS["nSamples"], 1)
    assert np.all(np.isfinite(out["theta"]))
    assert np.all((out["theta"] > 0) & (out["theta"] < 1))


def test_all_candidates_zero_density_raises():
    rng = np.random.default_rng(0)

    def draw_func(n):
        return np.full((n, 1), -1.0)

    with pytest.raises(RuntimeError, match=r"All \d+ initial theta"):
        sampler(logpost_unit_interval, draw_func, rng, dict(SPECS))


def test_nan_candidates_are_skipped():
    rng = np.random.default_rng(0)

    def logpost(theta, return_grad=False):
        lp = logpost_unit_interval(theta)
        lp[theta[:, 0] < 0.5] = np.nan
        return lp

    def draw_func(n):
        return np.linspace(0.01, 0.99, n)[:, None]

    out = sampler(logpost, draw_func, rng, dict(SPECS, stepParam=np.array([1e-5])))
    assert out["theta"][0, 0] >= 0.5


@pytest.mark.parametrize("theta0", [np.array([[-1.0]]), np.array([[2.0]])])
def test_given_theta0_zero_density_raises(theta0):
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError, match="Initial theta returns"):
        sampler(logpost_unit_interval, lambda n: np.full((n, 1), 0.5), rng,
                dict(SPECS, theta0=theta0))
