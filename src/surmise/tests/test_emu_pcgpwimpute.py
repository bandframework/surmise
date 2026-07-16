import numpy as np
import scipy.stats as sps
import pytest
from surmise.emulation import emulator

from .conftest import does_not_raise
from .shared_scenario import x_lin as x, theta_lin as theta, f_lin as f, _datagen

pytestmark = pytest.mark.usefixtures('seeded_rng')
##############################################
#            Simple scenarios                #
##############################################
f1 = f[0:15, :]
f2 = f[:, 0:25]
theta1 = theta[0:25, :]
x1 = x[0:15, :]
f0d = np.array(1)
theta0d = np.array(1)
x0d = np.array(1)
simsd = 1e-3 * np.ones_like(f)

# missing mask
maskU = sps.uniform.rvs(size=f.size, random_state=_datagen).reshape(*f.shape)
# tests missing data
f_miss = f.copy()
f_miss[maskU < 0.2] = np.nan


@pytest.mark.parametrize(
    "imputemethod, expectation",
    [
     ('BayesianRidge', does_not_raise()),
     ('KNN', does_not_raise()),
     ('RandomForest', does_not_raise()),
     ],
    )
def test_imputemethod(imputemethod, expectation):
    with expectation:
        assert emulator(x=x, theta=theta, f=f_miss,
                        method='PCGPwImpute',
                        args={'completionmethod': imputemethod}) is not None


@pytest.mark.parametrize(
    "input1, expectation",
    [
     (f, does_not_raise()),
     (f_miss, does_not_raise()),
     ],
    )
def test_fmissing(input1, expectation):
    with expectation:
        assert emulator(x=x, theta=theta, f=input1,
                        method='PCGPwImpute') is not None
