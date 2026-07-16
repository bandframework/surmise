import numpy as np
import pytest
from surmise.emulation import emulator

from .conftest import does_not_raise
from .shared_scenario import x_lin as x, theta_lin as theta, f_lin as f

pytestmark = pytest.mark.usefixtures('seeded_rng')


##############################################
#            Simple scenarios                #
##############################################
f1 = f[0:15, :]
f2 = f[:, 0:25]
theta1 = theta[0:25, :]
x1 = x[0:15, :]


@pytest.mark.parametrize(
    "input,expectation",
    [
     ('PCGP', does_not_raise()),
     ('PCGPwM', does_not_raise()),
     ('indGP', does_not_raise()),
     ('PCGPwImpute', does_not_raise()),
     ('PCSK', does_not_raise()),
     ('XXXX', pytest.raises(ValueError)),
     ],
    )
def test_repr(input, expectation):
    with expectation:
        if input != 'PCSK':
            assert emulator(x=x,
                            theta=theta,
                            f=f,
                            method=input) is not None
        else:
            simsd = 1e-3 * np.ones_like(f)
            assert emulator(x=x,
                            theta=theta,
                            f=f,
                            method=input,
                            args={'simsd': simsd}) is not None
