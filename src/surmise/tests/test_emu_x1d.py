import pytest
from surmise.emulation import emulator

from .conftest import does_not_raise
from .shared_scenario import x_lin as x, theta_lin as theta, f_lin as f

pytestmark = pytest.mark.usefixtures('seeded_rng')
##############################################
#            Simple scenarios                #
##############################################


# test to check the predict with 1d-x
@pytest.mark.parametrize(
    "input1,input2,expectation",
    [
     (x, theta, does_not_raise()),
     (x.reshape(1, -1), theta, pytest.raises(ValueError)),
     ],
    )
def test_predict_multi(input1, input2, expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGP')
    with expectation:
        assert emu.predict(x=input1, theta=input2) is not None
