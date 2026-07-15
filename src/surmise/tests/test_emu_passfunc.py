import pytest
from surmise.emulation import emulator

from .conftest import does_not_raise
from .shared_scenario import borehole_model, x_bh as x, thetatot_bh as thetatot

pytestmark = pytest.mark.usefixtures('seeded_rng')


# test to check the emulator with a passed function
@pytest.mark.parametrize(
    "expectation",
    [
     (does_not_raise()),
     ],
    )
def test_passfunction(expectation):
    with expectation:
        assert emulator(passthroughfunc=borehole_model,
                        method='PCGP') is not None


# test to check the emulator predict with a passed function
@pytest.mark.parametrize(
    "x0, theta0, expectation",
    [
        (x, thetatot, does_not_raise()),
        (x, None, pytest.raises(ValueError)),
        (None, None, pytest.raises(ValueError))
     ],
    )
def test_passfunction_predict(x0, theta0,
                              expectation):
    with expectation:
        emu = emulator(passthroughfunc=borehole_model,
                       method='PCGP')
        assert emu.predict(x=x0, theta=theta0) is not None
