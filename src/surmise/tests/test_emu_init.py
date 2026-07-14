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
f0d = np.array(1)
theta0d = np.array(1)
x0d = np.array(1)


##############################################
# Unit tests to initialize an emulator class #
##############################################

# Followings are the tests to check the input configurations
# test to check none-type inputs
@pytest.mark.parametrize(
    "input1,input2,input3,expectation",
    [
     (x, theta, f, does_not_raise()),
     (x, None, f, pytest.raises(ValueError)),  # has not developed yet
     (None, theta, f, does_not_raise()),
     (x, theta, None, pytest.raises(ValueError)),
     (x, None, None, pytest.raises(ValueError)),
     (None, theta, None, pytest.raises(ValueError)),
     (None, None, f, pytest.raises(ValueError)),  # has not developed yet
     (None, None, None, pytest.raises(ValueError)),
     ],
    )
def test_none_input(input1, input2, input3, expectation):
    with expectation:
        assert emulator(x=input1,
                        theta=input2,
                        f=input3,
                        method='PCGP') is not None


# test to check the dimension of the inputs
@pytest.mark.parametrize(
    "input1,input2,input3,expectation",
    [
     (x, theta, f, does_not_raise()),
     (x, theta, f.T, pytest.raises(ValueError)),  # failure
     (x, None, f.T, pytest.raises(ValueError)),  # has not developed yet
     (x.T, theta, f, pytest.raises(ValueError)),
     (x.T, None, f, pytest.raises(ValueError)),
     (x, theta.T, f, pytest.raises(ValueError)),
     (x1, theta, f1, does_not_raise()),
     (x, theta, f1, pytest.raises(ValueError)),
     (x, theta, f2, pytest.raises(ValueError)),
     (x, theta1, f, pytest.raises(ValueError)),
     (None, theta1, f, pytest.raises(ValueError)),
     (None, theta, f.T, pytest.raises(ValueError)),
     (x1, theta, f, pytest.raises(ValueError)),
     ],
    )
def test_size_input(input1, input2, input3, expectation):
    with expectation:
        assert emulator(x=input1,
                        theta=input2,
                        f=input3,
                        method='PCGP') is not None


# test to check the dimension of the inputs
@pytest.mark.parametrize(
    "input1,input2,input3,expectation",
    [
     (x, theta, f0d, pytest.raises(ValueError)),
     (x0d, theta, f, pytest.raises(ValueError)),
     (x, theta0d, f, pytest.raises(ValueError)),
     ],
    )
def test_0d_input(input1, input2, input3, expectation):
    with expectation:
        assert emulator(x=input1,
                        theta=input2,
                        f=input3,
                        method='PCGP') is not None
# TO DO: Add tests for univariate data
# TO DO: Add tests for data including NAs and infs


# Following are the tests to check the emulator method configs
# test to check if an emulator module is imported
@pytest.mark.parametrize(
    "example_input,expectation",
    [
     ('PCGP', does_not_raise()),
     ('XXXX', pytest.raises(ValueError)),
     ],
    )
def test_method1(example_input, expectation):
    with expectation:
        assert emulator(x=x,
                        theta=theta,
                        f=f,
                        method=example_input) is not None


# test to check if 'thetareps' option is set correctly
@pytest.mark.parametrize(
    "input1,expectation",
    [
     (True, does_not_raise()),
     (False, does_not_raise()),
     (0, pytest.raises(ValueError)),
     (1, pytest.raises(ValueError)),
     (0.5, pytest.raises(ValueError)),
     ('XXXX', pytest.raises(ValueError)),
     ],
    )
def test_options1(input1, expectation):
    with expectation:
        assert emulator(x=x,
                        theta=theta,
                        f=f,
                        method='PCGP',
                        options={'thetareps': input1}) is not None


# test to check if 'xreps' option is set correctly
@pytest.mark.parametrize(
    "input1,expectation",
    [
     (True, does_not_raise()),
     (False, does_not_raise()),
     (0,  pytest.raises(ValueError)),
     (1,  pytest.raises(ValueError)),
     (0.5,  pytest.raises(ValueError)),
     ('XXXX', pytest.raises(ValueError)),
     ],
    )
def test_options2(input1, expectation):
    with expectation:
        assert emulator(x=x,
                        theta=theta,
                        f=f,
                        method='PCGP',
                        options={'xreps': input1}) is not None


# test to check if 'thetarmnan' option is set correctly
@pytest.mark.parametrize(
    "input1,expectation",
    [
     # (True, does_not_raise()),
     (False, does_not_raise()),
     # (0, does_not_raise()),
     (1, does_not_raise()),
     (0.5, does_not_raise()),
     (2, pytest.raises(ValueError)),
     # ('any', does_not_raise()),
     ('some', does_not_raise()),
     ('most', does_not_raise()),
     ('alot', does_not_raise()),
     ('all', does_not_raise()),
     ('never', does_not_raise()),
     ('XXXX', pytest.raises(ValueError)),
     ],
    )
def test_options3(input1, expectation):
    with expectation:
        assert emulator(x=x,
                        theta=theta,
                        f=f,
                        method='PCGP',
                        options={'thetarmnan': input1}) is not None


# test to check if 'xrmnan' option is set correctly
@pytest.mark.parametrize(
    "input1,expectation",
    [
     # (True, does_not_raise()),
     (False, does_not_raise()),
     # (0, does_not_raise()),
     (1, does_not_raise()),
     (0.5, does_not_raise()),
     (2, pytest.raises(ValueError)),
     # ('any', does_not_raise()),
     ('some', does_not_raise()),
     ('most', does_not_raise()),
     ('alot', does_not_raise()),
     ('all', does_not_raise()),
     ('never', does_not_raise()),
     ('XXXX', pytest.raises(ValueError)),
     ],
    )
def test_options4(input1, expectation):
    with expectation:
        assert emulator(x=x,
                        theta=theta,
                        f=f,
                        method='PCGP',
                        options={'xrmnan': input1}) is not None


# test to check if 'rmthetafirst' option is set correctly
@pytest.mark.parametrize(
    "input1,expectation",
    [
     (True, does_not_raise()),
     (False, does_not_raise()),
     (0, pytest.raises(ValueError)),
     (1, pytest.raises(ValueError)),
     (0.5, pytest.raises(ValueError)),
     ('XXXX', pytest.raises(ValueError)),
     ],
    )
def test_options5(input1, expectation):
    with expectation:
        assert emulator(x=x,
                        theta=theta,
                        f=f,
                        method='PCGP',
                        options={'rmthetafirst': input1}) is not None


# test to check if 'autofit' option is set correctly
@pytest.mark.parametrize(
    "input1,expectation",
    [
     (True, does_not_raise()),
     (False, does_not_raise()),
     (0, pytest.raises(ValueError)),
     (1, pytest.raises(ValueError)),
     (0.5, pytest.raises(ValueError)),
     ('XXXX', pytest.raises(ValueError)),
     ],
    )
def test_options6(input1, expectation):
    with expectation:
        assert emulator(x=x,
                        theta=theta,
                        f=f,
                        method='PCGP',
                        options={'autofit': input1}) is not None


# tests to check the emulator repr()
@pytest.mark.parametrize(
    "expectation",
    [
     (does_not_raise()),
     ],
    )
def test_repr(expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGP')
    with expectation:
        assert repr(emu) is not None


# tests to check the emulator call()
@pytest.mark.parametrize(
    "expectation",
    [
     (does_not_raise()),
     ],
    )
def test_call(expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGP')
    with expectation:
        assert emu(x=x, theta=theta) is not None
        emu.fit(args={})


# tests to check the emulator args
@pytest.mark.parametrize(
    "input1,expectation",
    [
     ({'epsilon': 1.5, 'hypregmean': -10, 'hypregLB': -20}, does_not_raise()),
     ],
    )
def test_args(input1, expectation):
    with expectation:
        assert emulator(x=x,
                        theta=theta,
                        f=f,
                        method='PCGP',
                        args=input1) is not None


@pytest.mark.parametrize(
    "expectation",
    [
     (does_not_raise()),
     ],
    )
def test_warning_filter(expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGP',
                   args={'warnings': True})
    with expectation:
        assert emu(x=x, theta=theta) is not None
