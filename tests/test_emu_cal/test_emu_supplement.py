import numpy as np
import scipy.stats as sps
import pytest
from contextlib import contextmanager
from surmise.emulation import emulator
from surmise.calibration import calibrator
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

##############################################
#            Simple scenarios                #
##############################################
def balldropmodel_linear(x, theta):
    f = np.zeros((theta.shape[0], x.shape[0]))
    for k in range(0, theta.shape[0]):
        t = x[:, 0]
        h0 = x[:, 1] + theta[k, 0]
        vter = theta[k, 1]
        f[k, :] = h0 - vter * t
    return f.T


tvec = np.concatenate((np.arange(0.1, 4.3, 0.1), np.arange(0.1, 4.3, 0.1)))
h0vec = np.concatenate((25 * np.ones(42), 50 * np.ones(42)))
x = np.array([[0.1, 25.],
              [0.2, 25.],
              [0.3, 25.],
              [0.4, 25.],
              [0.5, 25.],
              [0.6, 25.],
              [0.7, 25.],
              [0.9, 25.],
              [1.1, 25.],
              [1.3, 25.],
              [2.0, 25.],
              [2.4, 25.],
              [0.1, 50.],
              [0.2, 50.],
              [0.3, 50.],
              [0.4, 50.],
              [0.5, 50.],
              [0.6, 50.],
              [0.7, 50.],
              [0.8, 50.],
              [0.9, 50.],
              [1.0, 50.],
              [1.2, 50.],
              [2.6, 50.],
              [2.9, 50.],
              [3.1, 50.],
              [3.3, 50.],
              [3.5, 50.],
              [3.7, 50.], ]).astype('object')
xv = x.astype('float')


class priorphys_lin:
    """ This defines the class instance of priors provided to the method. """
    def lpdf(theta):
        return (sps.norm.logpdf(theta[:, 0], 0, 5) +
                sps.gamma.logpdf(theta[:, 1], 2, 0, 10)).reshape((len(theta), 1))

    def rnd(n):
        return np.vstack((sps.norm.rvs(0, 5, size=n),
                          sps.gamma.rvs(2, 0, 10, size=n))).T


theta = priorphys_lin.rnd(50)
f = balldropmodel_linear(xv, theta)
f1 = f[0:15, :]
f2 = f[:, 0:25]
theta1 = theta[0:25, :]
x1 = x[0:15, :]
x1d = x[:, 0].reshape((x.shape[0],))
theta4d = np.hstack((theta1, theta1))
thetarnd = priorphys_lin.rnd(20)
thetacomb = np.vstack((theta1, thetarnd))

def balldroptrue(x):
    def logcosh(x):
        # preventing crashing
        s = np.sign(x) * x
        p = np.exp(-2 * s)
        return s + np.log1p(p) - np.log(2)
    t = x[:, 0]
    h0 = x[:, 1]
    vter = 20
    g = 9.81
    y = h0 - (vter ** 2) / g * logcosh(g * t / vter)
    return y


obsvar = 4*np.ones(x.shape[0])
y = balldroptrue(xv)

#######################################################
# Unit tests for supplement() method of emulator class #
#######################################################


@contextmanager
def does_not_raise():
    yield


# test to check supplement_x
@pytest.mark.parametrize(
      "input1,input2,input3,expectation",
      [
      (5, x, x1, pytest.raises(ValueError)),  # not supported
      (0.25, x, x1, pytest.raises(ValueError)),  # must be integer
      (5, None, x1, pytest.raises(ValueError)),
      ],
      )
def test_supplement_x(input1, input2, input3, expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGPwM')
    with expectation:
          assert emu.supplement(size=input1,
                              x=input2,
                              xchoices=input3) is not None


# test to check supplement_theta
@pytest.mark.parametrize(
    "input1,input2,input3,expectation",
    [
      # replication of emu.__theta
      (0, theta, theta1, pytest.raises(ValueError)),  # 'No supptheta exists.'
      (5, theta, theta1, pytest.raises(ValueError)),  # 'Complete replication of self.__theta'
      (5, None, theta1, pytest.raises(ValueError)),  # 'Provide either x or (theta or cal).'
      (5, theta, theta4d, pytest.raises(ValueError)),  # 'Dimension.'
      (5, theta, None, pytest.raises(ValueError)),  # 'Complete replication of self.__theta'
      (5, theta4d, None, pytest.raises(ValueError)),
      (5, thetarnd, None, does_not_raise()),
      (5, thetacomb, None, does_not_raise()),
      ],
    )
def test_supplement_theta(input1, input2, input3, expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGPwM')
    with expectation:
        assert emu.supplement(size=input1,
                              theta=input2,
                              thetachoices=input3) is not None


# test to check supplement_theta
@pytest.mark.parametrize(
    "input1,input2,expectation",
    [
      (x, theta, pytest.raises(ValueError)), #ValueError: You must either provide either x or (theta or cal).
      (None, None, pytest.raises(ValueError)), #ValueError: You must either provide either x or (theta or cal).
      ],
    )
def test_supplement_x_theta(input1, input2, expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGPwM')
    with expectation:
        assert emu.supplement(size=10, x=input1, theta=input2) is not None


# test to check supplement_cal
@pytest.mark.parametrize(
    "expectation",
    [
      (does_not_raise()),
      ],
    )
def test_supplement_cal(expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGPwM')
    args1 = {'theta0': np.array([[0, 9]]),
              'numsamp': 50,
              'stepType': 'normal',
              'stepParam': [0.1, 1]}
    cal = calibrator(emu=emu,
                      y=y,
                      x=x,
                      thetaprior=priorphys_lin,
                      method='directbayes',
                      yvar=obsvar,
                      args=args1)
    with expectation:
        assert emu.supplement(size=10, cal=cal) is not None

# test to check supplement_cal
@pytest.mark.parametrize(
    "expectation",
    [
      (does_not_raise()),
      ],
    )
def test_supplement_supp(expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGPwM')
    emu.supplement(size=5, theta=thetarnd)

    with expectation:
        assert emu.supplement(size=0) is not None

# test to check supplement_cal
@pytest.mark.parametrize(
    "expectation",
    [
     (pytest.raises(ValueError)),
     ],
    )
def test_supplement_method(expectation):
    emu = emulator(x=x, theta=theta, f=f, method='PCGP')
    with expectation:
        assert emu.supplement(size=5, theta=thetarnd) is not None

# test to check supplement_theta
#@pytest.mark.parametrize(
#    "input1,expectation",
#    [
#    (thetacomb, does_not_raise()), #ValueError: You must either provide either x or (theta or cal).
#    ],
#    )
#def test_supplement_match(input1, expectation):
#    emu = emulator(x=x, theta=theta, f=f, method='PCGPwM')
#    with expectation:
#        assert emu.supplement(size=15, theta=theta, thetachoices=input1) is not None
