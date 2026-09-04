import numpy as np
import scipy.stats as sps
import pytest
from surmise.calibration import calibrator

from .conftest import does_not_raise
from .shared_scenario import x_lin as x, theta_lin, f_lin, y_lin as y, \
                             obsvar_lin as obsvar, priorphys_lin, RNG_SEED

pytestmark = pytest.mark.usefixtures('seeded_rng')

_rng = np.random.default_rng(seed=RNG_SEED)

# TODO: LMC will require 'expertMode' in lmc_option to run
SAMPLERS_IN_TEST = ['metropolis_hastings', 'PTLMC']  # , 'LMC']
METHODS_IN_TEST = ['directbayes', 'directbayeswoodbury']

##############################################
#            Simple scenarios                #
##############################################

# Additional examples
y1 = y[0:3]

# setting obsvar
obsvar1 = obsvar[0:10]
obsvar2 = -obsvar
obsvar3 = 10 ** (10) * obsvar

# 2-d x (30 x 2), 2-d theta (50 x 2), f1 (15 x 50)
f1 = f_lin[0:15, :]
# 2-d x (30 x 2), 2-d theta (50 x 2), f2 (30 x 25)
f2 = f_lin[:, 0:25]
# 2-d x (30 x 2), 2-d theta1 (25 x 2), f (30 x 50)
theta1 = theta_lin[0:25, :]
# 2-d x1 (15 x 2), 2-d theta (50 x 2), f (30 x 50)
x1 = x[0:15, :]

f0d = np.array(1)
theta0d = np.array(1)
x0d = np.array(1)


# ### #### #### different prior examples #### #### ### #
class prior_example1:
    def lpdf(theta):
        return (sps.norm.logpdf(theta[:, 0], 0, 5),
                sps.gamma.logpdf(theta[:, 1], 2, 0, 10)).reshape((len(theta), 1))

    def rnd(n):
        return np.vstack((sps.norm.rvs(0, 5, size=n, random_state=_rng),
                          sps.gamma.rvs(2, 0, 10, size=n, random_state=_rng))).T


class prior_rnd1:
    def lpdf(theta):
        return np.array([1, 2, 3])

    def rnd(n):
        return np.array([1, 2, 3])


class prior_rnd2:
    def nothing(self):
        return None


class prior_lpdf1:
    def lpdf(theta):
        return np.array([1, 2, 3])

    def rnd(n):
        return np.vstack((sps.norm.rvs(0, 5, size=n, random_state=_rng),
                          sps.gamma.rvs(2, 0, 10, size=n, random_state=_rng))).T


class prior_lpdf2:
    def rnd(n):
        return np.vstack((sps.norm.rvs(0, 5, size=n, random_state=_rng),
                          sps.gamma.rvs(2, 0, 10, size=n, random_state=_rng))).T


# Some additional args
mh_args1 = {'theta0': np.array([[0, 9]]),
            'numsamp': 50,
            'stepType': 'normal',
            'stepParam': [0.1, 1]}
mh_args2 = {'theta0': np.array([[0, 9]]),
            'numsamp': 50,
            'stepType': 'uniform',
            'stepParam': [0.1, 1]}
mh_args3 = {'theta0': np.array([[0, 9]]),
            'stepParam': [0.1, 1]}
mh_args4 = {'stepParam': [0.1, 1]}
mh_args5 = {'theta0': np.array([[0, 9]])}

#
ptlmc_args1 = {'theta0': np.array([[0, 9]]),
               'numsamp': 50,
               'numtemps': 8,
               'sampperchain': 25}
ptlmc_args2 = {'theta0': np.array([[0, 9]]),
               'numsamp': 50,
               'numchain': 8,
               'maxtemp': 30}
ptlmc_args3 = {'theta0': np.array([[0, 9]])}

#
lmc_args1 = {'theta0': np.array([[0, 9]]),
             'numsamp': 50,
             'expertMode': True}

args_dict = {'metropolis_hastings': [mh_args1, mh_args2, mh_args3, mh_args4, mh_args5],
             'PTLMC': [ptlmc_args1, ptlmc_args2, ptlmc_args3],
             'LMC': [lmc_args1]}

##############################################
# Unit tests to initialize an emulator class #
##############################################

# Generate all test pairs accordingly upfront
SAMPLER_ARGS_PAIRS = [
    pytest.param(sampler, args, method, id=f"{sampler}-args{i}-method{k}")
    for sampler in SAMPLERS_IN_TEST
    for i, args in enumerate(args_dict[sampler])
    for k, method in enumerate(METHODS_IN_TEST)
]


@pytest.mark.parametrize("sampler,args,method", SAMPLER_ARGS_PAIRS)
def test_cal_MLcal_wo_grad(sampler, args, method, emu_lin_pcgp):
    args_tmp = args.copy()
    args_tmp['sampler'] = sampler
    with does_not_raise():
        assert calibrator(emu=emu_lin_pcgp,
                          y=y,
                          x=x,
                          thetaprior=priorphys_lin,
                          method=method,
                          yvar=obsvar,
                          args=args_tmp) is not None


@pytest.mark.parametrize("sampler,args,method", SAMPLER_ARGS_PAIRS)
def test_cal_MLcal_w_grad(sampler, args, method, emu_lin_pcgpwm_wgrad):
    args_tmp = args.copy()
    args_tmp['sampler'] = sampler
    with does_not_raise():
        assert calibrator(emu=emu_lin_pcgpwm_wgrad,
                          y=y,
                          x=x,
                          thetaprior=priorphys_lin,
                          method=method,
                          yvar=obsvar,
                          args=args_tmp) is not None


@pytest.mark.parametrize('sampler', SAMPLERS_IN_TEST)
class TestSampler:

    @pytest.mark.parametrize(
        "input2,input3,input4,input5,expectation",
        [
            (y, x1, priorphys_lin, obsvar, pytest.raises(ValueError)),
            (y, x, priorphys_lin, obsvar1, pytest.raises(ValueError)),
            (y, x, priorphys_lin, obsvar2, pytest.raises(ValueError)),
            (y, x, priorphys_lin, obsvar3, pytest.raises(ValueError)),
            (y, x, prior_rnd1, obsvar, pytest.raises(ValueError)),
            (y, x, prior_rnd2, obsvar, pytest.raises(AttributeError)),
            (y, x, prior_lpdf1, obsvar, pytest.raises(ValueError)),
            (y, x, prior_lpdf2, obsvar, pytest.raises(ValueError)),
            (y, x, prior_example1, obsvar, pytest.raises(ValueError)),
            (y1, x, priorphys_lin, obsvar, pytest.raises(ValueError)),
        ],
    )
    def test_cal_emu_fails(self, sampler, emu_lin_pcgp, input2, input3, input4, input5, expectation):
        args_tmp = args_dict[sampler][0].copy()
        with expectation:
            args_tmp['sampler'] = sampler
            assert calibrator(emu=emu_lin_pcgp,
                              y=input2,
                              x=input3,
                              thetaprior=input4,
                              method='directbayes',
                              yvar=input5,
                              args=args_tmp) is not None

    @pytest.mark.parametrize(
        "input2,input3,input4,input5",
        [
            (y, x, priorphys_lin, obsvar)
        ]
    )
    def test_cal_emu(self, sampler, emu_lin_pcgp, input2, input3, input4, input5):
        args_tmp = args_dict[sampler][0].copy()
        with does_not_raise():
            args_tmp['sampler'] = sampler
            assert calibrator(emu=emu_lin_pcgp,
                              y=input2,
                              x=input3,
                              thetaprior=input4,
                              method='directbayes',
                              yvar=input5,
                              args=args_tmp) is not None

    @pytest.mark.parametrize(
        "input2,input3,input4,input5,input6,expectation",
        [
            (y, x, priorphys_lin, 'XXXX', obsvar, pytest.raises(ValueError)),
        ],
    )
    def test_cal_method1(self, sampler, emu_lin_pcgp, input2, input3, input4, input5, input6, expectation):
        with expectation:
            assert calibrator(emu=emu_lin_pcgp,
                              y=input2,
                              x=input3,
                              thetaprior=input4,
                              method=input5,
                              yvar=input6,
                              args={'sampler': sampler}) is not None

    def test_repr(self, sampler, emu_lin_pcgp):
        args_tmp = args_dict[sampler][0].copy()
        args_tmp['sampler'] = sampler
        cal = calibrator(emu=emu_lin_pcgp,
                         y=y,
                         x=x,
                         thetaprior=priorphys_lin,
                         method='directbayes',
                         yvar=obsvar,
                         args=args_tmp)
        with does_not_raise():
            assert repr(cal) is not None
