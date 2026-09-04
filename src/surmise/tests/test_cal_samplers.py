import numpy as np
import scipy.stats as sps
import pytest
from surmise.calibration import calibrator

from .conftest import does_not_raise
from .shared_scenario import (x_lin as x, y_lin as y,
                              obsvar_lin as obsvar, priorphys_lin, RNG_SEED,
                              DEFAULT_MH_SPECS, DEFAULT_PTLMC_SPECS)

pytestmark = pytest.mark.usefixtures('seeded_rng')

_rng = np.random.default_rng(seed=RNG_SEED)

# TODO: LMC will require 'expertMode' in lmc_option to run
SAMPLERS_IN_TEST = ['metropolis_hastings', 'PTLMC']  # , 'LMC']

##############################################
#            Simple scenarios                #
##############################################

# setting obsvar
obsvar1 = obsvar[0:10]
obsvar2 = -obsvar
obsvar3 = 10 ** (10) * obsvar


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
    def nothing():
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


lmc_args1 = {'theta0': np.array([[0, 9]]),
             'numsamp': 50,
             'expertMode': True}

args_dict = {'metropolis_hastings': [DEFAULT_MH_SPECS],
             'PTLMC': [DEFAULT_PTLMC_SPECS],
             'LMC': [lmc_args1]}

##############################################
# Unit tests to initialize an emulator class #
##############################################

# Generate all test pairs accordingly upfront
SAMPLER_ARGS_PAIRS = [
    pytest.param(sampler, args, id=f"{sampler}-args{i}")
    for sampler in SAMPLERS_IN_TEST
    for i, args in enumerate(args_dict[sampler])
]


@pytest.mark.parametrize("sampler,args", SAMPLER_ARGS_PAIRS)
def test_cal_MLcal(sampler, args, emu_lin_pcgp):
    args_tmp = args.copy()
    args_tmp['sampler'] = sampler
    with does_not_raise():
        assert calibrator(emu=emu_lin_pcgp,
                          y=y,
                          x=x,
                          thetaprior=priorphys_lin,
                          method='directbayes',
                          yvar=obsvar,
                          args=args_tmp) is not None


@pytest.mark.parametrize('sampler', SAMPLERS_IN_TEST)
class TestSampler:

    @pytest.mark.parametrize(
        "thetaprior,obsvar,expectation",
        [
            (priorphys_lin, obsvar1, pytest.raises(ValueError)),
            (priorphys_lin, obsvar2, pytest.raises(ValueError)),
            (priorphys_lin, obsvar3, pytest.raises(ValueError)),
            (prior_rnd1, obsvar, pytest.raises(ValueError)),
            (prior_rnd2, obsvar, pytest.raises(AttributeError)),
            (prior_lpdf1, obsvar, pytest.raises(ValueError)),
            (prior_lpdf2, obsvar, pytest.raises(ValueError)),
            (prior_example1, obsvar, pytest.raises(ValueError)),
        ],
    )
    def test_cal_emu_fails(self, sampler, emu_lin_pcgp, thetaprior, obsvar, expectation):
        args_tmp = args_dict[sampler][0].copy()
        with expectation:
            args_tmp['sampler'] = sampler
            assert calibrator(emu=emu_lin_pcgp,
                              y=y,
                              x=x,
                              thetaprior=thetaprior,
                              method='directbayes',
                              yvar=obsvar,
                              args=args_tmp) is not None

    def test_cal_emu(self, sampler, emu_lin_pcgp):
        args_tmp = args_dict[sampler][0].copy()
        with does_not_raise():
            args_tmp['sampler'] = sampler
            assert calibrator(emu=emu_lin_pcgp,
                              y=y,
                              x=x,
                              thetaprior=priorphys_lin,
                              method='directbayes',
                              yvar=obsvar,
                              args=args_tmp) is not None

    def test_cal_invalid_method(self, sampler, emu_lin_pcgp):
        with pytest.raises(ValueError):
            assert calibrator(emu=emu_lin_pcgp,
                              y=y,
                              x=x,
                              thetaprior=priorphys_lin,
                              method='XXXX',
                              yvar=obsvar,
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
