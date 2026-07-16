import numpy as np
import pytest
from surmise.emulation import emulator

from .conftest import does_not_raise
from .shared_scenario import x_lin as x, theta_lin as theta, f_lin as f, \
    balldropmodel_linear, theta_test_lin as theta_test

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


##############################################
# Unit tests to initialize an emulator class #
##############################################
@pytest.mark.parametrize(
    "cmdopt1,expectation",
    [
     ('PCGP', does_not_raise()),
     ('PCGPwM', does_not_raise()),
     ('PCSK', does_not_raise())
    ],
    )
# tests for prediction class methods:
def test_accuracy(cmdopt1, expectation):
    if cmdopt1 == 'PCSK':
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1, args={'simsd': simsd})
    else:
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
    ftest = balldropmodel_linear(x.astype(float), theta_test)
    pred_test = emu.predict(x=x, theta=theta_test)

    print('\n')
    print('R2: (as close to one as possible)')
    rsq = (1 - np.mean((ftest - pred_test.mean()) ** 2) /
           np.mean((ftest - np.mean(ftest)) ** 2))
    print('test R2:', np.round(rsq, 2))

    print('RMSE : (as small as possible)')
    rmse = np.sqrt(np.mean((ftest - pred_test.mean()) ** 2))
    print('test rmse:', np.round(rmse, 2))

    print('mean((f-fhat)/sqrt(var)) (should be close to 0):')
    print(np.round(np.mean((ftest - pred_test.mean()) / np.sqrt(pred_test.var())), 2))

    print('mean((f-fhat)**2/var)(should be close to 1):')
    print(np.round(np.mean((ftest - pred_test.mean()) ** 2 / pred_test.var()), 2))

    with expectation:
        residstand = np.empty([50, pred_test.covxhalf().shape[2]])
        for k in range(0, 50):
            residstand[k, :] = (np.linalg.pinv(pred_test.covxhalf()[:, k, :]) @
                                (ftest[:, k] - pred_test.mean()[:, k]))
        print('average normalized value (should be close to 1)):')
        print(np.mean(residstand ** 2))


@pytest.mark.parametrize(
    "cmdopt1,expectation",
    [
     ('PCGP', pytest.raises(ValueError)),
     ('PCGPwM', does_not_raise()),
     # ('PCSK', pytest.raises(np.linalg.LinAlgError))  # unknown method issue
    ],
    )
# tests for prediction class methods:
def test_predlpdf(cmdopt1, expectation):
    if cmdopt1 == 'PCSK':
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1, args={'simsd': simsd})
    else:
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1)
    ftest = balldropmodel_linear(x.astype(float), theta_test)
    pred_test = emu.predict(x=x, theta=theta_test)

    with expectation:
        assert pred_test.lpdf(f=ftest) is not None


@pytest.mark.parametrize(
    "cmdopt1,expectation",
    [
     ('PCGPwM', does_not_raise()),
     # ('PCSK', pytest.raises(np.linalg.LinAlgError))  # unknown method issue
    ],
    )
# tests for prediction class methods:
def test_predlpdf_wgrad(cmdopt1, expectation):
    if cmdopt1 == 'PCSK':
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1, args={'simsd': simsd, 'return_grad': True})
    else:
        emu = emulator(x=x, theta=theta, f=f, method=cmdopt1, args={'return_grad': True})
    ftest = balldropmodel_linear(x.astype(float), theta_test)
    pred_test = emu.predict(x=x, theta=theta_test)

    with expectation:
        assert pred_test.lpdf(f=ftest) is not None
