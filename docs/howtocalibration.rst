How to include a new calibrator
==============================================

In this tutorial, we describe how to include a new calibrator to the surmise's
framework. We illustrate this with ``directbayeswoodbury``--a calibrator method located
in the directory ``\calibrationmethods``.

In surmise, all calibrator methods inherit from the base class
:py:class:`surmise.calibration.calibrator`. A calibrator class calls the user
input method, and fits the corresponding calibrator.
:py:meth:`surmise.calibration.calibrator.fit` is the main
:py:class:`surmise.calibration.calibrator` class methods. It also provides the
functionality of updating and manipulating the fitted calibrator by
:py:meth:`surmise.calibration.calibrator.predict` class methods.

In order to use the functionality of the base class :py:class:`surmise.calibration.calibrator`, we categorize the functions to be included in a new emulation method (for example, ``directbayeswoodbury``) into two categories.

Mandatory functions
++++++++++++++++++++

:py:func:`fit` is the only obligatory function for a calibration
method. :py:func:`fit` takes the fitted emulator class object
:py:class:`surmise.emulation.emulator`, inputs :math:`\mathbf{X}`, and
observed values :math:`\mathbf{y}`, where :math:`\mathbf{X}\in\mathbb{R}^{N\times p}`,
:math:`\mathbf{y}\in\mathbb{R}^{N\times 1}`, and the dictionary ``fitinfo`` to
place the fitting information once complete. This dictionary is used to keep the
information that will be used by :py:func:`predict` below.


The :py:func:`surmise.calibrationmethods.directbayeswoodbury.fit` is given below for illustration:

.. currentmodule:: surmise.calibrationmethods.directbayeswoodbury

.. autofunction:: fit

Once the calibration method is fitted, the base
:py:class:`surmise.calibration.calibrator` assigns :py:attr:`surmise.calibration.calibrator.theta`
as an attribute of the class object to communicate with the fitted method through
general expressions. The attribute :py:attr:`surmise.calibration.calibrator.theta`
has methods :py:meth:`surmise.calibration.calibrator.theta.mean`,
:py:meth:`surmise.calibration.calibrator.theta.var`,
:py:meth:`surmise.calibration.calibrator.theta.rnd`, and
:py:meth:`surmise.calibration.calibrator.theta.lpdf`, which can be called
once the user obtains the fitted calibrator.

Those expressions are defined within the base class to simplify the usage of the fitted
models. In order to use those methods, the calibration method developers should either
include functions :py:func:`thetamean`, :py:func:`thetavar`, :py:func:`thetarnd`,
and/or, :py:func:`thetalpdf` in their methods, or define within the dictionary
``fitinfo`` using the keys ``thetamean``, ``thetavar``, ``thetarnd``, and/or, ``thetalpdf``.

An example is the :py:func:`thetalpdf` function provided from the ``directbayeswoodbury``:

.. autofunction:: thetalpdf

Optional functions
++++++++++++++++++++

.. autofunction:: predict

Diagnosing covariance problems
++++++++++++++++++++++++++++++++

The Gaussian likelihoods in ``directbayes`` and ``directbayeswoodbury`` involve factorizing a
covariance matrix. When the emulator returns a
predictive covariance that is not positive definite, or contains ``nan`` or
``inf``, the log-likelihood becomes ``nan`` and causes downstream issue. This can
happen at parameters where the emulator variance is close to zero and it can depend on
the platform and linear algebra library.

The results of
the checks for such ``nan`` likelihoods are stored in ``fitinfo['cov_diagnosis']``, available after fitting the calibrator as
``cal.info['cov_diagnosis']``. After sampling, ``fit`` will print a
``RuntimeWarning`` that reports any counts of non-singular matrix inversion, or if any covariance inversion is impacted by machine precision (see ``n_below_bound`` below).

.. code-block:: python

    cal = calibrator(emu=emu, y=y, x=x, thetaprior=prior,
                     method='directbayeswoodbury', yvar=obsvar,
                     args={'sampler': 'metropolis_hastings', ...})

    cov_diagnosis = cal.info['cov_diagnosis']
    rejected = cov_diagnosis['n_nonpd'] + cov_diagnosis['n_nonfinite']
    print(f"{rejected} of {cov_diagnosis['n_eval']} evaluations rejected")
    print(cov_diagnosis['theta'][:5])   # parameters that failed
    print(cov_diagnosis['eig'][:5])     # their three smallest eigenvalues

The record contains the following keys.

=================  ================================================================
Key                Meaning
=================  ================================================================
``n_eval``         Number of likelihood evaluations checked.
``n_nonfinite``    Rejected because the mean, covariance factor, or eigenvalues
                   contain ``nan`` or ``inf``.
``n_nonpd``        Rejected because the smallest eigenvalue is at or below
                   round-off tolerance.
``n_below_bound``  Kept, but the smallest eigenvalue fell below its exact lower
                   bound by more than round-off. This signals lost accuracy.
``min_eig``        Smallest eigenvalue seen across all finite evaluations.
``theta``          Up to ``max_store`` parameters that were rejected.
``eig``            The three smallest eigenvalues for each stored parameter.
``max_store``      Maximum number of stored parameters (default 50).
=================  ================================================================

A rejected parameter is treated as having zero posterior density. A few
rejections in low posterior probability regions are not uncommon.  However, if there are many rejections, or rejections near
the posterior mode, the emulator should be
revisited (for example a larger nugget lower bound or fewer, better spread
training points). To stop on the first problem instead of continuing, promote
the warning to an error:

.. code-block:: python

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        cal = calibrator(...)

The record is reset at the start of every ``fit``. Evaluations made
afterwards, for example through ``cal.theta.lpdf``, are added to the same record.

Adding the checks to a new calibration method
------------------------------------------------

The helpers are available in ``surmise.calibrationmethods._cov_diagnosis``. A new method
creates the record before sampling, checks each evaluation inside its
likelihood, and warns after sampling:

.. code-block:: python

    from ._cov_diagnosis import (new_cov_diagnosis, check_eigvals,
                                 warn_cov_diagnosis)

    def fit(fitinfo, emu, x, y, **sampler_args):
        ...
        fitinfo['cov_diagnosis'] = new_cov_diagnosis()
        results = sampler(logpost_func=logpostfull, ...)
        warn_cov_diagnosis(fitinfo['cov_diagnosis'], 'mymethod')

    def loglik(fitinfo, emu, theta, y, x):
        ...
        for k in range(theta.shape[0]):
            W, V = np.linalg.eigh(np.eye(J.shape[1]) + J.T @ J)
            cov_diagnosis = fitinfo.setdefault('cov_diagnosis',
                                               new_cov_diagnosis())
            if not check_eigvals(cov_diagnosis, theta[k], W,
                                 lower_bound=1.0, arrays=(m0, S0)):
                loglik[k, 0] = -np.inf
                continue
            ...

A method that also returns gradients should set the gradient of a rejected
parameter to zero so that gradient-based samplers do not receive ``nan``.

.. currentmodule:: surmise.calibrationmethods._cov_diagnosis

.. autofunction:: new_cov_diagnosis

.. autofunction:: check_eigvals

.. autofunction:: warn_cov_diagnosis
