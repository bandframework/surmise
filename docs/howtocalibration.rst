How to include a new calibrator
==============================================
When loaded, the surmise package automatically identifies all available
calibrators by locating all ``.py`` codes in the ``calibrationmethods`` folder
within its installation [#f1]_.  For example, it assumes that the
``calibrationmethods/directbayes.py`` file provides the ``directbayes``
calibrator, which is the calibrator name that users should provide when
choosing a particular calibrator to use for a fit.  Therefore, users can
integrate their own calibrator in surmise by placing their calibrator's Python
source code in that same folder.

This tutorial describes how to structure a
custom calibrator code so that this integration is successful.
We illustrate this with ``directbayeswoodbury``--a calibrator method located
in the directory ``calibrationmethods``.

In surmise, all calibrator methods inherit from the base class
:py:class:`surmise.calibrator`. A calibrator class calls the user
input method, and fits the corresponding calibrator.
:py:meth:`surmise.calibrator.fit` is the main
:py:class:`surmise.calibrator` class method. It also provides the
functionality of updating and manipulating the fitted calibrator by
:py:meth:`surmise.calibrator.predict` class methods.

In order to use the functionality of the base class :py:class:`surmise.calibrator`, we categorize the functions to be included in a new calibrator method (for example, ``directbayeswoodbury``) into two categories.

Mandatory functions
++++++++++++++++++++

:py:func:`fit` is the only obligatory function for a calibration
method. :py:func:`fit` takes the fitted emulator class object
:py:class:`surmise.emulator`, inputs :math:`\mathbf{X}`, and
observed values :math:`\mathbf{y}`, where :math:`\mathbf{X}\in\mathbb{R}^{N\times p}`,
:math:`\mathbf{y}\in\mathbb{R}^{N\times 1}`, and the dictionary ``fitinfo`` to
place the fitting information once complete. This dictionary is used to keep the
information that will be used by :py:func:`predict` below.


The :py:func:`surmise.calibrationmethods.directbayeswoodbury.fit` is given below for illustration:

.. currentmodule:: surmise.calibrationmethods.directbayeswoodbury

.. autofunction:: fit

Once the calibration method is fitted, the base
:py:class:`surmise.calibrator` assigns :py:attr:`surmise.calibrator.theta`
as an attribute of the class object to communicate with the fitted method through
general expressions. The attribute :py:attr:`surmise.calibrator.theta`
has methods :py:meth:`surmise.calibrator.theta.mean`,
:py:meth:`surmise.calibrator.theta.var`,
:py:meth:`surmise.calibrator.theta.rnd`, and
:py:meth:`surmise.calibrator.theta.lpdf`, which can be called
once the user obtains the fitted calibrator.

Those expressions are defined within the base class to simplify the usage of the fitted
models. To use these methods, calibration method developers should implement any of the
functions below in their method, or define the matching keys in the ``fitinfo``
dictionary.

========================  ================
Function                  ``fitinfo`` key
========================  ================
:py:func:`thetamean`      ``thetamean``
:py:func:`thetavar`       ``thetavar``
:py:func:`thetarnd`       ``thetarnd``
:py:func:`thetalpdf`      ``thetalpdf``
========================  ================

An example is the :py:func:`thetalpdf` function provided from the ``directbayeswoodbury``:

.. autofunction:: thetalpdf

Optional functions
++++++++++++++++++++

.. autofunction:: predict

Adding the covariance checks to a new calibration method
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The helpers for checking covariance are available in ``surmise.calibrationmethods._cov_diagnosis``. A new method
creates the record before sampling, checks each evaluation inside its
likelihood, and warns after sampling:

.. code-block:: python

    from ._cov_diagnosis import (new_cov_diagnosis, check_eigvals,
                                 warn_cov_diagnosis)

    def fit(fitinfo, emu, x, y, **sampler_args):
        ...
        fitinfo['cov_diagnosis'] = new_cov_diagnosis()
        results = sampler(...)
        warn_cov_diagnosis(fitinfo['cov_diagnosis'], 'mymethod')

    def loglik(fitinfo, emu, theta, y, x):
        ...
        for k in range(theta.shape[0]):
            W, V = np.linalg.eigh(np.eye(J.shape[1]) + J.T @ J)
            cov_diagnosis = fitinfo['cov_diagnosis']
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

.. rubric:: Footnotes

.. [#f1] The location of a surmise installation that was installed into a virtual environment, for example, might be ``~/local/venv/my_surmise/lib/python3.14/site-packages/surmise`` or, in Windows, ``~/local/surmise_venv/Lib/site-packages/surmise``
