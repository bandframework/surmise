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


The :py:func:`directbayeswoodbury.fit` is given below for illustration:

.. currentmodule:: directbayeswoodbury

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
