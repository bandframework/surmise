How to include a new emulator
==============================================

In this tutorial, we describe how to include a new emulator to the surmise's
framework. We illustrate this with ``PCGP``--an emulator method located in the
directory ``\emulationmethods``.

In surmise, all emulator methods inherit from the base class :py:class:`surmise.emulation.emulator`.
An emulator class calls the user input method, and fits the corresponding
emulator. :py:func:`surmise.emulation.emulator.fit` and :py:func:`surmise.emulation.emulator.predict`
are the main :py:class:`surmise.emulation.emulator` class methods.
It also provides the functionality of updating and manipulating the
fitted emulator by :py:func:`surmise.emulation.emulator.supplement`,
:py:func:`surmise.emulation.emulator.update`, and :py:func:`surmise.emulation.emulator.remove`
class methods.

In order to use the functionality of the base class :py:class:`surmise.emulation.emulator`,
we categorize the functions to be included into a new emulation method (for example ``PCGP``) into two categories.

Mandatory functions
++++++++++++++++++++

:py:func:`fit` and :py:func:`predict` are the two obligatory functions for an emulation
method. :py:func:`fit` takes the inputs :math:`\mathbf{X}`, parameters :math:`\theta`,
and the function evaluations :math:`\mathbf{f}`, where :math:`\mathbf{X}\in\mathbb{R}^{N\times p}`,
:math:`\theta\in\mathbb{R}^{M\times d}`, and :math:`\mathbf{f}\in\mathbb{R}^{N\times M}`.
In other words, each column in :math:`\mathbf{f}` should correspond to a row in
:math:`\theta`. Each row in :math:`\mathbf{f}` should correspond to a row in :math:`\mathbf{X}`.
In addition, the dictionary ``fitinfo`` is passed to the :py:func:`fit` function to
place the fitting information once complete. This dictionary is used keep the
information that will be used by :py:func:`predict` below.

The :py:func:`PCGP.fit` is given below for an illustration:

.. currentmodule:: PCGP

.. autofunction:: fit

Once the base class :py:class:`surmise.emulation.emulator` is initialized,
:py:func:`surmise.emulation.emulator.fit` method calls the developer's emulator's :py:func:`fit`
function, and places all information into the dictionary ``fitinfo``.

.. autofunction:: predict

:py:func:`surmise.emulation.emulator.predict` method returns a prediction class object,
which has methods :py:meth:`surmise.emulation.prediction.mean`,
:py:meth:`surmise.emulation.prediction.mean_gradtheta`,
:py:meth:`surmise.emulation.prediction.var`,
:py:meth:`surmise.emulation.prediction.covx`,
:py:meth:`surmise.emulation.prediction.covxhalf`,
:py:meth:`surmise.emulation.prediction.covxhalf_gradtheta`,
:py:meth:`surmise.emulation.prediction.rnd`,
and :py:meth:`surmise.emulation.prediction.lpdf`.

Those expressions are defined within the base class to simplify the usage of the fitted
models. In order to use those methods, the emulation method developers should either
include functions :py:func:`predictmean`, :py:func:`predictmean_gradtheta`, :py:func:`predictvar`,
:py:func:`predictcovx`, :py:func:`predictcovxhalf`, :py:func:`predictcovxhalf_gradtheta`, :py:func:`predictrnd`,
and :py:func:`predictlpdf` into their methods, or define within the dictionary
``fitinfo`` using the keys ``mean``, ``mean_gradtheta``, ``var``,
``covx``, ``covxhalf``, ``covxhalf_gradtheta``, ``rnd``,
and ``lpdf``.



Optional functions
++++++++++++++++++++

``supplementtheta()`` is an optional function for an emulation method.
