Overview
==============================================

surmise is a Python library for a modular calibration framework. Bayesian
framework allows reducing computer code predictive uncertainty by calibrating
parameters directly to the observational data.

The typical Bayesian workflow consists of three main steps:

* determination of a prior distribution to express the the background knowledge or expert opinion,
* identifying the likelihood function with the observational data,
* combining both the prior distribution and the likelihood function using Bayes’ theorem in the form of the posterior distribution.

However, implementing Bayesian calibration for high-dimensional data and complex
systems is very challenging. Because the posterior distribution often cannot be
obtained analytically, an approximate Bayesian inference should be used.
The most popular method for sampling from high-dimensional posterior distributions is
Markov chain Monte Carlo (MCMC). However, for even relatively fast computer
models implementation of Bayesian inference with MCMC would simply take too long.
A very fast approximation to the system code is thus required to use the
Bayesian approach. Surrogate models (or emulators) emulate
the behavior of the input/output relationship of the computer model and
are computationally inexpensive to allow MCMC sampling to be possible.

The purpose of surmise is to provide a modular model calibration software that
can input model observations and output draws.

surmise’s work is categorized into three routines:

* :ref:`emulation<emulation>`: Carries out Bayesian emulation of computer model output and generates inputs to ``calibration``

* :ref:`calibration<calibration>`: Generates estimates of the calibration parameters based onfield observations of the real process and an output from ``emulation``

* :ref:`utilities<utilities>`: Performs different utility tasks such as a sampler (e.g., Metropolis-Hastings) to generate posterior draws of calibration parameter

Examples of how to use ``emulation``, ``calibration``, ``utilities`` modules can be found in
the ``examples/`` directory.
