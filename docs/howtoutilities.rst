How to write a new utility routine
==============================================

In this tutorial, we describe how to include a new utility (specifically, a sampler)
to the surmise's framework. We illustrate this with ``metropolis_hastings``--a
well-known sampler located in the directory ``\utilitiesmethods``.

In surmise, all utilities inherit from the base class :py:class:`surmise.utilities.sampler`.
Note that for now, we only have samplers as utilities. Later, we plan to have
different classes (such as :py:class:`surmise.utilities.optimizer`) that can be
used during calibration.

A sampler takes the function returning the log of the posterior of given theta as
an input, and returns a dictionary including a random sample of thetas from the
posterior distribution. 

Mandatory functions
++++++++++++++++++++

:py:func:`sampler` is the only obligatory function for a sampler.

The :py:func:`metropolis_hastings.sampler` is given below for an illustration:

.. automodule:: metropolis_hastings
    :members:
    :undoc-members:
    :show-inheritance:

Once the base class :py:class:`surmise.utilities.sampler` is initialized,
:py:func:`surmise.utilities.sampler.draw_samples` method calls the developer's
sampler's :py:func:`sampler`
function, and places all information into the dictionary, and returns it.

Optional functions
++++++++++++++++++++

None. This section is under development.
