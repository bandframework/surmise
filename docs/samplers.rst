.. _samplers:

samplers
========
Users typically do not need to access samplers directly.  Instead, they inform
calibrators which sampler should be used and provide the calibrators with the
full set of configuration values required by their sampler.  This interface
information is provided so that users can determine which sampler they would
like to use and how to configure it for their needs.

.. autofunction:: surmise.utilitiesmethods.metropolis_hastings.sampler
.. autofunction:: surmise.utilitiesmethods.LMC.sampler
.. autofunction:: surmise.utilitiesmethods.PTLMC.sampler
