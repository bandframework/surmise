=============
Release Notes
=============

Below are the notes from all surmise releases.

Release 0.2.0
-------------

:Date: September 18, 2023

* Emulation methods:
    * Rename ``PCGPwMatComp`` to ``PCGPwImpute``.
    * Include a new emulation method ``indGP``.
    * Include a new emulation method ``PCGPR``, which requires ``scikit-learn``.
    * Include a new emulation method ``PCSK``.
    * Include a new emulation method ``nuclear-ROSE``, for integration with Python package ``nuclear-rose``.
    * Remove ``GPy`` from the list of emulation methods.
* Calibration methods:
    * Modify ``directbayes`` to admit Python package ``ptemcee`` as sampler.
* Utilities methods:
    * Modify ``metropolis_hastings`` to allow control of console output.
    * Include a new sampling method ``PTLMC``.
    * Include a new sampling method ``PTMC``, using Python package ``ptemcee``.
* Test suite:
    * Remove the use of `python setup.py test` for unit testing.

Release 0.1.1
-------------

:Date: July 9, 2021

* Include a new emulation method PCGPwM.py integrated with Cython (see ``surmise\emulationmethods``).
* Include a new emulation method PCGPwMatComp.py (see ``surmise\emulationmethods``).
* Include a new calibration method simulationpost.py (see ``surmise\calibrationmethods``).
* Include a new calibration method mlbayeswoodbury.py (see ``surmise\calibrationmethods``).
* Include a new sampling method PTLMC.py (see ``surmise\utilitiesmethods``).
* Update GPy.py to handle high-dimensional data and allow nan values (see ``surmise\emulationmethods``).
* Examples are updated to illustrate the new methods (see ``\examples``).
* Documentation is improved to provide a developer guide (see ``docs\contributing.rst``).

Release 0.1.0
-------------

:Date: February 6, 2021

* Initial release.
