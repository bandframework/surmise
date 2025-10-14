=============
Release Notes
=============

Below are the notes from all surmise releases.

Release 0.4.0
-------------

:Date: October 14, 2025

* Improve coverage of test suite for future integration of additional code.
* Designate methods to "research-grade" if methods are neither fully tested or maintained.
* Integrate a first example in Jupyterbook as a collection of usage examples.
* Improve Cython integration by reducing the number of external dependencies.
* Develop Github wiki pages to log developer notes in terms of packaging, Cythonizing, and major dependency intricacies.

**Full Changelog**: https://github.com/bandframework/surmise/compare/v0.3.1...v0.4.0

Release 0.3.0
-------------

:Date: September 12, 2024

* Tested in Python 3.9, 3.10, 3.11, 3.12, on Linux, macOS, and Windows.
* Functionality:
  * Include tox developer integration.
  * Include save/load functions for emulator and calibrator.
  * Include empirical coverage calculations in calibrator.
  * Modify `covx()`-returned object to improve usability.
  * Modify object `repr` to improve user feedback.
  * Update test suite to improve testing, including bookending versions and testing of major methods.
* Documentation:
  * Update documentation to emphasize introductory materials and usage tutorials.
  * Update scientific usage references.
  * Update README to include collaborator and contributor list.

**Full Changelog**: https://github.com/bandframework/surmise/compare/v0.2.1...v0.3.0

Release 0.2.1
-------------

:Date: September 26, 2023

* Updates README to improve installation and testing procedures.
* Updates Github action to build and upload .whl distribution files to PyPI.
* Fixes testing procedure to ensure shell script searches under `sys.path` for `pytest`.

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
