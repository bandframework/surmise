Running Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``surmise`` uses GitHub Actions that allow to set up Continuous Integration workflows.
After each push and pull request to develop and master branch, the code will be tested
with automated workflows, and the Python unit tests under ``tests/test_emu_cal``
will automatically be running. Reviewer will check that all tests have passed and will then approve merge.

The test suite requires the pytest_ and pytest-cov_ packages to be installed and
can all be run from the ``tests/`` directory of the source distribution by
running::

./run-tests.sh

Coverage reports are produced under the relevant directory only if all tests are
used.

If you have the source distribution, you can run the tests in the top-level
directory containing the setup script with ::

 python setup.py test

Further options are available for testing. To see a complete list of options,
run::

 ./run-tests.sh -h

To run tests only for the emulation module, run::

 ./run-tests.sh -e

To run specific tests only for the emulation module, provide a list of test
scripts, and run for example::

  ./run-tests.sh -e -l 'test_emu_init.py'

To run tests only for the calibration module, run::

 ./run-tests.sh -c

To run specific tests only for the calibration module, provide a list of test
scripts, and run for example::

 ./run-tests.sh -c -l 'test_cal_mh.py'

To run tests for a certain emulator (e.g., 'PCGP' in ``emulationmethods/``
directory), run::

  ./run-tests.sh -a 'PCGP'

To run tests for a certain calibrator (e.g., 'directbayes' in ``calibrationmethods/``
directory), run::

 ./run-tests.sh -b 'directbayes'
