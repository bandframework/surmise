
|

.. image:: https://badge.fury.io/py/surmise.svg
    :target: https://badge.fury.io/py/surmise

.. image:: https://readthedocs.org/projects/surmise/badge/?version=latest
   :target: https://surmise.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://github.com/surmising/surmise/actions/workflows/python-package.yml/badge.svg
    :target: https://github.com/surmising/surmise/actions/workflows/python-package.yml

|

.. after_badges_rst_tag

===========================
Introduction to surmise
===========================

surmise is a Python package that is designed to provide a surrogate model
interface for calibration, uncertainty quantification, and sensitivity analysis.

Dependencies
~~~~~~~~~~~~
* numpy>=1.18.3
* scipy>=1.7
* scikit-learn>=1.2.0

Installation
~~~~~~~~~~~~

From the command line, use the following command to install surmise::

 pip install surmise

Alternatively, the source code can be downloaded to the local folder, and the
package can be installed from the .tar file.

Testing
~~~~~~~

The test suite requires the pytest_ and pytest-cov_ packages to be installed
and can be run from the ``tests/`` directory of the source distribution by running::

 ./run-tests.sh

If you have the source distribution, you can run the tests in the top-level
directory containing the setup script with ::

 python setup.py test

Further options are available for testing. To see a complete list of options, run::

 ./run-tests.sh -h

Coverage reports are produced under the relevant directory only if all tests are used.

Documentation
~~~~~~~~~~~~~

The documentation is stored in ``docs/`` and is compiled with the Sphinx Python
documentation generator. It is written in the reStructuredText format. These
files are hosted at `Read the Docs <http://surmise.readthedocs.io>`_.

To compile the documentation, first ensure that Sphinx is installed. Then, to
generate documentation, run command ``make html`` from terminal within this directory as follows ::

 cd docs
 make html

The HTML files are then stored in ``docs/_build/html``


**Citation:**

- Please use the following to cite surmise in a publication:

.. code-block:: bibtex

   @techreport{surmise2023,
     author      = {Matthew Plumlee and Özge Sürer and Moses Y-H. Chan and Stefan M. Wild},
     title       = {{surmise 0.2.0} Users Manual},
     institution = {NAISE},
     number      = {Version 0.2.0},
     year        = {2023},
     url         = {https://surmise.readthedocs.io}
   }

Examples
~~~~~~~~

We provide examples in the ``examples/`` directory to illustrate the basic usage
of surmise.

.. _NumPy: http://www.numpy.org
.. _pytest-cov: https://pypi.org/project/pytest-cov/
.. _pytest: https://pypi.org/project/pytest/
.. _Python: http://www.python.org
.. _SciPy: http://www.scipy.org
