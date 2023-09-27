
|

.. image:: https://badge.fury.io/py/surmise.svg
    :target: https://badge.fury.io/py/surmise

.. image:: https://readthedocs.org/projects/surmise/badge/?version=latest
   :target: https://surmise.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://github.com/bandframework/surmise/actions/workflows/python-package.yml/badge.svg
    :target: https://github.com/bandframework/surmise/actions/workflows/python-package.yml

|

.. after_badges_rst_tag

===========================
Introduction to surmise
===========================

surmise is a Python package that is designed to provide a surrogate model
interface for calibration, uncertainty quantification, and sensitivity analysis.

Dependencies
~~~~~~~~~~~~
surmise is build for Python 3.8 or above, with the following dependencies:

* numpy>=1.18.3
* scipy>=1.7
* scikit-learn>=1.2.0

Installation
~~~~~~~~~~~~

From the command line, use the following command to install surmise::

 pip install surmise

The list of available .whl files can be found under `PyPI-wheel`_.  If a wheel file
for your preferred platform is not listed, surmise has to be built from source,
which requires extra dependencies::

 git clone https://github.com/bandframework/surmise/
 cd surmise
 pip install -r requirements.txt

.. note::

    Direct installation of surmise requires Cython to build C executable.
    On Windows platform Cython is supported by Microsoft build tools, for which installation
    is outside pip.  See `Microsoft build tools`_ for details.

Testing
~~~~~~~

The test suite requires the pytest_ and pytest-cov_ packages to be installed.  The packages
can be installed via::

 pip install pytest pytest-cov

The test suite can then be run from within the ``tests/`` directory of the source distribution by running::

 cd tests
 ./run-tests.sh

Further options are available for testing. To see a complete list of options, run::

 ./run-tests.sh -h

Coverage reports are produced under ``tests/cov_html`` directory only if all tests are used.

Documentation
~~~~~~~~~~~~~

The documentation is stored in ``docs/`` and is compiled with the Sphinx Python
documentation generator. It is written in the reStructuredText format. The
documentation is hosted at `Read the Docs <http://surmise.readthedocs.io>`_.

To compile the documentation, first ensure that Sphinx and its dependencies are installed.
To install Sphinx and/or ensure compatibility of dependencies, run ``make`` from a terminal within the ``docs/``
directory::

 cd docs
 make

To generate documentation, run command ``make html`` from a terminal within the ``docs/`` directory::

 (cd docs)
 make html

The HTML files are then stored in ``docs/_build/html``.


**Citation:**

- Please use the following to cite surmise in a publication:

.. code-block:: bibtex

   @techreport{surmise2023,
     author      = {Matthew Plumlee and \"Ozge S\"urer and Stefan M. Wild and Moses Y-H. Chan},
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

In addition, for a gentle introduction of emulation and calibration using Gaussian processes, visit
`surmise Jupyter notebook`_.

.. _NumPy: http://www.numpy.org
.. _pytest-cov: https://pypi.org/project/pytest-cov/
.. _pytest: https://pypi.org/project/pytest/
.. _Python: http://www.python.org
.. _SciPy: http://www.scipy.org
.. _`surmise Jupyter notebook`: https://colab.research.google.com/drive/1f4gKTCLEAGE8r-aMWOoGvY-O6zNqg1qj?usp=drive_link
.. _PyPI-wheel: https://pypi.org/project/surmise/#files
.. _`Microsoft build tools`: https://visualstudio.microsoft.com/downloads/?q=build+tools