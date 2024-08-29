
|

.. image:: https://badge.fury.io/py/surmise.svg
    :target: https://badge.fury.io/py/surmise

.. image:: https://readthedocs.org/projects/surmise/badge/?version=latest
   :target: https://surmise.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://github.com/bandframework/surmise/actions/workflows/python-package.yml/badge.svg
    :target: https://github.com/bandframework/surmise/actions/workflows/python-package.yml

.. image:: https://coveralls.io/repos/github/bandframework/surmise/badge.svg
    :target: https://coveralls.io/github/bandframework/surmise

|

.. after_badges_rst_tag

===========================
Introduction to surmise
===========================

surmise is a Python package that is designed to provide a surrogate model
interface for calibration, uncertainty quantification, and sensitivity analysis.

Dependencies
~~~~~~~~~~~~

surmise is built with numpy and scipy, with an optional scikit-learn dependency.

Please refer to the [project] and [project.optional-dependencies] sections of pyproject.toml 
for details.

Installation
~~~~~~~~~~~~

From the command line, use one of the following commands to install surmise::

 pip install surmise
 pip install surmise[scikit-learn]      # to include scikit-learn in installation
 pip install surmise[all]               # to include all optional dependencies

The package scikit-learn is required by specific methods as stated above.
These packages can be installed along with surmise via the commands listed.

The list of available .whl files can be found under `PyPI-wheel`_.  If a wheel file
for your preferred platform is not listed, surmise has to be built from source,
which requires extra dependencies::

 git clone https://github.com/bandframework/surmise/
 cd surmise
 pip install build Cython
 pip install scikit-learn (optional, required by full test suite)
 python -m build --wheel
 pip install dist/surmise-<version info>.whl

.. note::

    Direct installation of surmise requires Cython to build C executable.
    On a Windows platform Cython is supported by Microsoft build tools, for which installation
    is outside pip; see `Microsoft build tools`_ for details.

Testing
~~~~~~~

Testing of surmise can be performed after cloning the repository. The test suite requires the pytest_,
pytest-cov_, and scikit-learn_ packages to be installed.  These packages can be installed via::

 pip install pytest pytest-cov scikit-learn

The test suite can then be run from within the ``tests/`` directory of the source distribution by running::

 cd tests
 ./run-tests.sh

Further options are available for testing. To see a complete list of options, run::

 ./run-tests.sh -h

Coverage reports are produced under ``tests/cov_html`` directory only if all tests are used.

Documentation
~~~~~~~~~~~~~

The documentation is stored in ``docs/`` and is hosted at `Read the Docs <http://surmise.readthedocs.io>`_.

Users and developers that would like to generate the documentation locally are
encouraged to use ``tox``, which automatically creates a dedicated,
fully-functioning virtual environment for the task.  Refer to the online
developer's guide (or ``docs/tox.rst``) for help setting up ``tox`` and using
it to generate documentation.


**Citation:**

- Please use the following to cite surmise in a publication:

.. code-block:: bibtex

   @techreport{surmise2023,
     author      = {Matthew Plumlee and \"Ozge S\"urer and Stefan M. Wild and Moses Y-H. Chan},
     title       = {{surmise 0.2.1+dev} Users Manual},
     institution = {NAISE},
     number      = {Version 0.2.1+dev},
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
.. _scikit-learn: https://scikit-learn.org/stable/install.html
