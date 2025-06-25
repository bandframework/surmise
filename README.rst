
|

.. image:: https://badge.fury.io/py/surmise.svg
    :target: https://badge.fury.io/py/surmise

.. image:: https://readthedocs.org/projects/surmise/badge/?version=latest
   :target: https://surmise.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://github.com/bandframework/surmise/actions/workflows/python-installation.yml/badge.svg
    :target: https://github.com/bandframework/surmise/actions/workflows/python-installation.yml

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

surmise is built with numpy, scipy, and dill, with an optional scikit-learn dependency.

Please refer to the [_requires] sections of setup.py for details.

Installation
~~~~~~~~~~~~

From the command line, use one of the following commands to install surmise::

 pip install surmise
 pip install surmise[scikit-learn]      # to include scikit-learn in installation
 pip install surmise[all]               # to include all optional dependencies

The package scikit-learn is required by specific methods as stated above as well
as for testing an installation.  This package can be installed along with
surmise via the commands provided above.

The list of available .whl files can be found under `PyPI-wheel`_.  If a wheel
file for your preferred platform is not listed, then surmise has to be built
from source.  There is C code in the package that will be compiled for your
setup by `setuptools`_ during this process.  Therefore, a valid C compiler must
be installed beforehand.  In such cases, the installation should be built
automatically from the source distribution in PyPI when installed via pip
(**TBC**).  For those who prefer to work from a clone, please execute an
appropriate version of::

 git clone https://github.com/bandframework/surmise/
 cd surmise
 python -m pip install --upgrade pip
 python -m pip install --upgrade setuptools
 python -m pip install build
 python -m build --wheel
 python -m pip install scikit-learn (optional, required by full test suite)
 python -m pip install dist/surmise-<version info>.whl

.. note::
    Currently surmise on Windows system is built and tested with MinGW, in order to support the GCC compiler.
    The expectation is that other Windows build environments are compatible as well.

Testing
~~~~~~~

An automated test suite is integrated into the package so that users can
directly test their installation.  Some tests in the suite require the
installation of the otherwise optional scikit-learn_ package, which can be
installed as shown above or via::

 pip install scikit-learn

After ensuring that this additional dependence is installed, an installation can
be tested by executing, for example::

 $ python
 >>> import surmise
 >>> surmise.__version__
 <version string>
 >>> surmise.test()

The pytest_ output should indicate clearly if all tests passed or provide
information related to any failures otherwise.

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

   @techreport{surmise2024,
     author      = {Matthew Plumlee and \"Ozge S\"urer and Stefan M. Wild and Moses Y-H. Chan},
     title       = {{surmise 0.3.0} Users Manual},
     institution = {NAISE},
     number      = {Version 0.3.0},
     year        = {2024},
     url         = {https://surmise.readthedocs.io}
   }

Examples
~~~~~~~~

We provide examples in the ``examples/`` directory to illustrate the basic usage
of surmise.

In addition, for a gentle introduction of emulation and calibration using Gaussian processes, visit
`surmise Jupyter notebook`_.

.. _NumPy: http://www.numpy.org
.. _pytest: https://pypi.org/project/pytest/
.. _Python: http://www.python.org
.. _SciPy: http://www.scipy.org
.. _Setuptools: https://setuptools.pypa.io
.. _`surmise Jupyter notebook`: https://colab.research.google.com/drive/1f4gKTCLEAGE8r-aMWOoGvY-O6zNqg1qj?usp=drive_link
.. _PyPI-wheel: https://pypi.org/project/surmise/#files
.. _scikit-learn: https://scikit-learn.org/stable/install.html

