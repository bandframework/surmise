``tox`` Developer Environments
==============================
.. _tox: https://tox.wiki
.. _webinar: https://www.youtube.com/watch?v=PrAyvH-tm8E

The tool tox_ has been integrated in the repository.  Developers can optionally
use this tool to execute different tasks as well as to setup and manage Python
virtual environments dedicated to different development purposes based on the
configuration managed by surmise maintainers.  Developers that would like to
use our ``tox`` environments should learn at the very least the difference
between calling ``tox -r -e <task>`` and ``tox -e <task>``.

To use ``tox``, developers must first install it.  The following procedure
installs ``tox`` in a dedicated, minimal virtual environment and is based on a
webinar_ presented by Oliver Bestwalter.

Execute the following with changes made to adapt to the developer's needs

.. code-block:: console

    $ /path/to/target/python --version
    $ /path/to/target/python -m venv $HOME/.toxbase
    $ $HOME/.toxbase/bin/pip list
    $ $HOME/.toxbase/bin/python -m pip install --upgrade pip
    $ $HOME/.toxbase/bin/python -m pip install --upgrade setuptools
    $ $HOME/.toxbase/bin/python -m pip install tox
    $ $HOME/.toxbase/bin/tox --version
    $ $HOME/.toxbase/bin/pip list

Alter ``PATH`` so that ``tox`` is immediately available.  To follow Oliver's
suggestion, execute some variation of

.. code-block:: console

    $ mkdir $HOME/local/bin
    $ ln -s $HOME/.toxbase/bin/tox $HOME/local/bin/tox
    $ vi $HOME/.bash_profile (add $HOME/local/bin to PATH)
    $ . $HOME/.bash_profile
    $ which tox
    $ tox --version

By default, ``tox`` will not carry out any work if only ``tox`` or ``tox -r``
is executed.  The following tasks can be run from the root of any surmise
clone.

* ``tox -e coverage``

    * Run the full surmise test suite with coverage-by-line enabled and
      using the current state of the code in the local clone
    * The coverage-by-line results are stored for generating a coverage report.
      See ``tox -e report`` below.

* ``tox -e nocoverage``

    * Run the full surmise test suite with surmise installed as a Python package

* ``tox -e emu_cal``

    * Run the standard emulator/calibrator tests of code integration using the
      current state of the code in the local clone

* ``tox -e new_emu``

    * Run the new emulator tests using the current state of the code in the
      local clone

* ``tox -e new_cal``

    * Run the new calibrator tests using the current state of the code in the
      local clone

* ``tox -e report``

    * Generate an HTML-format coverage-by-line report for inspection.  This
      assumes that ``coverage`` was already called or is called at the same
      time.

* ``tox -e check``

    * Run checks on the code and report any possible issues.  Note that this
      task does **not** alter the contents of any files.

* ``tox -e html``

    *  Generate in ``docs/build_html`` the package's sphinx-based documentation
       in HTML format 

* ``tox -e pdf``

    *  Generate in ``docs/build_pdf`` the package's sphinx-based documentation
       as a PDF file

Developers can also run multiple tasks at once by executing, for example,

.. code-block::

    tox -e report,coverage

Note that developers can directly use the virtual environments created by
``tox`` by activating them as usual.  For example, to use a virtual environment
with surmise installed in editable or developer mode (i.e., using ``pip install
-e surmise``), run ``tox -e coverage`` if you have not already and execute

.. code-block:: console

    . /path/to/surmise/.tox/coverage/bin/activate
