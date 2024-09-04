Usage Examples
========

The Python scripts and the corresponding notebooks of the examples are located in
``\examples`` directory.  In addition, a full emulation and calibration example with
Gaussian process models can be found at `surmise Jupyter notebook`_.

In addition, for a gentle introduction of emulation and calibration using Gaussian processes, visit
`surmise Jupyter notebook`_.

Examples linked below require `matplotlib` as an additional plotting package to visualize results, which can be installed via

.. code-block:: console

    $ pip install matplotlib

Example 1
##################################################

To illustrate how the ``surmise``'s  ``emulator`` object works in practice, we
use the falling ball example.

`Link to Example 1 <https://nbviewer.jupyter.org/github/bandframework/surmise/blob/main/examples/Example1/Example1_nb.ipynb>`_.

Example 2
##################################################

To illustrate how the ``surmise``'s  ``calibrator`` object works in practice, we
use Example 1's falling ball example.

`Link to Example 2 <https://nbviewer.jupyter.org/github/bandframework/surmise/blob/main/examples/Example2/Example2_nb.ipynb>`_.

Example 3
##################################################

This example is discussed in Chapter 8 in `Gramacy, 2020 <https://bookdown.org/rbg/surrogates/chap8.html>`_.

It demonstrates how to use ``surmise``'s  ``emulator`` and  ``calibrator`` objects.

`Link to Example 3 <https://nbviewer.jupyter.org/github/bandframework/surmise/blob/main/examples/Example3/Example3_nb.ipynb>`_.

Example 4
##################################################

This example illustrates the Bayesian parameter inference of Susceptible-Infected
Recovered (SIR) type epidemic model via ``surmise``'s  ``emulator`` and ``calibrator`` objects.

Although there are many model parameters, we estimated most of them based on the epidemiological studies of COVID-19, and infer only 10 influential parameters in this example.

`Link to Example 4 <https://nbviewer.jupyter.org/github/bandframework/surmise/blob/main/examples/Example4/Example4_nb.ipynb>`_.

.. _`surmise Jupyter notebook`: https://colab.research.google.com/drive/1f4gKTCLEAGE8r-aMWOoGvY-O6zNqg1qj?usp=drive_link
