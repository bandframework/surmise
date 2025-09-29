Usage Examples
==============

The Python scripts and the corresponding notebooks of the examples are located in
``\examples`` directory.  In addition, a full emulation and calibration example with
Gaussian process models can be found at `surmise Jupyter notebook`_.

In addition, for a gentle introduction of emulation and calibration using Gaussian processes, visit
`surmise Jupyter notebook`_.

Examples linked below require `matplotlib` as an additional plotting package to visualize results, which can be installed via

.. code-block:: console

    $ pip install matplotlib

Example 1 (A falling ball example with two physics models)
##########################################################

The falling ball example illustrates the usage of both ``emulator`` and
``calibrator`` in conducting the calibration of two possible models in
describing the falling ball mechanics.

`Link to falling ball example`_

Example 2 (Emulation of nuclear physics simulation)
###################################################

This example illustrates the usage of Principal Component Stochastic Kriging model (PCSK) on simulation data from a
Viscous Anisotropic Hydrodynamic model via ``surmise``'s ``emulator`` object.

`Link to VAH model emulation`_

Example 3 (Acceleration due to gravity)
##################################################

This example is discussed in Chapter 8 in `Gramacy, 2020 <https://bookdown.org/rbg/surrogates/chap8.html>`_.

It demonstrates how to use ``surmise``'s  ``emulator`` and  ``calibrator`` objects.

`Link to gravity example`_

Example 4 (Emulation and calibration of epidemic model)
#######################################################

This example illustrates the Bayesian parameter inference of Susceptible-Infected
Recovered (SIR) type epidemic model via ``surmise``'s  ``emulator`` and ``calibrator`` objects.

Although there are many model parameters, we estimated most of them based on the epidemiological studies of COVID-19, and infer only 10 influential parameters in this example.

`Link to epidemic model calibration`_


.. _`surmise Jupyter notebook`: https://colab.research.google.com/drive/1f4gKTCLEAGE8r-aMWOoGvY-O6zNqg1qj?usp=drive_link
.. _`Link to falling ball example`: https://to_be_updated_to_gh_page
.. _`Link to gravity example`: https://nbviewer.jupyter.org/github/bandframework/surmise/blob/main/examples/Example3/Example3_nb.ipynb
.. _`Link to VAH model emulation`: https://nbviewer.org/github/bandframework/surmise/blob/main/examples/bayesian_vah/emulation_and_calibration/emulator_validation.ipynb
.. _`Link to epidemic model calibration`: https://nbviewer.jupyter.org/github/bandframework/surmise/blob/main/examples/Example4/Example4_nb.ipynb
