.. surmise documentation main file, created by
   sphinx-quickstart on Thu Jan 14 16:50:19 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to surmise's documentation!
====================================

.. image:: images/surmise-named-logo.png
    :align: center
    :alt: surmise

surmise is a Python library for facilitating Bayesian calibration with statistical emulation.  surmise's modular design
allows for mix-and-matching emulation and calibration strategies for a specific scientific problem.

To begin using surmise, we encourage checking out the following pages:

* :doc:`Quickstart <introduction>`
* :doc:`Basic usage examples <examples>`
* `Jupyter notebook`_: Full usage with Gaussian process emulation on Google Colab.
* :doc:`Expected use case examples <use_cases>` and scientific examples.
* `Github project page`_: surmise is open source and provided under the MIT license.

.. toctree::
   :caption: User Guide:
   :maxdepth: 4
	
   Quickstart<introduction>
   understanding	
   programming_surmise

.. toctree::
   :maxdepth: 2
   :caption: Examples & Tutorials:

   examples
   tutorials

.. toctree::
   :caption: Developer Guide:
   :maxdepth: 4

   contributing

.. toctree::
   :caption: Collaborators & Contributors:
   :maxdepth: 2

   collaborators_contributors

Indices and tables
====================================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _`Github project page`: https://github.com/bandframework/surmise
.. _`Jupyter notebook`: https://colab.research.google.com/drive/1f4gKTCLEAGE8r-aMWOoGvY-O6zNqg1qj?usp=drive_link