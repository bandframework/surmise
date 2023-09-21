Use cases
==============================================

Below are some expected surmise use cases that we support (or are working
to support) and plan to have examples of:

- A user wants to emulate a computationally expensive model at a limited number of points in parameter space and interpolate those results through the whole space. Then, a user can use one of the methods in the ``\emulationmethods`` directory.

- A user wants to emulate a computationally expensive model at a limited number of points in parameter space and interpolate those results through the whole space using a user-developed emulation method. Then, a user can drop the script of the method into the ``\emulationmethods`` directory, and use it within the surmise framework.

- A user wants to perform principled uncertainty quantification that calibrates the models against data. Then, a user can use one of the methods in the ``\calibrationmethods`` directory.

- A user wants to perform principled uncertainty quantification that calibrates the models against data via a user-developed calibration method. Then, a user can drop the script of the method into the ``\calibrationmethods`` directory, and use it within the surmise framework.

- A user wants to test a new sampler within the surmise framework. Then, a user can drop the script of the method into the ``\utilitiesmethods`` directory, and use it within the surmise framework.

Reference to examples
================================
Below is a non-exhaustive list of references utilizing surmise (in alphabetical order):

.. bibliography::
    :style: plain
    :all:
