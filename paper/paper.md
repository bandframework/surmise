---
title: 'surmise: A modular Python interface for surrogate models and Bayesian calibration'
tags:
  - Python
  - surrogate models
  - Gaussian processes
  - Bayesian calibration
  - uncertainty quantification
authors:
  - name: Moses Y.-H. Chan^[Corresponding author]         # fill in the actual dev team + ORCIDs
    orcid: 0000-0002-4188-8953
    affiliation: 1
  - name: Jared O'Neal
    orcid: 0000-0003-2603-7314
    affiliation: 2
  - name: \"Ozge S\"urer
    orcid: 0000-0003-4854-9759
    affiliation: 4
  - name: Stefan M. Wild
    orcid: 0000-0002-6099-2772
    affiliation: 1,3
affiliations:
  - name: Department of Industrial Engineering and Management Sciences, Northwestern University, Evanston, Illinois 60208, United States of America
    index: 1
  - name: Mathematics and Computer Science Division, Argonne National Laboratory, Lemont, Illinois 60439, United States of America
    index: 2
  - name: Applied Mathematics and Computational Research Division, Lawrence Berkeley National Laboratory, Berkeley, California 94720, United States of America
    index: 3
  - name: Department of Information Systems & Analytics, Miami University, Oxford, Ohio 45056, United States of America
    index: 4
date: 12 July 2026
bibliography: paper.bib
---

# Summary

`surmise` is a Python package providing a modular interface for surrogate
models (emulators) that connect to Bayesian calibration, uncertainty
quantification, and sensitivity analysis tools. Its design allows users to
mix and match emulation and calibration strategies for a given scientific
problem: emulators and calibrators are implemented as interchangeable
modules behind a common API. The package includes several state-of-the-art Gaussian-process-based
emulators and is distributed on PyPI with prebuilt wheels.

# Statement of Need

Computer experiments in science and engineering often rely on expensive
simulation models; statistical emulators are required for calibration and
UQ at feasible cost.

# Acknowledgements


# References