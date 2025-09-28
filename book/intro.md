# Introduction

This collection of notebooks aims to provide the basic yet practical 
usage of {{surmise}} in constructing statistical emulators and conducting 
Bayesian calibration.  In a nutshell, Bayesian calibration refers to the task
of learning unknown parameters of a simulation model through estimating the 
posterior distribution of the parameters.  When a simulation model is computationally
expensive, emulators serve as fast approximations that are constructed based on 
simulation outputs. 

Readers may use these notebooks for several purposes.
First and foremost, the notebooks are examples for learning the functionalities 
offered in {{surmise}}.  Second, the notebooks are case studies of general topics 
in Bayesian calibration, for which each notebook introduces the context and topics
it covers.

For an in-depth introduction of Bayesian calibration, readers may refer to 
{cite:t}`kennedy2001bayesian`.  For exploring the basics of Gaussian process 
emulators, which form the foundation of many available emulators in {{surmise}}, 
readers are directed to {cite:t}`santner2003design, williams2006gaussian, gramacy2020surrogates`.


## Documentation
The full documentation including the programmatic interface of the package is
available in surmise's [user and developer guides](https://surmise.readthedocs.io).

## Interactivity
The notebooks included in this book can be launched {{via}}
[Binder](https://jupyter.org/binder) so that users can run them interactively.
Please note that the notebooks often contain cells that are not present in the
book's rendering.  These cells can contain

* commented out lines that specify other emulator, calibrator, or sampler
  methods that could be used in the notebook or
* code that checks correct execution of the notebook.

The latter can be ignored since they are for development and maintenance
purposes only.

## Table of contents

```{tableofcontents}
```
