Random Number Generation
========================
.. _scientific Python RNG specification: https://scientific-python.org/specs/spec-0007/

Following typical practices, we refer to pseudorandom number generation and
generators more generically as random number generation and random number
generators (RNGs).

Please familiarize yourself with the RNG content in :ref:`rng_user_guide` before
reading this section.  Similarly, reviewing the historic record of |surmise| RNG
requirements might be helpful to motivate the design and explain certain design
decisions detailed here.

Our design and implementation should be informed by and, where possible, follow the
advice and decisions reported in the `scientific Python RNG specification`_.

Design
------
.. _`generator documentation`: https://numpy.org/doc/stable/reference/random/index.html#quick-start
.. _`statistically independent RNGs`: https://numpy.org/doc/stable/reference/random/parallel.html#parallel-random-number-generation

Design discussions originally considered two possible schemes for providing
|surmise| code access to user-provided RNGs:

#. **Manual propagation scheme** - all |surmise| code elements that use RNGs
   accept and require an RNG argument, which they use directly.
#. **Single "global" RNG scheme** - users set into |surmise| a single RNG object
   that all |surmise| code elements that use RNGs access directly for direct use.

..
    #### Manual propagation scheme
    All surmise code elements that draw samples shall have either an `RNG`
    argument or accept keyword arguments (i.e., `kwargs`).  If the code element
    uses `kwargs`, then the element shall insist that an `RNG` `(key, value)`
    pair be provided.  Valid `RNG` actual arguments are a single `scipy.stats`
    RNG object or `"default"`.  For the former case, the code element shall use
    the provided RNG directly and exclusively; the latter case, the code element
    shall create and use a single default `scipy.stats` RNG.

    One difficulty with allowing for `RNG="default"` is that `scipy.stats`'s
    definition of default could change over time.  If surmise is not actively
    updated to match those changes (and potentially create default RNGs based on
    the version of `scipy` in use), users might be aware of the newest
    definition and incorrectly assume that surmise is using that definition.

Since it was decided that the |surmise| design and use cases are consistent with
users providing a single RNG for use by all |surmise| code, we adopted the
latter access scheme.

|surmise| is effectively an application, and, therefore, we do not believe that
this decision is contrary to the scientific Python RNG specification, which is
geared toward libraries.

High-level
^^^^^^^^^^
To adhere to the |surmise| RNG requirements related to easing implementation and
maintenance by using only one, large statistics package with RNG capabilities,
this design stipulates that all sampling of random numbers in |surmise| occur
using either the

* ``scipy.stats`` package (preferably using improved interface introduced at
  v1.15.0) or
* ``scipy.stats`` RNG currently in use (|eg| using the RNG's ``choice`` method).

In particular, no other packages, such as ``numpy.random``, should be used in
|surmise| even if the current RNG is compatible with that package.  This
decision is also motivated by the fact that

* ``scipy.stats`` is considered to be sufficient for correct statistics-based
  modeling and simulation (but not more demanding than that),
* the ``scipy.stats`` RNG can be used with cryptographically-strong seeds
  generated with ``secrets.randbits`` (as suggested by the `generator
  documentation`_), and
* the ``scipy.stats`` RNG supports the creation of sets of
  `statistically independent RNGs`_ (|eg| ``spawn()``).

Note that, for simplicity's sake, code that accepts an RNG will be constrained to
only accept ``scipy.stats`` RNG objects rather than, as suggested by the scientific
Python RNG specification, any object that is or could be used to construct such
an RNG object.

..
    We will use only `scipy.stats` throughout surmise instead of, for example,
    `numpy.random` since the former not only provides tools for drawing from a
    suite of many typical distributions but also additional software tools for
    working with those distributions.  While there is no analogue to
    `numpy.random.choice` in `scipy.stats`, we can avoid using `numpy.random`
    since `numpy` now dictates that users should call `choice` directly from
    their `numpy.random.Generator` object, which is the RNG to be used with
    `scipy.stats`, rather than use `numpy.random.choice`.  Therefore, we will
    only use `scipy.stats` and its official `Generator` objects to draw random
    numbers.

    The consistent use of just `scipy.stats` is indeed important since the
    interfaces of both `scipy.stats` and `numpy.random` have been changing
    significantly recently.  Presently, it appears that we could allow for
    simultaneous use of both packages since both use
    [`numpy.random` RNGs](https://docs.scipy.org/doc/scipy/tutorial/stats/probability_distributions.html#random-number-generation).
    However, restricting use to just `scipy.stats` does indeed reduce the
    complexity of understanding and managing the use of two related by different
    packages.  In addition, it protects surmise from any possible decoupling of
    the two packages that might result in the two packages using different RNGs.

    While wrapping the `scipy.stats` RNGs in a simple surmise interface might
    aid in maintainability as the `scipy.stats` package evolves and improves, it
    might give the users the false impression that surmise takes some
    responsibility for correct creation and use of the RNGs.  In addition,
    including in the wrapper capabilities such as correct creation of a set of
    independent RNGs adds more development and maintenance costs.  Forcing users
    to explicitly manage RNGs at the level of `scipy.stats` is therefore
    preferred and deemed acceptable since these RNGs are well-documented and
    their use does not require excessive amounts of programming.

RNG Access Pattern
^^^^^^^^^^^^^^^^^^
.. _`Singleton pattern`: https://en.wikipedia.org/wiki/Singleton_pattern

We enforce the restriction of having at most one single RNG in existence at any
time by designing the dedicated, internal :py:class:`_RandomNumberGenerator`
class using the `Singleton pattern`_.  In accordance with requirements,
|surmise| code must

* never set, change, or alter the single RNG,
* access the single RNG through the Singleton interface and use it exclusively
  for random number generation,
* be designed and implemented where possible so that results are identical when
  rerun with an identical random number generation scenario, and
* be designed so that all related random number generation (|eg| performing a
  single calibration and generating a random reordering of ts MCMC samples) are
  performed such that calling code cannot alter the single RNG during the middle
  of that computational process.

In this design, only external, user code should set the current, "global" RNG
and no |surmise| code should store the current RNG for later use.  Instead, upon
each invocation |surmise| code should always acquire the current "global" RNG
from the package.

A typical use of the RNG in a |surmise| routine might be

.. code:: python

    from ._RandomNumberGenerator import RandomNumberGenerator

    def my_surmise_code(...):
        global_rng = RandomNumberGenerator().scipy_start_RNG
        scipy.stats.normal.rvs(..., random_state=global_rng)

To keep this class in the private package interface, the public interface
includes the :py:func:`set_RNG` function, which accepts from the user the RNG
object to be used and sets it into the Singleton object.  See
:ref:`rng_user_guide` for more information regarding the RNG public interface.

We recognize that in |surmise| the emulators and calibrators play a prominent,
application-specific role and are, therefore, in the public interface.  However,
the MCMC samplers are general statistical tools that are hidden behind the
calibrators.  To decouple the samplers from |surmise| design decisions and
enable their implementations to be more generic and widely useful, we exempt
them from having to access the global RNG |via| the |surmise| RNG design, and
instead require that the calibrators

* access the global |surmise| RNG as specified,
* determine the correct RNG usage for the larger calibration process including
  sampling, and
* pass to its sampler the necessary ``scipy.stats``-compatible RNG for its
  exclusive use directly and with ``scipy.stats`` for random number generation
  during its current invocation and only for that invocation.

Note that this design decision does effectively treat samplers as libraries so
that this part of the design does follow the suggestions of the scientific
Python RNG specification.  However, rather than name the RNG arguments ``rng``
as suggested, we prefer to name them ``scipy_stats_rng`` so that the code
explicitly reflects our design decision to use only one package.

External Packages
^^^^^^^^^^^^^^^^^
We plan to extend |surmise| so that external packages, including user-provided
code, can be used as part of executing its work.  For instance, |surmise|
calibrators will hopefully use both |surmise| internal and external |bilby|
samplers.  Since |surmise| cannot impose RNG use rules on external code,
inclusion of external code must only be made official if the use of RNGs in the
external code are compatible with |surmise| and allow for users to perform
statistically correct studies.  Users are responsible for determining if RNG use
in their user-provided code is valid.
