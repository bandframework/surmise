Diagnosing covariance problems
++++++++++++++++++++++++++++++++

The Gaussian likelihoods in ``directbayes`` and ``directbayeswoodbury`` involve factorizing a
covariance matrix. When the emulator returns a
predictive covariance that is not positive definite, or contains ``nan`` or
``inf``, the log-likelihood becomes ``nan`` and causes downstream issue. This can
happen at parameters where the emulator variance is close to zero and it can depend on
the platform and linear algebra library.

The results of
the checks for such ``nan`` likelihoods are stored in ``fitinfo['cov_diagnosis']``, available after fitting the calibrator as
``cal.info['cov_diagnosis']``. After sampling, ``fit`` will print a
``RuntimeWarning`` that reports any counts of non-singular matrix inversion, or if any covariance inversion is impacted by machine precision (see ``n_below_bound`` below).

.. code-block:: python

    cal = calibrator(emu=emu, y=y, x=x, thetaprior=prior,
                     method='directbayeswoodbury', yvar=obsvar,
                     args={'sampler': 'metropolis_hastings', ...})

    cov_diagnosis = cal.info['cov_diagnosis']
    rejected = cov_diagnosis['n_nonpd'] + cov_diagnosis['n_nonfinite']
    print(f"{rejected} of {cov_diagnosis['n_eval']} evaluations rejected")
    print(cov_diagnosis['theta'][:5])   # parameters that failed
    print(cov_diagnosis['eig'][:5])     # their three smallest eigenvalues

The record contains the following keys.

=================  ================================================================
Key                Meaning
=================  ================================================================
``n_eval``         Number of likelihood evaluations checked.
``n_nonfinite``    Rejected because the mean, covariance factor, or eigenvalues
                   contain ``nan`` or ``inf``.
``n_nonpd``        Rejected because the smallest eigenvalue is at or below
                   round-off tolerance.
``n_below_bound``  Kept, but the smallest eigenvalue fell below its exact lower
                   bound by more than round-off. This signals lost accuracy.
``min_eig``        Smallest eigenvalue seen across all finite evaluations.
``theta``          Up to the first ``max_store`` parameters that were rejected.
``eig``            The three smallest eigenvalues for each stored parameter.
``max_store``      Maximum number of stored parameters (default 50).
=================  ================================================================

A rejected parameter is treated as having zero posterior density. A few
rejections in low posterior probability regions are not uncommon.  However, if there are many rejections, or rejections near
the posterior mode, the emulator should be
revisited (for example a larger nugget lower bound or fewer, better spread
training points). To stop on the first problem instead of continuing, promote
the warning to an error:

.. code-block:: python

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        cal = calibrator(...)

The record is reset at the start of every ``fit``. Evaluations made
afterwards, for example through ``cal.theta.lpdf``, are added to the same record.
