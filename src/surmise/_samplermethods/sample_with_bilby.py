import copy
import bilby

from .BilbyWrappers import (
    BilbyJointPriorDist, BilbyJointPrior,
    BilbyLikelihood
)


def sample_with_bilby(log_joint_prior, log_likelihood, draw_func,
                      sampler, options=None):
    """
    .. todo::
        * Is there a better way to get the parameter order
        * It would be nice if we could pass the actual calibration method name
          to the likelihood for logging by bilby
        * Include more bilby outputs, including method-specific results, in our
          returned results?
    """
    kwargs = copy.deepcopy(options)

    if "parameter_order" not in kwargs:
        raise ValueError("Please provide parameter_order sampler option")
    parameter_order = kwargs["parameter_order"]
    del kwargs["parameter_order"]

    bilby_joint_dist = BilbyJointPriorDist(
        parameter_order, log_joint_prior.lpdf, log_joint_prior.rnd)
    bilby_prior = {key: BilbyJointPrior(bilby_joint_dist, key)
                    for key in parameter_order}

    bilby_likelihood = BilbyLikelihood(parameter_order, log_likelihood)

    result = bilby.run_sampler(
        likelihood=bilby_likelihood,
        priors=bilby_prior,
        sampler=sampler,
        **kwargs
    )

    return {"theta": result.samples}
