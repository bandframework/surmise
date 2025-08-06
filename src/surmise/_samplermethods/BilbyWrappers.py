import bilby

import numpy as np


class BilbyJointPriorDist(bilby.core.prior.BaseJointPriorDist):
    def __init__(self, names, log_joint_prior, draw_samples):
        """
        :param name: List of parameters included in the 
        """
        super().__init__(names=names)

        self.log_joint_prior = log_joint_prior
        self.draw_samples = draw_samples

    def __repr__(self):
        return "User-provided joint prior distribution"
    
    def _ln_prob(self, theta, lnprob, outbounds):
        # I believe that the ordering of the given theta matches the ordering of
        # the names argument given at instantiation.
        #
        # Therefore, calling code should match the actual names argument to the
        # ordering expected by the given actual log_joint_prior argument given
        # at instantiation.
        assert theta.ndim == 2
        assert theta.shape[1] == len(self)
        n_theta = theta.shape[0]

        lpdf = self.log_joint_prior(theta)
        if lpdf.ndim != 2:
            raise ValueError("User-provided log prior is not 2D")
        elif lpdf.shape != (n_theta, 1):
            raise ValueError("User-provided log prior is wrong size")

        if n_theta == 1:
            return lpdf[0]

        return lpdf

    def _sample(self, size, **kwargs):
        # Bilby maps the parameter names provided at instantiation onto the
        # columns.  Therefore, the user must ensure that the draw_samples
        # routine they provided respects the ordering that they provided.
        samples = self.draw_samples(size)
        if samples.shape != (size, len(self)):
            raise ValueError("User-provided samples are wrong size")
        return samples

    def _rescale(self, samp, **kwargs):
        raise NotImplementedError("No rescaling implemented yet")


class BilbyJointPrior(bilby.core.prior.JointPrior):
    def __init__(self, dist, name, latex_label=None, unit=None):
        if not isinstance(dist, BilbyJointPriorDist):
            raise ValueError("Invalid bilby Joint Distribution")

        super().__init__(
            dist=dist, name=name, latex_label=latex_label, unit=unit
        )


class BilbyLikelihood(bilby.Likelihood):
    def __init__(self, parameter_order, log_likelihood):
        self.__parameter_order = parameter_order
        self.__log_likelihood = log_likelihood

        self.parameters = dict.fromkeys(self.__parameter_order)

        super().__init__(parameters=self.parameters)

    def log_likelihood(self):
        theta = np.array([[self.parameters[k] for k in self.__parameter_order]])
        return self.__log_likelihood(theta)
