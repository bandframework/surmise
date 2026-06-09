from IndependentJointDistribution import IndependentJointDistribution
from UniformDistribution import UniformDistribution


class JointUniformDistribution(IndependentJointDistribution):
    def __init__(self, *args):
        univariate_distributions = []
        for i, ival_i in enumerate(args):
            assert len(ival_i) == 2
            a_i, b_i = ival_i
            univariate_distributions.append(UniformDistribution(a_i, b_i, i+1))

        super().__init__(univariate_distributions)
