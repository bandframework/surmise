import numpy as np

from surmise.utilitiesmethods.metropolis_hastings import sampler as MH_sampler


def create_sampler(test_setup):
    """
    .. todo::
        * Allow for more than one sampler
    """
    # -- Outputs
    sampler = None
    sampler_cfg = None

    # -- Metropolis-Hastings Sampler
    # Extract sampler-specific configuration info
    step_cfg = test_setup["StepDistribution"]
    step_type = step_cfg["Name"]
    if "Scale" in step_cfg:
        step_scale = np.atleast_1d(np.squeeze(step_cfg["Scale"]))
    else:
        step_scale = None

    sampler = MH_sampler
    sampler_cfg = {"stepType": step_type, "stepParam": step_scale}

    return sampler, sampler_cfg
