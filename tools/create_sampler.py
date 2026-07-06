import numpy as np

from surmise.utilitiesmethods.metropolis_hastings import sampler as MH_sampler
from surmise.utilitiesmethods.LMC import sampler as LMC_sampler
from surmise.utilitiesmethods.PTLMC import sampler as PTLMC_sampler


def create_sampler(test_setup):
    """
    .. todo::
        * Allow for more than one sampler
    """
    # -- Outputs
    sampler = None
    sampler_cfg = None

    sampler_name = test_setup["Sampler"]["Name"]
    if sampler_name.upper() == "MH":
        # -- Metropolis-Hastings Sampler
        # Extract sampler-specific configuration info
        step_cfg = test_setup["Sampler"]["StepDistribution"]
        step_type = step_cfg["Name"]
        if "Scale" in step_cfg:
            step_scale = np.atleast_1d(np.squeeze(step_cfg["Scale"]))
            assert step_scale.ndim == 1
        else:
            step_scale = None

        sampler = MH_sampler
        sampler_cfg = {"stepType": step_type, "stepParam": step_scale}
    elif sampler_name.upper() == "LMC":
        # -- Langevin MC Sampler
        sampler = LMC_sampler
        sampler_cfg = {}
    elif sampler_name.upper() == "PTLMC":
        # -- Parallel-Tempering Langevin MC Sampler
        sampler = PTLMC_sampler
        sampler_cfg = {
            "numtemps": test_setup["Sampler"]["numtemps"],
            "numchain": test_setup["Sampler"]["numchain"],
            "sampperchain": test_setup["Sampler"]["sampperchain"],
            "maxtemp": test_setup["Sampler"]["maxtemp"]
        }
    else:
        raise ValueError(f"Unsupported sampler ({sampler_name})")

    return sampler_name, sampler, sampler_cfg
