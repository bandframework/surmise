import numpy as np

import surmise


def create_sampler(test_setup):
    # -- Outputs
    sampler_cfg = None

    sampler_name = test_setup["Sampler"]["Name"]
    if sampler_name.lower() == "metropolis_hastings":
        # -- Metropolis-Hastings Sampler
        # Extract sampler-specific configuration info
        sampler_cfg = test_setup["Sampler"]
        step_cfg = sampler_cfg["StepDistribution"]
        if "Scale" in step_cfg:
            step_scale = np.atleast_1d(np.squeeze(step_cfg["Scale"]))
            assert step_scale.ndim == 1
        else:
            step_scale = None

        sampler_cfg = {"stepType": step_cfg["Name"],
                       "stepParam": step_scale,
                       "burnSamples": sampler_cfg["n_burn_samples"],
                       "verbose": sampler_cfg["verbose"]}
        sampler = surmise.create_sampler(sampler_name, sampler_cfg)
    elif sampler_name.upper() == "LMC":
        # -- Langevin MC Sampler
        sampler_cfg = {"expertMode": test_setup["Sampler"]["expertMode"]}
        sampler = surmise.create_sampler(sampler_name, sampler_cfg)
        sampler_cfg = {}
    elif sampler_name.upper() == "PTLMC":
        # -- Parallel-Tempering Langevin MC Sampler
        sampler_cfg = {
            "numtemps": test_setup["Sampler"]["numtemps"],
            "numchain": test_setup["Sampler"]["numchain"],
            "sampperchain": test_setup["Sampler"]["sampperchain"],
            "maxtemp": test_setup["Sampler"]["maxtemp"]
        }
        sampler = surmise.create_sampler(sampler_name, sampler_cfg)
    else:
        raise ValueError(f"Unsupported sampler ({sampler_name})")

    return sampler_name, sampler, sampler_cfg
