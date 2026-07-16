import numpy as np

import surmise


def create_sampler(test_setup):
    # -- Outputs
    sampler_cfg = {"numsamp": test_setup["n_samples"]}

    theta_0 = None
    if "theta_0" in test_setup:
        theta_0 = np.atleast_2d(np.squeeze(test_setup["theta_0"]))
        assert theta_0.ndim == 2
        assert theta_0.shape[0] == 1
    sampler_cfg["theta0"] = theta_0

    sampler_name = test_setup["Sampler"]["Name"]
    if sampler_name.lower() == "metropolis_hastings":
        # -- Metropolis-Hastings Sampler
        # Extract sampler-specific configuration info
        cfg = test_setup["Sampler"]
        step_cfg = cfg["StepDistribution"]
        if "Scale" in step_cfg:
            step_scale = np.atleast_1d(np.squeeze(step_cfg["Scale"]))
            assert step_scale.ndim == 1
        else:
            step_scale = None

        sampler_cfg["stepType"] = step_cfg["Name"]
        sampler_cfg["stepParam"] = step_scale
        sampler_cfg["burnSamples"] = cfg["n_burn_samples"]
        sampler_cfg["verbose"] = cfg["verbose"]
        sampler = surmise.create_sampler(sampler_name, sampler_cfg)
    elif sampler_name.upper() == "LMC":
        # -- Langevin MC Sampler
        sampler_cfg["expertMode"] = test_setup["Sampler"]["expertMode"]
        sampler = surmise.create_sampler(sampler_name, sampler_cfg)
        del sampler_cfg["expertMode"]
    elif sampler_name.upper() == "PTLMC":
        # -- Parallel-Tempering Langevin MC Sampler
        sampler_cfg["numtemps"] = test_setup["Sampler"]["numtemps"]
        sampler_cfg["numchain"] = test_setup["Sampler"]["numchain"]
        sampler_cfg["sampperchain"] = test_setup["Sampler"]["sampperchain"]
        sampler_cfg["maxtemp"] = test_setup["Sampler"]["maxtemp"]
        sampler = surmise.create_sampler(sampler_name, sampler_cfg)
    else:
        raise ValueError(f"Unsupported sampler ({sampler_name})")

    return sampler_name, sampler, sampler_cfg
