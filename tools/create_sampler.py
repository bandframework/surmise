import numpy as np

import surmise


def create_sampler(test_setup):
    # -- Outputs
    sampler_cfg = {"nSamples": test_setup["nSamples"],
                   "verbose": test_setup["verbose"]}

    theta_0 = None
    if "theta_0" in test_setup:
        theta_0 = np.atleast_2d(np.squeeze(test_setup["theta_0"]))
        assert theta_0.ndim == 2
        assert theta_0.shape[0] == 1
    sampler_cfg["theta0"] = theta_0

    expert_mode = False
    sampler_name = test_setup["Name"]
    if sampler_name.lower() == "metropolis_hastings":
        # -- Metropolis-Hastings Sampler
        # Extract sampler-specific configuration info
        step_cfg = test_setup["StepDistribution"]
        if "Scale" in step_cfg:
            step_scale = np.atleast_1d(np.squeeze(step_cfg["Scale"]))
            assert step_scale.ndim == 1
        else:
            step_scale = None

        sampler_cfg["stepType"] = step_cfg["Name"]
        sampler_cfg["stepParam"] = step_scale
        sampler_cfg["nBurnSamples"] = test_setup["nBurnSamples"]
    elif sampler_name.upper() == "LMC":
        # -- Langevin MC Sampler
        expert_mode = test_setup["expertMode"]
    elif sampler_name.upper() == "PTLMC":
        # -- Parallel-Tempering Langevin MC Sampler
        sampler_cfg["nTemperatures"] = test_setup["nTemperatures"]
        sampler_cfg["nChains"] = test_setup["nChains"]
        sampler_cfg["samplesPerChain"] = test_setup["samplesPerChain"]
        sampler_cfg["maxTemperature"] = test_setup["maxTemperature"]
    else:
        raise ValueError(f"Unsupported sampler ({sampler_name})")

    sampler = surmise.create_sampler(sampler_name, expert_mode=expert_mode)

    return sampler_name, sampler, sampler_cfg
