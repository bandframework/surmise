import h5py

from pathlib import Path


def save_mcmc_results(filename, results, overwrite=False):
    GROUP = "/MCMC"

    if Path(filename).resolve().exists() and (not overwrite):
        raise ValueError(f"{filename} already exists")

    with h5py.File(filename, "w") as fptr:
        fptr["/"].create_group("MCMC")
        # Common configuration
        # TODO: Add these

        # Method-specific configuration
        fptr[GROUP].attrs["Method"] = "Metropolis"
        fptr[GROUP].attrs["AcceptanceRate"] = results["acc_rate"]
        # TODO: Add others

        fptr[GROUP].create_dataset("OfficialSamples", data=results["theta"])
