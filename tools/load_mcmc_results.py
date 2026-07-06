import h5py

import numpy as np

from pathlib import Path


def load_mcmc_results(filename):
    GROUP = "/MCMC"

    results = {}

    fname = Path(filename).resolve()
    with h5py.File(fname, "r") as fptr:
        table_name = Path(GROUP).joinpath("OfficialSamples")
        results["theta"] = np.array(fptr[str(table_name.as_posix())])

        method = fptr[GROUP].attrs["Method"]
        if method == "Metropolis":
            results["acc_rate"] = fptr[GROUP].attrs["AcceptanceRate"]

    return results
