import h5py

import numpy as np

from pathlib import Path


def load_mcmc_results(filename):
    GROUP = "/MCMC"

    fname = Path(filename).resolve()
    with h5py.File(fname, "r") as fptr:
        method = fptr[GROUP].attrs["Method"]
        assert method == "Metropolis"
        acceptance_rate = fptr[GROUP].attrs["AcceptanceRate"]

        table_name = Path(GROUP).joinpath("OfficialSamples")
        samples = np.array(fptr[str(table_name.as_posix())])

    return {"theta": samples,
            "acc_rate": acceptance_rate}
