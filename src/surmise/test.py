import pytest


def test(level=0):
    """
    Run full set of surmise tests.

    Parameters
    ----------
    level : int
        Smaller values indicate less logging

    Returns
    -------
    bool
        True if all tests passed; False, otherwise.
    """
    VERBOSITY = [0, 1, 2]

    if level not in VERBOSITY:
        raise ValueError(f"level must be in {VERBOSITY}")

    cmd = [f"--verbosity={level}", "--pyargs", "surmise.tests"]

    return (pytest.main(cmd) == 0)
