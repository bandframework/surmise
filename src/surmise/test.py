import pytest


def test(level=0):
    """
    Run full set of surmise tests.

    :param level: Smaller values indicate less logging
    """
    VERBOSITY = [0, 1, 2]

    args = []
    if level not in VERBOSITY:
        raise ValueError(f"level must be in {VERBOSITY}")
    args = [f"--verbosity={level}"]

    return (pytest.main(args + ["--pyargs", "surmise.tests"]) == 0)
