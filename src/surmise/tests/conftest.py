# We presently have all tests located in this single tests folder in a flat
# layout.  It appears that as a result we can have only this single conftest.py
# file with this single addoption function.
#
# Therefore, we must add in the union of all flags needed across all tests and
# all tests have the potential to check all flags.  However, each of the flags
# is restricted to a single type of tests, which should be clear from the flag
# name.  Therefore, it is the responsibility of test developers/maintainers to
# only check the flags that make sense for their tests.
def pytest_addoption(parser):
    # For test_new_emu_* tests
    parser.addoption("--emu", action="store", help="Name of an emulator")
    # For test_new_cal_* tests
    parser.addoption("--cal", action="store", help="Name of an calibrator")
