# TODO: A bit ugly to make the two flags globally accessible?  If it is, are we
# forced to add in a subfolder hierarchy of tests?
def pytest_addoption(parser):
    # For test_new_emu_* tests
    parser.addoption("--emu", action="store", help="Name of an emulator")
    # For test_new_cal_* tests
    parser.addoption("--cal", action="store", help="Name of an calibrator")
