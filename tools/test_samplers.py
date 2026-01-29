#!/usr/bin/env python

import sys
import argparse
import traceback

from pathlib import Path


import unittest

from TestSampler import TestSampler


def main():
    # ----- HARDCODED VALUES
    # We should be able to run this correctly in CI test servers
    SUCCESS = 0
    FAILURE = 1

    def log_and_abort(msg):
        print(f"\n{msg}\n")
        print(traceback.format_exc())
        sys.exit(FAILURE)

    # ----- SPECIFY COMMAND LINE USAGE
    DESCRIPTION = (
        "Run all Metropolis-Hastings tests in the given test suite"
    )
    JSON_HELP = (
        "JSON-format file that encodes the desired test suite to run"
    )
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("json", nargs=1, type=str, help=JSON_HELP)

    # ----- GET COMMAND LINE ARGUMENTS
    args = parser.parse_args()
    fname_json = Path(args.json[0]).resolve()
    if not fname_json.is_file():
        log_and_abort(f"{fname_json} does not exist or is not a file")

    # Run only a single test in the TestSampler test case.  That case should be
    # developed and maintained such that this is the only test routine that it
    # contains.
    test_case = TestSampler("testAllSetups", test_spec=fname_json)

    suite = unittest.TestSuite()
    suite.addTest(test_case)

    result = unittest.TextTestRunner().run(suite)
    return SUCCESS if result.wasSuccessful() else FAILURE


if __name__ == "__main__":
    sys.exit(main())
