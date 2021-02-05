
# SURmise Test Runner

# Test options
export TEST_LIST=test_*.py    #Provide a list of tests #unordered # override with -y
#export TEST_LIST='test_compute.py'#'test_cal_directbayes.py test_cal_MLcal.py' #selected
export RUN_COV_TESTS=true     #Provide a coverage report


# Options for test types (only matters if "true" or anything else)
export RUN_EMU_TESTS=true
export RUN_CAL_TESTS=true
export RUN_UTI_TESTS=true

usage() {
  echo -e "\nUsage:"
  echo "  $0 [-hec] [-a <string>]" 1>&2;
  echo ""
  echo "Options:"
  echo "  -h              Show this help message and exit"
  echo "  -e              Run only the emulator tests"
  echo "  -c              Run only the calibration tests"
  echo "  -u              Run only the utilities tests"
  echo "  -a {args}       Supply inputs to emulator tests"
  echo "  -b {args}       Supply inputs to calibrator tests"
  echo "  -l {args}       Supply a list of tests as a reg. expression  e.g. -l 'test_cal_MLcal.py'"
  echo ""
  echo "Note: If none of [-ecu] are given, the default is to run tests for all"
  echo ""
  exit 1
}

while getopts ":a:b:l:ecuth" opt; do
  case $opt in
    a)
      echo "Parameter supplied for emulator test args: $OPTARG" >&2
      export TEST_INPUT="$OPTARG"
      export RUN_CAL_TESTS=false
      export TEST_LIST=test_new*.py
      ;;
    b)
      echo "Parameter supplied for calibrator test args: $OPTARG" >&2
      export TEST_INPUT="$OPTARG"
      export RUN_EMU_TESTS=false
      export TEST_LIST=test_new*.py
      ;;
    l)
      echo "Running with user supplied test list"
      export TEST_LIST="$OPTARG"
      ;;
    e)
      echo "Running only the emulator tests"
      export RUN_CAL_TESTS=false
      export RUN_UTI_TESTS=false
      export TEST_LIST=test_emu*.py
      ;;
    c)
      echo "Running only the calibrator tests"
      export RUN_EMU_TESTS=false
      export RUN_UTI_TESTS=false
      export TEST_LIST=test_cal*.py
      ;;
    u)
      echo "Running only the utilities tests"
      export RUN_EMU_TESTS=false
      export RUN_CAL_TESTS=false
      ;;
    h)
      usage
      ;;
    \?)
      echo "Invalid option supplied: -$OPTARG" >&2
      usage
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Try using git root dir
root_found=false
ROOT_DIR=$(git rev-parse --show-toplevel) && root_found=true
#If not found using git - try search up tree for setup.py
if [[ $root_found == "false" ]]; then
    search_dir=`pwd`
    search_for="setup.py"
    while [ "$search_dir" != "/" ]
    do
        file_found=$(find_file_in_dir $search_dir $search_for)
        if [[ $file_found = "true" ]]; then
            ROOT_DIR=$search_dir
            root_found=true
            break;
        fi;
        search_dir=`dirname "$search_dir"`
    done
fi;

# Test Directories - all relative to project root dir
export TESTING_DIR=$ROOT_DIR/tests
export TEST_SUBDIR_EMU_CAL=$ROOT_DIR/tests/test_emu_cal
export TEST_SUBDIR_EMU=$ROOT_DIR/tests/test_emulator
export TEST_SUBDIR_CAL=$ROOT_DIR/tests/test_calibrator
export TEST_SUBDIR_NEW_EMU=$ROOT_DIR/tests/test_new_emulator
export TEST_SUBDIR_NEW_CAL=$ROOT_DIR/tests/test_new_calibrator

tput bold
tput sgr 0
echo -e "Selected:"
[ $RUN_EMU_TESTS = "true" ] && echo -e "Emulator tests"
[ $RUN_CAL_TESTS = "true" ]  && echo -e "Calibrator tests"
[ $RUN_COV_TESTS = "true" ]  && echo -e "Including coverage analysis"

COV_LINE_SERIAL=''
if [ $RUN_COV_TESTS = "true" ]; then
   COV_LINE_SERIAL='--cov --cov-report html:cov_html'
fi;

# Run Tests -----------------------------------------------------------------------
if [ "$RUN_EMU_TESTS" = true ] && [ "$RUN_CAL_TESTS" = true ]; then
  echo -e "\n************** Running: surmise Test-Suite **************\n"
  pytest $COV_LINE_SERIAL -k 'not new'
else
  if [ "$RUN_EMU_TESTS" = true ]; then
    if [ -z "$TEST_INPUT" ]; then
      echo -e "\n************** Running: surmise.emulation Test-Suite **************\n"
      for DIR in $TEST_SUBDIR_EMU_CAL
      do
        cd $DIR
        for TEST_SCRIPT in $TEST_LIST
        do
          pytest $TEST_SCRIPT $COV_LINE_SERIAL
        done
      done
    else
      echo -e "\n************** Running: New surmise.emulation Test-Suite **************\n"
      for DIR in $TEST_SUBDIR_NEW_EMU
      do
        cd $DIR
        for TEST_SCRIPT in $TEST_LIST
        do
          pytest $TEST_SCRIPT $COV_LINE_SERIAL --cmdopt1=$TEST_INPUT
        done
      done
    fi;
  fi;
  if [ "$RUN_CAL_TESTS" = true ]; then
    if [ -z "$TEST_INPUT" ]; then
      echo -e "\n************** Running: surmise.calibration Test-Suite **************\n"
      for DIR in $TEST_SUBDIR_EMU_CAL
      do
        cd $DIR
        for TEST_SCRIPT in $TEST_LIST
        do
          pytest $TEST_SCRIPT $COV_LINE_SERIAL
        done
      done
    else
      echo -e "\n************** Running: New surmise.calibration Test-Suite **************\n"
      for DIR in $TEST_SUBDIR_NEW_CAL
      do
        cd $DIR
        for TEST_SCRIPT in $TEST_LIST
        do
          pytest $TEST_SCRIPT $COV_LINE_SERIAL --cmdopt2=$TEST_INPUT
        done
      done
    fi;
  fi;
fi;
