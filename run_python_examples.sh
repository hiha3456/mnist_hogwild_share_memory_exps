#!/usr/bin/env bash
#
# This script runs through the code in each of the python examples.
# The purpose is just as an integrtion test, not to actually train
# models in any meaningful way. For that reason, most of these set
# epochs = 1 and --dry-run.
#
# Optionally specify a comma separated list of examples to run.
# can be run as:
# ./run_python_examples.sh "install_deps,run_all"
# to pip install dependencies (other than pytorch), run all examples,
# and remove temporary/changed data files.
# Expects pytorch, torchvision to be installed.

BASE_DIR=`pwd`"/"`dirname $0`
EXAMPLES=`echo $1 | sed -e 's/ //g'`

USE_CUDA=$(python -c "import torchvision, torch; print(torch.cuda.is_available())")
case $USE_CUDA in
  "True")
    echo "using cuda"
    CUDA=1
    CUDA_FLAG="--cuda"
    ;;
  "False")
    echo "not using cuda"
    CUDA=0
    CUDA_FLAG=""
    ;;
  "")
    exit 1;
    ;;
esac

ERRORS=""

function error() {
  ERR=$1
  ERRORS="$ERRORS\n$ERR"
  echo $ERR
}

function install_deps() {
  echo "installing requirements"
  cat $BASE_DIR/*/requirements.txt | \
    sort -u | \
    # testing the installed version of torch, so don't pip install it.
    grep -vE '^torch$' | \
    pip install -r /dev/stdin || \
    { error "failed to install dependencies"; exit 1; }
}

function start() {
  EXAMPLE=${FUNCNAME[1]}
  cd $BASE_DIR/$EXAMPLE
  echo "Running example: $EXAMPLE"
}

function mnist_hogwild() {
  start
  python main.py --epochs 1 --dry-run $CUDA_FLAG || error "mnist hogwild failed"
}


function run_all() {
  mnist_hogwild
}

# by default, run all examples
if [ "" == "$EXAMPLES" ]; then
  run_all
else
  for i in $(echo $EXAMPLES | sed "s/,/ /g")
  do
    echo "Starting $i"
    $i
    echo "Finished $i, status $?"
  done
fi

if [ "" == "$ERRORS" ]; then
  echo "Completed successfully with status $?"
else
  echo "Some examples failed:"
  printf "$ERRORS"
  #Exit with error (0-255) in case of failure in one of the tests.
  exit 1

fi
