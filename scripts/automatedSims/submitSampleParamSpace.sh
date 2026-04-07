#!/bin/bash
#SBATCH --job-name=paramSweep
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --partition=standard

# Usage:
#   sbatch submitSampleParamSpace.sh <num_params>

set -e  #exit on first error

#----------- CONFIGURATION -----------
NUM_PARAMETER_SETS=${1:-1}  #number of parameter sets (override with commandline argument)
SCRIPT_DIR=$(pwd)   #path to the Python script
PYTHON_SCRIPT="sampleParamSpace.py"

#----------- EXECUTION -----------
cd "$SCRIPT_DIR" || exit 1  #navigate to submission directory

#Check that the script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: $PYTHON_SCRIPT not found in $SCRIPT_DIR"
    exit 1
fi

#Run the Python script with automatic input
python3 "$PYTHON_SCRIPT" << EOF
$NUM_PARAMETER_SETS
EOF

#Capture exit code
EXIT_CODE=$?

#if [ $EXIT_CODE -eq 0 ]; then
#    echo "Script completed successfully"
#else
#    echo "Script failed with exit code: $EXIT_CODE"
#fi
#echo "End time: $(date)"

exit $EXIT_CODE
