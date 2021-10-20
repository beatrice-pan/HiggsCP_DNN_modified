#!/bin/bash -l
## Job name
#SBATCH -J CPc012regr
## Number of nodes
#SBATCH -N 1
## Number of tasks per node
#SBATCH --ntasks-per-node=1
## Memory per CPU (default: 5GB)
#SBATCH --mem-per-cpu=20GB
## Max job time (HH:MM:SS format)
#SBATCH --time=16:00:00
## Pratition specification
#SBATCH -p plgrid
#SBATCH --array=0-1

## Setup
if [ -f setup ]; then source setup; fi
export PYTHONPATH=$ANACONDA_PYTHON_PATH
module add plgrid/apps/cuda/7.5

## Chaninging dir to workdir
cd $WORKDIR

METHODS=(hits_c1s hits_c2s)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}

## Command
echo "CPMix CPc012regr. Method: " $METHOD
$ANACONDA_PYTHON_PATH/python2.7 $WORKDIR/main.py -e 25 --training_method regr_c012s --hits_c012s $METHOD
