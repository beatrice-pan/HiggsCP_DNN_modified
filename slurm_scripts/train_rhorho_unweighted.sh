#!/bin/bash -l
## Job name
#SBATCH -J TrainUnweightedRhoRho
## Number of nodes
#SBATCH -N 1
## Number of tasks per node
#SBATCH --ntasks-per-node=4
## Memory per CPU (default: 5GB)
#SBATCH --mem-per-cpu=5GB
## Max job time (HH:MM:SS format)
#SBATCH --time=16:00:00
## Pratition specification
#SBATCH -p plgrid
#SBATCH --array=0-4

## Setup
if [ -f setup ]; then source setup; fi
export PYTHONPATH=$ANACONDA_PYTHON_PATH
module add plgrid/apps/cuda/7.5

## Chaninging dir to workdir
cd $WORKDIR

## Command
FEATURE_SETS=(Variant-All Variant-1.0 Variant-1.1 Variant-2.0 Variant-2.1 Variant-2.2 Variant-3.0)
FEATURE_SET=${FEATURE_SETS[$SLURM_ARRAY_TASK_ID]}
echo "TrainUnweightedRhoRho Job. Method: " $FEATURE_SET
$ANACONDA_PYTHON_PATH/python2.7 $WORKDIR/main.py -t nn_rhorho -i $RHORHO_DATA -e 250 -f $FEATURE_SET -d 0.0 --unweighted True
