#!/bin/bash -l
## Job name
#SBATCH -J TrainFeaturesA1A1Dropout
## Number of nodes
#SBATCH -N 1
## Number of tasks per node
#SBATCH --ntasks-per-node=1
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
echo "TrainFeaturesA1A1 Job. Dropout 0.2. Method: " $FEATURE_SET
$ANACONDA_PYTHON_PATH/python2.7 $WORKDIR/main.py -t nn_a1a1 -i $A1A1_DATA -e 25 -f $FEATURE_SET -d 0.2 -l 6 -s 300
