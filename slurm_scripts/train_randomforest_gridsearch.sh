#!/bin/bash -l
## Job name
#SBATCH -J TrainRandomForestGridsearch
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
#SBATCH --array=0-9

## Setup
if [ -f setup ]; then source setup; fi
export PYTHONPATH=$ANACONDA_PYTHON_PATH
module add plgrid/apps/cuda/7.5

## Chaninging dir to workdir
cd $WORKDIR

## Command
MAX_FEAT_OPTIONS=(log2 sqrt)
MAX_FEAT_ID=$(($SLURM_ARRAY_TASK_ID%2))
MAX_FEAT=${MAX_FEAT_OPTIONS[$MAX_FEAT_ID]}

FEATURE_SETS=(Model-Oracle Model-OnlyHad Model-Benchmark Model-1 Model-2)
FEATURE_SET_ID=$(($(($SLURM_ARRAY_TASK_ID-$MAX_FEAT_ID))/2))
FEATURE_SET=${FEATURE_SETS[$FEATURE_SET_ID]}

MAX_DEPTH_OPTIONS=(24 6 20 18 18)
MAX_DEPTH=${MAX_DEPTH_OPTIONS[$FEATURE_SET_ID]}

ESTIMATORS=128

echo "TrainRandomForestGridsearch Job. Feature set: "$FEATURE_SET" Max feat: "$MAX_FEAT" Max depth: "$MAX_DEPTH" Estimators: "$ESTIMATORS
$ANACONDA_PYTHON_PATH/python2.7 $WORKDIR/main.py -t random_forest -i $RHORHO_DATA -f $FEATURE_SET --forest_max_feat $MAX_FEAT --forest_max_depth $MAX_DEPTH --forest_estimators $ESTIMATORS