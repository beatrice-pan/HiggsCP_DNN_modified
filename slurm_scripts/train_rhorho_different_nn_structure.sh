#!/bin/bash -l
## Job name
#SBATCH -J TrainRhoRhoDifferentNNStructure
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
#SBATCH --array=0-89

## Setup
if [ -f setup ]; then source setup; fi
export PYTHONPATH=$ANACONDA_PYTHON_PATH
module add plgrid/apps/cuda/7.5

## Chaninging dir to workdir
cd $WORKDIR

## Command
LAYERS=(2 3 4 5 6 7 8 9 10)
LAYER_ID=$(($SLURM_ARRAY_TASK_ID%9))
LAYER_NUM=${LAYERS[$LAYER_ID]}

SIZES=(20 50 100 200 300 400 500 600 700 800)
SIZE_ID=$(($(($SLURM_ARRAY_TASK_ID-$LAYER_ID))/9))
SIZE=${SIZES[$SIZE_ID]}

FEATURE_SETS=(Model-Oracle Model-OnlyHad Model-Benchmark Model-1 Model-2)
FEATURE_SET=${FEATURE_SETS[$SLURM_ARRAY_TASK_ID]}
echo "TrainRhoRhoDifferentNNStructure Job. Dropout. Model-OnlyHad. Layers: " $LAYER_NUM " Size: " $SIZE

$ANACONDA_PYTHON_PATH/python2.7 $WORKDIR/main.py -t nn_rhorho -i $RHORHO_DATA -e 150 -f Variant-1.0 -d 0.2 -l $LAYER_NUM -s $SIZE
