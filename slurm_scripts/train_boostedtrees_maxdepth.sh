#!/bin/bash -l
## Job name
#SBATCH -J TrainBoostedTreesMaxDepth
## Number of nodes
#SBATCH -N 1
## Number of tasks per node
#SBATCH --ntasks-per-node=4
## Memory per CPU (default: 5GB)
#SBATCH --mem-per-cpu=5GB
## Max job time (HH:MM:SS format)
#SBATCH --time=72:00:00
## Pratition specification
#SBATCH -p plgrid
#SBATCH --array=3-20

## Setup
if [ -f setup ]; then source setup; fi
export PYTHONPATH=$ANACONDA_PYTHON_PATH
module add plgrid/apps/cuda/7.5

## Chaninging dir to workdir
cd $WORKDIR

## Command
echo "TrainBoostedTreesMaxDepth Model-OnlyHad Job. Depth: " $SLURM_ARRAY_TASK_ID
$ANACONDA_PYTHON_PATH/python2.7 $WORKDIR/main.py -t boosted_trees -i $RHORHO_DATA -f Model-OnlyHad --treedepth $SLURM_ARRAY_TASK_ID