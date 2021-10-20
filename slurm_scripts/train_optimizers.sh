#!/bin/bash -l
## Job name
#SBATCH -J TrainOptimizers
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
#SBATCH --array=0-7

## Setup
if [ -f setup ]; then source setup; fi
export PYTHONPATH=$ANACONDA_PYTHON_PATH
module add plgrid/apps/cuda/7.5

## Chaninging dir to workdir
cd $WORKDIR

## Command
OPTIMIZERS=(GradientDescentOptimizer AdadeltaOptimizer AdagradOptimizer ProximalAdagradOptimizer AdamOptimizer FtrlOptimizer ProximalGradientDescentOptimizer RMSPropOptimizer)
OPTIMIZER=${OPTIMIZERS[$SLURM_ARRAY_TASK_ID]}
echo "TrainOptimizers Job. Optimizer: " $OPTIMIZER
$ANACONDA_PYTHON_PATH/python2.7 $WORKDIR/main.py -t nn_rhorho -i $RHORHO_DATA -e 50 -f Variant-1.0 -d 0.2 -o $OPTIMIZER -l 6 -s 300
