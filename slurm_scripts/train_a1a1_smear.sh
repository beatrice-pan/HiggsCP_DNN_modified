#!/bin/bash -l
## Job name
#SBATCH -J TrainSmearA1A1
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
#SBATCH --array=0-8

## Setup
if [ -f setup ]; then source setup; fi
export PYTHONPATH=$ANACONDA_PYTHON_PATH
module add plgrid/apps/cuda/7.5

## Chaninging dir to workdir
cd $WORKDIR

## Command
BETAS=(0.2 0.4 0.6 0.2 0.4 0.6 0.2 0.4 0.6)
BS=(0 0 0 0.3 0.3 0.3 0.9 0.9 0.9)
CS=(0 0 0 0.8 0.8 0.8 0.9 0.9 0.9)
BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}
B=${BS[$SLURM_ARRAY_TASK_ID]}
C=${CS[$SLURM_ARRAY_TASK_ID]}

echo "TrainSmear a1a1 Job. Beta: " $BETA
echo "B: " $B
echo "C: " $C

$ANACONDA_PYTHON_PATH/python2.7 $WORKDIR/main.py -t nn_a1a1 -i $A1A1_DATA -e 25 -f Variant-3.1 -d 0.2 --lambda $BETA --beta $BETA --pol_b $B --pol_c $C -l 6 -s 300
