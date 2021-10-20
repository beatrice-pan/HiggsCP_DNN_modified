#!/bin/bash -l
## Job name
#SBATCH -J TrainSVMGridsearch
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
#SBATCH --array=0-48

## Setup
if [ -f setup ]; then source setup; fi
export PYTHONPATH=$ANACONDA_PYTHON_PATH
module add plgrid/apps/cuda/7.5

## Chaninging dir to workdir
cd $WORKDIR

## Command

FEATURE_SET=Model-Oracle
C_OPTIONS=(0.001 0.01 0.1 1 10 100 1000)
C_ID=$(($SLURM_ARRAY_TASK_ID%7))
C=${C_OPTIONS[$C_ID]}

GAMMA_OPTIONS=(0.001 0.01 0.1 1 10 100 1000)
GAMMA_ID=$(($(($SLURM_ARRAY_TASK_ID-$C_ID))/7))
GAMMA=${GAMMA_OPTIONS[$GAMMA_ID]}

echo "TrainSVMGridsearch Job. Method: "$FEATURE_SET" C: "$C" GAMMA: "$GAMMA
$ANACONDA_PYTHON_PATH/python2.7 $WORKDIR/main.py -t svm -i $RHORHO_DATA -f $FEATURE_SET --miniset True --svm_c $C --svm_gamma $GAMMA