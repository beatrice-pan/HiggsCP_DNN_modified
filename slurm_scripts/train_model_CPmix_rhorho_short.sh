#!/bin/bash -l
## Job name
#SBATCH -J TrainTauVertexRHORHO
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
#SBATCH --array=0-0

## Setup
if [ -f setup ]; then source setup; fi
export PYTHONPATH=$ANACONDA_PYTHON_PATH
module add plgrid/apps/cuda/7.5

## Chaninging dir to workdir
cd $WORKDIR

## Command

echo "TrainCPmix Job. Dropout 0.2. RHORHO"
$ANACONDA_PYTHON_PATH/python2.7 $WORKDIR/main.py -t nn_rhorho_CPmix -i $RHORHO_DATA -e 5 -f Variant-1.0 -d 0.2 -l 6 -s 300
