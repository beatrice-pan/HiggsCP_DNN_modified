#!/bin/bash -l
## Job name
#SBATCH -J TrainAllMethods
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
#SBATCH --array=0-19

## Setup
if [ -f setup ]; then source setup; fi
export PYTHONPATH=$ANACONDA_PYTHON_PATH
module add plgrid/apps/cuda/7.5

## Chaninging dir to workdir
cd $WORKDIR

FEATURE_SETS=(Variant-All Variant-1.0 Variant-1.1 Variant-2.0 Variant-2.1 Variant-2.2 Variant-3.0)
FEATURE_SET_ID=$(($SLURM_ARRAY_TASK_ID%5))
FEATURE_SET=${FEATURE_SETS[$FEATURE_SET_ID]}
CLASSIFIER=$(($(($SLURM_ARRAY_TASK_ID-$FEATURE_SET_ID))/5))
TREE_DEPTHS=(24 6 20 18 18)
TREE_DEPTH=${TREE_DEPTHS[$FEATURE_SET_ID]}

## Command
case $CLASSIFIER in
    0 )
		echo "Neural network "$FEATURE_SET
        time $ANACONDA_PYTHON_PATH/python2.7 $WORKDIR/main.py -t nn_rhorho -i $RHORHO_DATA -e 250 -f $FEATURE_SET -d 0.2 -l 6 -s 300
        ;;
    1 )
		echo "SVM "$FEATURE_SET
        time $ANACONDA_PYTHON_PATH/python2.7 $WORKDIR/main.py -t svm -i $RHORHO_DATA -f $FEATURE_SET --miniset True --svm_c 10 --svm_gamma 0.1
        ;;
    2 ) 
		echo "Boosted trees "$FEATURE_SET
		time $ANACONDA_PYTHON_PATH/python2.7 $WORKDIR/main.py -t boosted_trees -i $RHORHO_DATA -f $FEATURE_SET --treedepth $TREE_DEPTH
		;;
	3 )
		echo "Random forest "$FEATURE_SET
		time $ANACONDA_PYTHON_PATH/python2.7 $WORKDIR/main.py -t random_forest -i $RHORHO_DATA -f $FEATURE_SET --forest_max_feat sqrt --forest_max_depth 30 --forest_estimators 300
		;;
esac
