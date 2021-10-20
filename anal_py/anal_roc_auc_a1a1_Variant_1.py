import numpy as np
from glob import glob
import os, errno

import matplotlib.pyplot as plt

from anal_utils import test_roc_auc


filelist = []

filelist.append('npy/nn_a1a1_Variant-1.1_Unweighted_False_NO_NUM_CLASSES_2')
filelist.append('npy/nn_a1a1_Variant-1.1_Unweighted_False_NO_NUM_CLASSES_4')
filelist.append('npy/nn_a1a1_Variant-1.1_Unweighted_False_NO_NUM_CLASSES_6')
filelist.append('npy/nn_a1a1_Variant-1.1_Unweighted_False_NO_NUM_CLASSES_8')
filelist.append('npy/nn_a1a1_Variant-1.1_Unweighted_False_NO_NUM_CLASSES_10')
filelist.append('npy/nn_a1a1_Variant-1.1_Unweighted_False_NO_NUM_CLASSES_12')
filelist.append('npy/nn_a1a1_Variant-1.1_Unweighted_False_NO_NUM_CLASSES_14')
filelist.append('npy/nn_a1a1_Variant-1.1_Unweighted_False_NO_NUM_CLASSES_16')
filelist.append('npy/nn_a1a1_Variant-1.1_Unweighted_False_NO_NUM_CLASSES_18')
filelist.append('npy/nn_a1a1_Variant-1.1_Unweighted_False_NO_NUM_CLASSES_20')

idx = 0
nc = 2
test_roc_auc(filelist[idx], nc)

idx = 1
nc = 4
test_roc_auc(filelist[idx], nc)

idx = 2
nc = 6
test_roc_auc(filelist[idx], nc)

idx = 9
nc = 20
test_roc_auc(filelist[idx], nc)
