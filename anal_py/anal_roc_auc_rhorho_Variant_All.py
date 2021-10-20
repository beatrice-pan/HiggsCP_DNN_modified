import numpy as np
from glob import glob
import os, errno

import matplotlib.pyplot as plt

from anal_utils import test_roc_auc


filelist = []

filelist.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_2')
filelist.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_4')
filelist.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_6')
filelist.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_8')
filelist.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_10')
filelist.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_12')
filelist.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_14')
filelist.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_16')
filelist.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_18')
filelist.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_20')
filelist.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_25')
filelist.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_50')
filelist.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_100')

# predicted weights are used for binary classification ROC AUC

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
calc_roc_auc_20, roc_auc_20 = test_roc_auc(filelist[idx], nc)
#---------------------------------------------------------------------

pathOUT  = "figures/"
filename = "rhorho_roc_auc_w_Variant_All_nc_20"

x = np.arange(1,21)
plt.plot(x, calc_roc_auc_20,'o', label='Oracle')
plt.plot(x, roc_auc_20,'v', label='Variant-All')

plt.ylim([0.5, 0.8])
plt.xticks(x)
plt.legend()
plt.xlabel('Number of classes')
plt.ylabel('ROC AUC')
plt.title('Feautures list: Variant-All')

ax = plt.gca()
plt.tight_layout()

if filename:
    try:
        os.makedirs(pathOUT)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.savefig(pathOUT + filename+".eps")
    print('Saved '+pathOUT + filename+".eps")
    plt.savefig(pathOUT + filename+".pdf")
    print('Saved '+pathOUT + filename+".pdf")
else:
    plt.show()

#---------------------------------------------------------------------
plt.clf()
#---------------------------------------------------------------------

