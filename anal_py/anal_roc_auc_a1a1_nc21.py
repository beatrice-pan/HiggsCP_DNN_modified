import numpy as np
from glob import glob
import os, errno

import matplotlib.pyplot as plt

from anal_utils import test_roc_auc

file_Variant_1 = 'npy/nn_a1a1_Variant-1.1_Unweighted_False_NO_NUM_CLASSES_20'
file_Variant_4 = 'npy/nn_a1a1_Variant-4.1_Unweighted_False_NO_NUM_CLASSES_20'
file_Variant_All = 'npy/nn_a1a1_Variant-All_Unweighted_False_NO_NUM_CLASSES_20'

nc = 20
oracle_roc_auc_Variant_All, roc_auc_Variant_All = test_roc_auc(file_Variant_All, nc)
oracle_roc_auc_Variant_1, roc_auc_Variant_1 = test_roc_auc(file_Variant_1, nc)
oracle_roc_auc_Variant_4, roc_auc_Variant_4 = test_roc_auc(file_Variant_4, nc)

#---------------------------------------------------------------------

pathOUT  = "figures/"
filename = "a1a1_roc_auc_w_nc_20"

x = np.arange(1,21)
plt.plot(x, oracle_roc_auc_Variant_All,'o', label='Oracle')
plt.plot(x, roc_auc_Variant_All,'x', label='Variant-All')
plt.plot(x, roc_auc_Variant_4,'d', label='Variant-4.1')
plt.plot(x, roc_auc_Variant_1,'v', label='Variant-1.1')

plt.ylim([0.5, 0.8])
plt.xticks(x)
plt.legend()
plt.xlabel('Class index')
plt.ylabel('AUC vs class index = 1')
plt.title(r'$a_1^\pm-a_1^\mp$ channel;class index =1, 20 (scalar), =10 (pseudoscalar)')

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
