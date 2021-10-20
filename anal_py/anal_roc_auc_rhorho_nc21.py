import numpy as np
from glob import glob
import os, errno

import matplotlib.pyplot as plt

from anal_utils import test_roc_auc

file_Variant_1 = '../laptop_results/nn_rhorho_Variant-1.1_soft_Unweighted_False_NO_NUM_CLASSES_21/monit_npy/'
file_Variant_4 = '../laptop_results/nn_rhorho_Variant-4.1_soft_Unweighted_False_NO_NUM_CLASSES_21/monit_npy/'
file_Variant_All = '../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_21/monit_npy/'

nc = 21
oracle_roc_auc_Variant_All, roc_auc_Variant_All = test_roc_auc(file_Variant_All, nc)
oracle_roc_auc_Variant_1, roc_auc_Variant_1 = test_roc_auc(file_Variant_1, nc)
oracle_roc_auc_Variant_4, roc_auc_Variant_4 = test_roc_auc(file_Variant_4, nc)

#---------------------------------------------------------------------

pathOUT  = "figures/"
filename = "rhorho_roc_auc_w_nc_21"

k2PI=6.28
x = np.linspace(0, k2PI, 21)
plt.plot(x, oracle_roc_auc_Variant_All,'o', label='Oracle')
plt.plot(x, roc_auc_Variant_All,'x', label='Variant-All')
plt.plot(x, roc_auc_Variant_4,'d', label='Variant-4.1')
plt.plot(x, roc_auc_Variant_1,'v', label='Variant-1.1')

plt.ylim([0.5, 0.85])
#plt.xticks(x)
plt.legend()
plt.xlabel(r'$\alpha^{CP}$ [rad]')
plt.ylabel(r'AUC vs $\alpha^{CP}$ = 0.0')
plt.title(r'$\rho^\pm-\rho^\mp$ channel')

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
