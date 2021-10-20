import numpy as np
from glob import glob
import os, errno

import matplotlib.pyplot as plt

from anal_utils import evaluate_roc_auc  

directory = '../../HiggsCP_data/a1rho/'
w_00 = np.load(os.path.join(directory, 'a1rho_raw.w_00.npy'))
w_10 = np.load(os.path.join(directory, 'a1rho_raw.w_10.npy'))
w_20 = np.load(os.path.join(directory, 'a1rho_raw.w_20.npy'))

w_02 = np.load(os.path.join(directory, 'a1rho_raw.w_02.npy'))
w_04 = np.load(os.path.join(directory, 'a1rho_raw.w_04.npy'))
w_06 = np.load(os.path.join(directory, 'a1rho_raw.w_06.npy'))
w_08 = np.load(os.path.join(directory, 'a1rho_raw.w_08.npy'))
w_12 = np.load(os.path.join(directory, 'a1rho_raw.w_12.npy'))
w_14 = np.load(os.path.join(directory, 'a1rho_raw.w_14.npy'))
w_16 = np.load(os.path.join(directory, 'a1rho_raw.w_16.npy'))
w_18 = np.load(os.path.join(directory, 'a1rho_raw.w_18.npy'))

for i in range(0, 10):
    print w_00[i], w_10[i], w_20[i]
print '-------------------'

roc_auc = evaluate_roc_auc(w_10/(w_00+w_10), w_10, w_00)
print  'oracle  s/ps        roc_auc =', roc_auc
print '-------------------'

roc_auc_oracle = []

roc_auc = evaluate_roc_auc(w_00/(w_00+w_00), w_00, w_00)
roc_auc_oracle += [roc_auc]
print  'oracle 00/00           roc_auc =', roc_auc
print '-------------------'

roc_auc = evaluate_roc_auc(w_00/(w_00+w_02), w_00, w_02)
roc_auc_oracle += [roc_auc]
print  'oracle 00/02           roc_auc =', roc_auc
print '-------------------'

roc_auc = evaluate_roc_auc(w_00/(w_00+w_04), w_00, w_04)
roc_auc_oracle += [roc_auc]
print  'oracle 00/04           roc_auc =', roc_auc
print '-------------------'

roc_auc = evaluate_roc_auc(w_00/(w_00+w_06), w_00, w_06)
roc_auc_oracle += [roc_auc]
print  'oracle 00/06           roc_auc =', roc_auc
print '-------------------'

roc_auc = evaluate_roc_auc(w_00/(w_00+w_08), w_00, w_08)
roc_auc_oracle += [roc_auc]
print  'oracle 00/08           roc_auc =', roc_auc
print '-------------------'

roc_auc = evaluate_roc_auc(w_00/(w_00+w_10), w_00, w_10)
roc_auc_oracle += [roc_auc]
print  'oracle 00/10           roc_auc =', roc_auc
print '-------------------'

roc_auc = evaluate_roc_auc(w_00/(w_00+w_12), w_00, w_12)
roc_auc_oracle += [roc_auc]
print  'oracle 00/12           roc_auc =', roc_auc
print '-------------------'

roc_auc = evaluate_roc_auc(w_00/(w_00+w_14), w_00, w_14)
roc_auc_oracle += [roc_auc]
print  'oracle 00/14           roc_auc =', roc_auc
print '-------------------'

roc_auc = evaluate_roc_auc(w_00/(w_00+w_16), w_00, w_16)
roc_auc_oracle += [roc_auc]
print  'oracle 00/16           roc_auc =', roc_auc
print '-------------------'

roc_auc = evaluate_roc_auc(w_00/(w_00+w_18), w_00, w_18)
roc_auc_oracle += [roc_auc]
print  'oracle 00/18           roc_auc =', roc_auc
print '-------------------'

roc_auc = evaluate_roc_auc(w_00/(w_00+w_20), w_00, w_20)
roc_auc_oracle += [roc_auc]
print  'oracle 00/20           roc_auc =', roc_auc
print '-------------------'
#---------------------------------------------------------------------

pathOUT  = "figures/"
filename = "a1rho_roc_auc_w_nc_20"

x = np.arange(0,11)*2
plt.plot(x,roc_auc_oracle,'o', label='Oracle')

plt.ylim([0.5, 0.8])
plt.xticks(x)
plt.legend()
plt.xlabel('Index of class')
plt.ylabel('ROC AUC')
plt.title('Matrix element spin weights')

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
