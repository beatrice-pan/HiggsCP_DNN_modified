import sys
import os, errno
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from scipy import optimize
from src_py.metrics_utils import  calculate_deltas_signed

pathIN  = "../laptop_results/nn_rhorho_Variant-All_soft_argmaxs_Unweighted_False_NO_NUM_CLASSES_21/monit_npy/"
pathOUT = "figures/"

calc_hits_argmaxs  = np.load(pathIN+'train_soft_calc_hits_argmaxs.npy')
preds_hits_argmaxs  = np.load(pathIN+'train_soft_preds_hits_argmaxs.npy')

print calc_hits_argmaxs[0]
print preds_hits_argmaxs[0]

data_len = calc_hits_argmaxs.shape[0]
preds_argmaxs = np.zeros((data_len, 1))
calc_argmaxs = np.zeros((data_len, 1))

for i in range(data_len):
    preds_argmaxs[i] = np.argmax(preds_hits_argmaxs[i])
    calc_argmaxs[i] = np.argmax(calc_hits_argmaxs[i])


delt_argmaxs =  calculate_deltas_signed(np.argmax(preds_hits_argmaxs[:], axis=1), np.argmax(calc_hits_argmaxs[:], axis=1), 21)      

k2PI= 2* np.pi
#----------------------------------------------------------------------------------
filename = "soft_calc_argmaxs_rhorho_Variant-All_nc21"

plt.hist(calc_argmaxs, histtype='step', bins=20,  color = 'black', label = "generated")
plt.hist(preds_argmaxs, histtype='step', bins=20,  color = 'red', label = "predicted")
plt.xlim([0, 20])
plt.xlabel(r'$Class index$')
plt.title('Features list: Variant-All')

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

plt.clf()

#----------------------------------------------------------------------------------
filename = "delt_soft_argmaxs_rhorho_Variant-All_nc21"

plt.hist(delt_argmaxs, histtype='step', bins=21,  color = 'black')
plt.xlabel(r'$\Delta_{class}$')
plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_argmaxs)
std  = np.std(delt_argmaxs)
meanrad = np.mean(delt_argmaxs) * k2PI/21.0
stdrad  = np.std(delt_argmaxs) * k2PI/21.0
ax.annotate("mean = {:0.3f}[idx] \nstd =  {:1.3f}[idx]".format(mean, std), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)
ax.annotate("mean = {:0.3f} [rad] \nstd =  {:1.3f} [rad]".format(meanrad, stdrad), xy=(0.65, 0.65), xycoords='axes fraction', fontsize=12)

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

plt.clf()
