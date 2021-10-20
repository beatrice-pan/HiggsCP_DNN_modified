import sys
import os, errno
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from scipy import optimize

from anal_utils import weight_fun, calc_weights
from src_py.metrics_utils import  calculate_deltas_signed



pathIN  = "../laptop_results/nn_rhorho_Variant-All_regr_argmaxs_Unweighted_False_NO_NUM_CLASSES_0/monit_npy/"
pathOUT = "figures/"

calc_argmaxs  = np.load(pathIN+'test_regr_calc_argmaxs.npy')
preds_argmaxs = np.load(pathIN+'test_regr_preds_argmaxs.npy')


print calc_argmaxs
print preds_argmaxs
print calc_argmaxs - preds_argmaxs




delt_argmaxs = calc_argmaxs - preds_argmaxs
delt_argmaxs = calculate_deltas_signed(calc_argmaxs, preds_argmaxs, 10000.)

k2PI = 2 * np.pi
calc_argmaxs= calc_argmaxs/k2PI
print calc_argmaxs

#----------------------------------------------------------------------------------
filename = "calc_argmaxs_rhorho_Variant-All"

plt.hist(calc_argmaxs, histtype='step', bins=50,  color = 'black')
plt.xlim([0, k2PI])
plt.ylim([0, 1200])
plt.xlabel(r'$\alpha^{CP}_{max}$')
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
filename = "preds_argmaxs_rhorho_Variant-All"

plt.hist(calc_argmaxs, histtype='step', bins=50,  color = 'black', label="generated")
plt.hist(preds_argmaxs, histtype='step', bins=50, color = 'red', label="predicted")
plt.xlim([0, k2PI])
#plt.ylim([800, 1200])
plt.xlabel(r'$\alpha^{CP}_{max}$')
plt.title('Features list: Variant-All')
plt.legend()

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
filename = "delt_regr_argmaxs_rhorho_Variant-All"

plt.hist(delt_argmaxs, histtype='step', bins=50,  color = 'black')
plt.xlim([-k2PI, k2PI])
plt.xlabel(r'$\Delta \alpha^{CP}_{max}$')
plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_argmaxs)
std  = np.std(delt_argmaxs)
ax.annotate("mean = {:0.3f}[rad] \nstd =  {:1.3f} [rad]".format(mean, std), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)

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
