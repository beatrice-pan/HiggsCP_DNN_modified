import os, errno
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from anal_utils import weight_fun, calc_weights

from scipy import optimize, stats
from src_py.metrics_utils import  calculate_deltas_signed

classes = np.linspace(0, 2, 51) * np.pi
print classes
print classes[0], classes[50]

print calculate_deltas_signed(0, 2, 3)
print calculate_deltas_signed(0, 50, 51)
print calculate_deltas_signed(0, 49, 51)
print calculate_deltas_signed(0, 48, 51)
print calculate_deltas_signed(1, 3, 51)
print calculate_deltas_signed(2, 48, 51)



pathIN  = "../laptop_results_dropout=0/nn_rhorho_Variant-All_regr_c012s_hits_c0s_Unweighted_False_NO_NUM_CLASSES_0/monit_npy/"
pathOUT = "figures/"

calc_c012s  = np.load(pathIN+'train_regr_calc_c012s.npy')


print calc_c012s[:,0]
#----------------------------------------------------------------------------------

filename = "c012s_C0_rhorho"
plt.hist(calc_c012s[:,0], histtype='step', color = 'black', linestyle='--', bins=50)
plt.xlim([-0.0, 2.0])
plt.xlabel(r'$C_{0}$')

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
print calc_c012s[:,1]
#----------------------------------------------------------------------------------

filename = "c012s_C1_rhorho"
plt.hist(calc_c012s[:,1], histtype='step', color = 'black', linestyle='--', bins=50)
plt.xlim([-1.2, 1.2])
plt.xlabel(r'$C_{0}$')

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
print calc_c012s[:,2]
#----------------------------------------------------------------------------------

filename = "c012s_C2_rhorho"
plt.hist(calc_c012s[:,2], histtype='step', color = 'black', linestyle='--', bins=50)
plt.xlim([-1.2, 1.2])
plt.xlabel(r'$C_{2}$')

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
