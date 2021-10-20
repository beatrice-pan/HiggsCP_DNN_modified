import sys
import os, errno
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from scipy import optimize, stats
from src_py.metrics_utils import  calculate_deltas_signed

pathIN  = "../laptop_results/nn_rhorho_Variant-All_regr_weights_Unweighted_False_NO_NUM_CLASSES_21/monit_npy/"
pathOUT = "figures/"

calc_w  = np.load(pathIN+'test_regr_calc_weights.npy')
preds_w = np.load(pathIN+'test_regr_preds_weights.npy')

k2PI = 6.28
#----------------------------------------------------------------------------------
                           
i = 1
filename = "calc_preds_regr_w_rhorho_Variant-All_nc_21_event_1"
x = np.arange(1,22)
plt.plot(x,calc_w[i], 'o', label='generated')
plt.plot(x,preds_w[i], 'd', label='predicted')
plt.legend()
#plt.ylim([0.0, 0.125])
plt.xlabel('Class index')
plt.xticks(x)
plt.ylabel(r'$wt$')
plt.title('Features list: Variant-All')
    
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
                           
i = 10
filename = "calc_preds_regr_w_rhorho_Variant-All_nc_21_event_10"
x = np.arange(1,22)
plt.plot(x,calc_w[i], 'o', label='generated')
plt.plot(x,preds_w[i], 'd', label='predicted')
plt.legend()
#plt.ylim([0.0, 0.125])
plt.xlabel('Class index')
plt.xticks(x)
plt.ylabel(r'$wt$')
plt.title('Features list: Variant-All')
    
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
                           
i = 1000
filename = "calc_preds_regr_w_rhorho_Variant-All_nc_21_event_1000"
x = np.arange(1,22)
plt.plot(x,calc_w[i], 'o', label='generated')
plt.plot(x,preds_w[i], 'd', label='predicted')
plt.legend()
#plt.ylim([0.0, 0.1])
plt.xlabel('Class index')
plt.xticks(x)
plt.ylabel(r'$wt$')
plt.title('Features list: Variant-All')
    
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
                           
i = 2000
filename = "calc_preds_regr_w_rhorho_Variant-All_nc_21_event_2000"
x = np.arange(1,22)
plt.plot(x,calc_w[i], 'o', label='generated')
plt.plot(x,preds_w[i], 'd', label='predicted')
plt.legend()
#plt.ylim([0.0, 0.1])
plt.xlabel('Class index')
plt.xticks(x)
plt.ylabel(r'$wt$')
plt.title('Features list: Variant-All')
    
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
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------

pathIN  = "../laptop_results/nn_rhorho_Variant-All_regr_weights_Unweighted_False_NO_NUM_CLASSES_21/monit_npy/"
pathOUT = "figures/"

calc_w_nc21  = np.load(pathIN+'test_regr_calc_weights.npy')
preds_w_nc21 = np.load(pathIN+'test_regr_preds_weights.npy')
delt_argmax_nc21 =  calculate_deltas_signed(np.argmax(preds_w_nc21[:], axis=1), np.argmax(calc_w_nc21[:], axis=1), 21)      

filename = "delt_argmax_rhorho_Variant-All_nc_21"
plt.hist(delt_argmax_nc21, histtype='step', bins=21)
plt.xlabel(r'$\Delta_{class}$')
plt.title('Features list: Variant-All')

ax = plt.gca()
nc21=21
mean = np.mean(delt_argmax_nc21, dtype=np.float64)
std  = np.std(delt_argmax_nc21, dtype=np.float64)
meanerr = stats.sem(delt_argmax_nc21)
meanrad = np.mean(delt_argmax_nc21, dtype=np.float64) * 6.28/21.0
stdrad  = np.std(delt_argmax_nc21, dtype=np.float64) * 6.28/21.0
meanerrrad = stats.sem(delt_argmax_nc21) * 6.28/21
ax.annotate("mean = {:0.3f}[idx] \n        +- {:1.3f}[idx] \nstd =  {:1.3f} [idx] ".format(mean,meanerr, std ), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)
ax.annotate("mean = {:0.3f}[rad] \n        +- {:1.3f}[rad] \nstd =  {:1.3f} [rad] ".format(meanrad,meanerrrad, stdrad ), xy=(0.65, 0.65), xycoords='axes fraction', fontsize=12)

nc21=21
mean = np.mean(delt_argmax_nc21) * 6.28/21.0
std  = np.std(delt_argmax_nc21) * 6.28/21.0
ax.annotate("mean = {:0.3f} [rad] \nstd =  {:1.3f} [rad]".format(mean, std), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)

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
#----------------------------------------------------------------------------------
