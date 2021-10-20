import sys
import os, errno
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from scipy import optimize, stats
from src_py.metrics_utils import  calculate_deltas_signed

pathIN  = "../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_21/monit_npy/"
pathOUT = "figures/"

calc_w  = np.load(pathIN+'softmax_calc_w.npy')
preds_w = np.load(pathIN+'softmax_preds_w.npy')

k2PI = 6.28
#----------------------------------------------------------------------------------
                           
i = 1
filename = "calc_preds_w_rhorho_Variant-All_nc_21_event_1"
x = np.arange(1,22)
plt.plot(x,calc_w[i]/sum(calc_w[i]), 'o', label='generated')
plt.plot(x,preds_w[i], 'd', label='predicted')
plt.legend()
plt.ylim([0.0, 0.125])
plt.xlabel('Class index')
plt.xticks(x)
plt.ylabel(r'$wt^{norm}$')
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
filename = "calc_preds_w_rhorho_Variant-All_nc_21_event_10"
x = np.arange(1,22)
plt.plot(x,calc_w[i]/sum(calc_w[i]), 'o', label='generated')
plt.plot(x,preds_w[i], 'd', label='predicted')
plt.legend()
plt.ylim([0.0, 0.125])
plt.xlabel('Class index')
plt.xticks(x)
plt.ylabel(r'$wt^{norm}$')
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
filename = "calc_preds_w_rhorho_Variant-All_nc_21_event_1000"
x = np.arange(1,22)
plt.plot(x,calc_w[i]/sum(calc_w[i]), 'o', label='generated')
plt.plot(x,preds_w[i], 'd', label='predicted')
plt.legend()
plt.ylim([0.0, 0.1])
plt.xlabel('Class index')
plt.xticks(x)
plt.ylabel(r'$wt^{norm}$')
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
filename = "calc_preds_w_rhorho_Variant-All_nc_21_event_2000"
x = np.arange(1,22)
plt.plot(x,calc_w[i]/sum(calc_w[i]), 'o', label='generated')
plt.plot(x,preds_w[i], 'd', label='predicted')
plt.legend()
plt.ylim([0.0, 0.1])
plt.xlabel('Class index')
plt.xticks(x)
plt.ylabel(r'$wt^{norm}$')
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

pathIN  = "../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_5/monit_npy/"
pathOUT = "figures/"

calc_w_nc5  = np.load(pathIN+'softmax_calc_w.npy')
preds_w_nc5 = np.load(pathIN+'softmax_preds_w.npy')
delt_argmax_nc5 =  calculate_deltas_signed(np.argmax(preds_w_nc5[:], axis=1), np.argmax(calc_w_nc5[:], axis=1), 5)      

filename = "delt_argmax_rhorho_Variant-All_nc_5"
plt.hist(delt_argmax_nc5, histtype='step', bins=5)
plt.xlabel(r'$\Delta_{class}$')
plt.title('Features list: Variant-All')

ax = plt.gca()
nc5=5
mean = np.mean(delt_argmax_nc5) 
std  = np.std(delt_argmax_nc5) 
meanerr = stats.sem(delt_argmax_nc5)
meanrad = np.mean(delt_argmax_nc5, dtype=np.float64) * 6.28/nc5
stdrad  = np.std(delt_argmax_nc5, dtype=np.float64) * 6.28/nc5
meanerrrad = stats.sem(delt_argmax_nc5) * 6.28/nc5
ax.annotate("mean = {:0.3f}[idx] \n        +- {:1.3f}[idx] \nstd =  {:1.3f} [idx] ".format(mean,meanerr, std ), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)
ax.annotate("mean = {:0.3f}[rad] \n        +- {:1.3f}[rad] \nstd =  {:1.3f} [rad] ".format(meanrad,meanerrrad, stdrad ), xy=(0.65, 0.65), xycoords='axes fraction', fontsize=12)

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

pathIN  = "../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_11/monit_npy/"
pathOUT = "figures/"

calc_w_nc11  = np.load(pathIN+'softmax_calc_w.npy')
preds_w_nc11 = np.load(pathIN+'softmax_preds_w.npy')
preds_argmax_nc11 =  np.argmax(preds_w_nc11[:], axis=1) * k2PI/11.0    
calc_argmax_nc11 =  np.argmax(calc_w_nc11[:], axis=1) * k2PI/11.0    

delt_argmax_nc11 =  calculate_deltas_signed(np.argmax(preds_w_nc11[:], axis=1), np.argmax(calc_w_nc11[:], axis=1), 11)      

filename = "calc_argmax_rhorho_Variant-All_nc_11_soft"
plt.hist(calc_argmax_nc11, histtype='step', color = "black", bins=100, label = "generated")
plt.hist(preds_argmax_nc11, histtype='step', color = "red", bins=100, label = "predicted")
#plt.ylim([0, 800])
plt.ylabel('Entries')
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

filename = "delt_argmax_rhorho_Variant-All_nc_11"
plt.hist(delt_argmax_nc11, histtype='step', bins=11)
plt.ylabel('Entries')
plt.xlabel(r'$\Delta_{class}$')
plt.title('Features list: Variant-All')

ax = plt.gca()
nc11=11
mean = np.mean(delt_argmax_nc11, dtype=np.float64)
std  = np.std(delt_argmax_nc11, dtype=np.float64)
meanerr = stats.sem(delt_argmax_nc11)
meanrad = np.mean(delt_argmax_nc11, dtype=np.float64) * 6.28/11.0
stdrad  = np.std(delt_argmax_nc11, dtype=np.float64) * 6.28/11.0
meanerrrad = stats.sem(delt_argmax_nc11) * 6.28/11
ax.annotate("mean = {:0.3f}[idx] \n        +- {:1.3f}[idx] \nstd =  {:1.3f} [idx] ".format(mean,meanerr, std ), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)
ax.annotate("mean = {:0.3f}[rad] \n        +- {:1.3f}[rad] \nstd =  {:1.3f} [rad] ".format(meanrad,meanerrrad, stdrad ), xy=(0.65, 0.65), xycoords='axes fraction', fontsize=12)


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

pathIN  = "../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_21/monit_npy/"
pathOUT = "figures/"

calc_w_nc21  = np.load(pathIN+'softmax_calc_w.npy')
preds_w_nc21 = np.load(pathIN+'softmax_preds_w.npy')
delt_argmax_nc21 =  calculate_deltas_signed(np.argmax(preds_w_nc21[:], axis=1), np.argmax(calc_w_nc21[:], axis=1), 21)      

filename = "delt_argmax_rhorho_Variant-All_nc_21"
plt.hist(delt_argmax_nc21, histtype='step', bins=21)
plt.xlabel(r'$\Delta_{class}$')
plt.title('Features list: Variant-All')

nc21=21
mean = np.mean(delt_argmax_nc21, dtype=np.float64)
std  = np.std(delt_argmax_nc21, dtype=np.float64)
meanerr = stats.sem(delt_argmax_nc21)
meanrad = np.mean(delt_argmax_nc21, dtype=np.float64) * 6.28/21.0
stdrad  = np.std(delt_argmax_nc21, dtype=np.float64) * 6.28/21.0
meanerrrad = stats.sem(delt_argmax_nc21) * 6.28/21
ax.annotate("mean = {:0.3f}[idx] \n        +- {:1.3f}[idx] \nstd =  {:1.3f} [idx] ".format(mean,meanerr, std ), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)
ax.annotate("mean = {:0.3f}[rad] \n        +- {:1.3f}[rad] \nstd =  {:1.3f} [rad] ".format(meanrad,meanerrrad, stdrad ), xy=(0.65, 0.65), xycoords='axes fraction', fontsize=12)

ax = plt.gca()
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

pathIN  = "../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_25/monit_npy/"
pathOUT = "figures/"

calc_w_nc25  = np.load(pathIN+'softmax_calc_w.npy')
preds_w_nc25 = np.load(pathIN+'softmax_preds_w.npy')
delt_argmax_nc25 =  calculate_deltas_signed(np.argmax(preds_w_nc25[:], axis=1), np.argmax(calc_w_nc25[:], axis=1), 25)      

filename = "delt_argmax_rhorho_Variant-All_nc_25"
plt.hist(delt_argmax_nc25, histtype='step', bins=50)
plt.xlabel(r'$\Delta_{class}$')
plt.title('Features list: Variant-All')

ax = plt.gca()
nc25=25
mean = np.mean(delt_argmax_nc25, dtype=np.float64)
std  = np.std(delt_argmax_nc25, dtype=np.float64)
meanerr = stats.sem(delt_argmax_nc25)
meanrad = np.mean(delt_argmax_nc25, dtype=np.float64) * 6.28/25.0
stdrad  = np.std(delt_argmax_nc25, dtype=np.float64) * 6.28/25.0
meanerrrad = stats.sem(delt_argmax_nc25) * 6.28/25
ax.annotate("mean = {:0.3f}[idx] \n        +- {:1.3f}[idx] \nstd =  {:1.3f} [idx] ".format(mean,meanerr, std ), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)
ax.annotate("mean = {:0.3f}[rad] \n        +- {:1.3f}[rad] \nstd =  {:1.3f} [rad] ".format(meanrad,meanerrrad, stdrad ), xy=(0.65, 0.65), xycoords='axes fraction', fontsize=12)

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

pathIN  = "../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_51/monit_npy/"
pathOUT = "figures/"

calc_w_nc51  = np.load(pathIN+'softmax_calc_w.npy')
preds_w_nc51 = np.load(pathIN+'softmax_preds_w.npy')
delt_argmax_nc51 =  calculate_deltas_signed(np.argmax(preds_w_nc51[:], axis=1), np.argmax(calc_w_nc51[:], axis=1), 51)      

filename = "delt_argmax_rhorho_Variant-All_nc_51"
plt.hist(delt_argmax_nc51, histtype='step', bins=51)
plt.ylabel('Entries')
plt.xlabel(r'$\Delta_{class}$')
plt.title('Features list: Variant-All')

ax = plt.gca()
nc51=51
mean = np.mean(delt_argmax_nc51, dtype=np.float64)
std  = np.std(delt_argmax_nc51, dtype=np.float64)
meanerr = stats.sem(delt_argmax_nc51)
meanrad = np.mean(delt_argmax_nc51, dtype=np.float64) * 6.28/51.0
stdrad  = np.std(delt_argmax_nc51, dtype=np.float64) * 6.28/51.0
meanerrrad = stats.sem(delt_argmax_nc51) * 6.28/51
ax.annotate("mean = {:0.3f}[idx] \n        +- {:1.3f}[idx] \nstd =  {:1.3f} [idx] ".format(mean,meanerr, std ), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)
ax.annotate("mean = {:0.3f}[rad] \n        +- {:1.3f}[rad] \nstd =  {:1.3f} [rad] ".format(meanrad,meanerrrad, stdrad ), xy=(0.65, 0.65), xycoords='axes fraction', fontsize=12)


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

pathIN  = "../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_101/monit_npy/"
pathOUT = "figures/"

calc_w_nc101  = np.load(pathIN+'softmax_calc_w.npy')
preds_w_nc101 = np.load(pathIN+'softmax_preds_w.npy')
delt_argmax_nc101 = np.argmax(calc_w_nc101[:], axis=1) - np.argmax(preds_w_nc101[:], axis=1)
for i in range (len(delt_argmax_nc101)):
    if  delt_argmax_nc101[i] > 101.0/2.0 :
        delt_argmax_nc101[i] = 101.0 -  delt_argmax_nc101[i]
    if  delt_argmax_nc101[i] < - 101.0/2.0 :
        delt_argmax_nc101[i] = - 101.0 -  delt_argmax_nc101[i]

filename = "delt_argmax_rhorho_Variant-All_nc_101"
plt.hist(delt_argmax_nc101, histtype='step', bins=101)
plt.xlabel(r'$\Delta_{class}$')
plt.title('Features list: Variant-All')

ax = plt.gca()
nc101=101
mean = np.mean(delt_argmax_nc101, dtype=np.float64)
std  = np.std(delt_argmax_nc101, dtype=np.float64)
meanerr = stats.sem(delt_argmax_nc101)
meanrad = np.mean(delt_argmax_nc101, dtype=np.float64) * 6.28/101.0
stdrad  = np.std(delt_argmax_nc101, dtype=np.float64) * 6.28/101.0
meanerrrad = stats.sem(delt_argmax_nc101) * 6.28/101
ax.annotate("mean = {:0.3f}[idx] \n        +- {:1.3f}[idx] \nstd =  {:1.3f} [idx] ".format(mean,meanerr, std ), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)
ax.annotate("mean = {:0.3f}[rad] \n        +- {:1.3f}[rad] \nstd =  {:1.3f} [rad] ".format(meanrad,meanerrrad, stdrad ), xy=(0.65, 0.65), xycoords='axes fraction', fontsize=12)

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
classes = np.linspace(0, 2, 10) * np.pi
print classes
classes = np.linspace(0, 2, 11) * np.pi
print classes
