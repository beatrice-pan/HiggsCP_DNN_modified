import sys
import os, errno
import matplotlib.pyplot as plt
import numpy as np

from scipy import optimize

pathIN  = "npy/nn_a1rho_Variant-All_Unweighted_False_NO_NUM_CLASSES_10/"
pathOUT = "figures/"

calc_w  = np.load(pathIN+'softmax_calc_w.npy')
preds_w = np.load(pathIN+'softmax_preds_w.npy')

#----------------------------------------------------------------------------------
#ERW
# why it is plotting two dots in the legend box?

i = 1
filename = "calc_preds_w_a1rho_Variant-All_nc_10_event_1"
x = np.arange(1,11)
plt.plot(x,calc_w[i]/sum(calc_w[i]), 'o', label='calc_w')
plt.plot(x,preds_w[i], 'd', label='preds_w')
plt.legend()
plt.ylim([0.0, 0.2])
plt.xlabel('Index of class')
plt.xticks(x)
plt.ylabel('w')
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

pathIN  = "npy/nn_a1rho_Variant-All_Unweighted_False_NO_NUM_CLASSES_4/"
pathOUT = "figures/"

calc_w_nc4  = np.load(pathIN+'softmax_calc_w.npy')
preds_w_nc4 = np.load(pathIN+'softmax_preds_w.npy')
delt_argmax_nc4 = np.argmax(calc_w_nc4[:], axis=1) - np.argmax(preds_w_nc4[:], axis=1)
for i in range (len(delt_argmax_nc4)):
    if  delt_argmax_nc4[i] > 4.0/2.0 :
        delt_argmax_nc4[i] = 4.0 -  delt_argmax_nc4[i]
    if  delt_argmax_nc4[i] < - 4.0/2.0 :
        delt_argmax_nc4[i] = - 4.0 -  delt_argmax_nc4[i]

filename = "delt_argmax_a1rho_Variant-All_nc_4"
plt.hist(delt_argmax_nc4, histtype='step', bins=100)
plt.xlabel(r'$\Delta$  class index')
plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_argmax_nc4) * 6.28/4.0
std  = np.std(delt_argmax_nc4) * 6.28/4.0
ax.annotate("Mean = {:0.3f} (rad) \nSTD =  {:1.3f} (rad)".format(mean, std), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)

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

acc0 = (np.abs(np.argmax(calc_w_nc4[:], axis=1) - np.argmax(preds_w_nc4[:], axis=1)) <= 0).mean()
acc1 = (np.abs(np.argmax(calc_w_nc4[:], axis=1) - np.argmax(preds_w_nc4[:], axis=1)) <= 1).mean()
acc2 = (np.abs(np.argmax(calc_w_nc4[:], axis=1) - np.argmax(preds_w_nc4[:], axis=1)) <= 2).mean()
acc3 = (np.abs(np.argmax(calc_w_nc4[:], axis=1) - np.argmax(preds_w_nc4[:], axis=1)) <= 3).mean()
print('---------')
print('Acc0_nc_4', acc0)
print('Acc1_nc_4', acc1)
print('Acc2_nc_4', acc2)
print('Acc3_nc_4', acc3)
print('---------')
#----------------------------------------------------------------------------------

pathIN  = "npy/nn_a1rho_Variant-All_Unweighted_False_NO_NUM_CLASSES_10/"
pathOUT = "figures/"

calc_w_nc10  = np.load(pathIN+'softmax_calc_w.npy')
preds_w_nc10 = np.load(pathIN+'softmax_preds_w.npy')
delt_argmax_nc10 = np.argmax(calc_w[:], axis=1) - np.argmax(preds_w[:], axis=1)
for i in range (len(delt_argmax_nc10)):
    if  delt_argmax_nc10[i] > 10.0/2.0 :
        delt_argmax_nc10[i] = 10.0 -  delt_argmax_nc10[i]
    if  delt_argmax_nc10[i] < - 10.0/2.0 :
        delt_argmax_nc10[i] = - 10.0 -  delt_argmax_nc10[i]

filename = "delt_argmax_a1rho_Variant-All_nc_10"
plt.hist(delt_argmax_nc10, histtype='step', bins=100)
plt.xlabel(r'$\Delta$  class index')
plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_argmax_nc10) * 6.28/10.0
std  = np.std(delt_argmax_nc10) * 6.28/10.0
ax.annotate("Mean = {:0.3f} (rad) \nSTD =  {:1.3f} (rad)".format(mean, std), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)

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

acc0 = (np.abs(np.argmax(calc_w_nc10[:], axis=1) - np.argmax(preds_w_nc10[:], axis=1)) <= 0).mean()
acc1 = (np.abs(np.argmax(calc_w_nc10[:], axis=1) - np.argmax(preds_w_nc10[:], axis=1)) <= 1).mean()
acc2 = (np.abs(np.argmax(calc_w_nc10[:], axis=1) - np.argmax(preds_w_nc10[:], axis=1)) <= 2).mean()
acc3 = (np.abs(np.argmax(calc_w_nc10[:], axis=1) - np.argmax(preds_w_nc10[:], axis=1)) <= 3).mean()
print('---------')
print('Acc0_nc_10', acc0)
print('Acc1_nc_10', acc1)
print('Acc2_nc_10', acc2)
print('Acc3_nc_10', acc3)
print('---------')
#----------------------------------------------------------------------------------

pathIN  = "npy/nn_a1rho_Variant-All_Unweighted_False_NO_NUM_CLASSES_20/"
pathOUT = "figures/"

calc_w_nc20  = np.load(pathIN+'softmax_calc_w.npy')
preds_w_nc20 = np.load(pathIN+'softmax_preds_w.npy')
delt_argmax_nc20 = np.argmax(calc_w_nc20[:], axis=1) - np.argmax(preds_w_nc20[:], axis=1)
for i in range (len(delt_argmax_nc20)):
    if  delt_argmax_nc20[i] > 20.0/2.0 :
        delt_argmax_nc20[i] = 20.0 -  delt_argmax_nc20[i]
    if  delt_argmax_nc20[i] < - 20.0/2.0 :
        delt_argmax_nc20[i] = - 20.0 -  delt_argmax_nc20[i]

filename = "delt_argmax_a1rho_Variant-All_nc_20"
plt.hist(delt_argmax_nc20, histtype='step', bins=100)
plt.xlabel(r'$\Delta$  class index')
plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_argmax_nc20) * 6.28/20.0
std  = np.std(delt_argmax_nc20) * 6.28/20.0
ax.annotate("Mean = {:0.3f} (rad) \nSTD =  {:1.3f} (rad)".format(mean, std), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)

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

acc0 = (np.abs(np.argmax(calc_w_nc20[:], axis=1) - np.argmax(preds_w_nc20[:], axis=1)) <= 0).mean()
acc1 = (np.abs(np.argmax(calc_w_nc20[:], axis=1) - np.argmax(preds_w_nc20[:], axis=1)) <= 1).mean()
acc2 = (np.abs(np.argmax(calc_w_nc20[:], axis=1) - np.argmax(preds_w_nc20[:], axis=1)) <= 2).mean()
acc3 = (np.abs(np.argmax(calc_w_nc20[:], axis=1) - np.argmax(preds_w_nc20[:], axis=1)) <= 3).mean()
print('---------')
print('Acc0_nc_20', acc0)
print('Acc1_nc_20', acc1)
print('Acc2_nc_20', acc2)
print('Acc3_nc_20', acc3)
print('---------')
#----------------------------------------------------------------------------------
