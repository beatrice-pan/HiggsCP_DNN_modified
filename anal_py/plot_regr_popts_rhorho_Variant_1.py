import sys
import os, errno
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from scipy import optimize

from anal_utils import weight_fun



pathIN  = "npy/nn_rhorho_Variant-1.1_regr_popts_Unweighted_False_NO_NUM_CLASSES_0/monit_npy/"
pathOUT = "figures/"

calc_popts  = np.load(pathIN+'valid_regr_calc_popts.npy')
preds_popts = np.load(pathIN+'valid_regr_preds_popts.npy')

k2PI = 6.28
#----------------------------------------------------------------------------------
# ERW
# should normalise to same area which is not the case for now
                           
i = 1
filename = "regr_preds_popts_rhorho_Variant-1.1_event_1"
x = np.linspace(0, k2PI, 100)
plt.plot(x,weight_fun(x, *calc_popts[i]), 'o', label='calc_w')
plt.plot(x,weight_fun(x, *preds_popts[i]), 'd', label='preds_w')
plt.legend()
plt.ylim([0.0, 2.0])
plt.xlabel(r'$\alpha^{CP}$ [rad]')
plt.xticks(x)
plt.ylabel('w')
plt.title('Features list: Variant-1.1')
    
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
# ERW
# should normalise to same area which is not the case for now
                           
i = 10
filename = "regr_preds_popts_rhorho_Variant-1.1_event_10"
x = np.linspace(0, k2PI, 100)
plt.plot(x,weight_fun(x, *calc_popts[i]), 'o', label='calc_w')
plt.plot(x,weight_fun(x, *preds_popts[i]), 'd', label='preds_w')
plt.legend()
plt.ylim([0.0, 2.0])
plt.xlabel(r'$\alpha^{CP}$ [rad]')
plt.ylabel('w')
plt.title('Features list: Variant-1.1')
    
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
# ERW
# should normalise to same area which is not the case for now
                           
i = 100
filename = "regr_preds_popts_rhorho_Variant-1.1_event_100"
x = np.linspace(0, k2PI, 100)
plt.plot(x,weight_fun(x, *calc_popts[i]), 'o', label='calc_w')
plt.plot(x,weight_fun(x, *preds_popts[i]), 'd', label='preds_w')
plt.legend()
plt.ylim([0.0, 2.0])
plt.xlabel(r'$\alpha^{CP}$ [rad]')
plt.ylabel('w')
plt.title('Features list: Variant-1.1')
    
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

delt_popts= calc_popts - preds_popts

print calc_popts[:,0]
print delt_popts[:,0]
print delt_popts[:,0]/calc_popts[:,0]

filename = "delt_popts_A_rhorho_Variant-1.1"
plt.hist(delt_popts[:,0], histtype='step', bins=100)
plt.xlim([-1.5, 1.5])
plt.xlabel(r'$\Delta$A')
plt.title('Features list: Variant-1.1')

ax = plt.gca()
mean = np.mean(delt_popts[:,0])
std  = np.std(delt_popts[:,0])
ax.annotate("Mean = {:0.3f} \nSTD =  {:1.3f}".format(mean, std), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)

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

delt_popts= calc_popts - preds_popts

print calc_popts[:,0]
print delt_popts[:,0]
print delt_popts[:,0]/calc_popts[:,0]
#----------------------------------------------------------------------------------

filename = "popts_A_rhorho_Variant-1.1"
plt.hist(calc_popts[:,0], histtype='step', bins=100)
plt.hist(preds_popts[:,0], histtype='step', bins=100)
plt.xlim([-0.0, 2.0])
plt.xlabel(r'A')
plt.title('Features list: Variant-1.1')

ax = plt.gca()
mean = np.mean(calc_popts[:,0])
std  = np.std(calc_popts[:,0])
calc_mean = np.mean(calc_popts[:,0])
calc_std  = np.std(calc_popts[:,0])
preds_mean = np.mean(preds_popts[:,0])
preds_std  = np.std(preds_popts[:,0])
ax.annotate("Calc:  mean = {:0.3f}, \n           std =  {:1.3f}".format(calc_mean, calc_std), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)
ax.annotate("Preds: mean = {:0.3f}, \n           std =  {:1.3f}".format(preds_mean, preds_std), xy=(0.65, 0.65), xycoords='axes fraction', fontsize=12)


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
print calc_popts[:,1]
print delt_popts[:,1]
print delt_popts[:,1]/calc_popts[:,1]
#----------------------------------------------------------------------------------

filename = "popts_B_rhorho_Variant-1.1"
plt.hist(calc_popts[:,1], histtype='step', bins=100)
plt.hist(preds_popts[:,1], histtype='step', bins=100)
plt.xlim([-2.0, 2.0])
plt.xlabel(r'B')
plt.title('Features list: Variant-1.1')

ax = plt.gca()
calc_mean = np.mean(calc_popts[:,1])
calc_std  = np.std(calc_popts[:,1])
preds_mean = np.mean(preds_popts[:,1])
preds_std  = np.std(preds_popts[:,1])
ax.annotate("Calc:  mean = {:0.3f}, \n           std =  {:1.3f}".format(calc_mean, calc_std), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)
ax.annotate("Preds: mean = {:0.3f}, \n           std =  {:1.3f}".format(preds_mean, preds_std), xy=(0.65, 0.65), xycoords='axes fraction', fontsize=12)

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

filename = "delt_popts_B_rhorho_Variant-1.1"
plt.hist(delt_popts[:,1], histtype='step', bins=100)
plt.xlim([-1.5, 1.5])
plt.xlabel(r'$\Delta$B')
plt.title('Features list: Variant-1.1')

ax = plt.gca()
mean = np.mean(delt_popts[:,1])
std  = np.std(delt_popts[:,1])
ax.annotate("Mean = {:0.3f} \nSTD =  {:1.3f}".format(mean, std), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)


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
print calc_popts[:,2]
print delt_popts[:,2]
print delt_popts[:,2]/calc_popts[:,2]
#----------------------------------------------------------------------------------

filename = "popts_C_rhorho_Variant-1.1"
plt.hist(calc_popts[:,2], histtype='step', bins=100)
plt.hist(preds_popts[:,2], histtype='step', bins=100)
plt.xlim([-2.0, 2.0])
plt.xlabel(r'B')
plt.title('Features list: Variant-1.1')

ax = plt.gca()
calc_mean = np.mean(calc_popts[:,2])
calc_std  = np.std(calc_popts[:,2])
preds_mean = np.mean(preds_popts[:,2])
preds_std  = np.std(preds_popts[:,2])
ax.annotate("Calc:  mean = {:0.3f}, \n           std =  {:1.3f}".format(calc_mean, calc_std), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)
ax.annotate("Preds: mean = {:0.3f}, \n           std =  {:1.3f}".format(preds_mean, preds_std), xy=(0.65, 0.65), xycoords='axes fraction', fontsize=12)

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

filename = "delt_popts_C_rhorho_Variant-1.1"
plt.hist(delt_popts[:,2], histtype='step', bins=100)
plt.xlim([-1.5, 1.5])
plt.xlabel(r'$\Delta$C')
plt.title('Features list: Variant-1.1')

ax = plt.gca()
mean = np.mean(delt_popts[:,0])
std  = np.std(delt_popts[:,0])
ax.annotate("Mean = {:0.3f} \nSTD =  {:1.3f}".format(mean, std), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)

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
