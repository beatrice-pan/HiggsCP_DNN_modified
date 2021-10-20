import sys
import os, errno
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
#


from scipy import optimize


pathIN  = "../temp_results/nn_rhorho_Variant-All_soft_weights_hits_c0s_Unweighted_False_NO_NUM_CLASSES_21_SIZE_1000_LAYERS_6/monit_npy/"
pathOUT = "figures/"

train_losses    = np.load(pathIN+'train_losses.npy')
valid_losses    = np.load(pathIN+'valid_losses.npy')

train_accs      = np.load(pathIN+'train_accs.npy')
test_accs       = np.load(pathIN+'test_accs.npy')
valid_accs      = np.load(pathIN+'valid_accs.npy')

train_l1_deltas = np.load(pathIN+'train_L1_deltas.npy')
test_l1_deltas  = np.load(pathIN+'test_L1_deltas.npy')
valid_l1_deltas = np.load(pathIN+'valid_L1_deltas.npy')

train_l2_deltas = np.load(pathIN+'train_L2_deltas.npy')
test_l2_deltas  = np.load(pathIN+'test_L2_deltas.npy')
valid_l2_deltas = np.load(pathIN+'valid_L2_deltas.npy')

#----------------------------------------------------------------------------------

filename = "soft_wt_dnn_train_loss_rhorho_Variant-All_nc_21"
x = np.arange(1,len(train_losses)+1)
plt.plot([0], marker='None',
           linestyle='None', label=r'Classification: wt')
plt.plot(x,train_losses, 'o', color = 'black', label='Training')
plt.plot(x,valid_losses, 'd', color = 'orange', label='Validation')
plt.legend()
plt.ylim([2.875, 2.925])
plt.xlabel('Number of epochs')
plt.xticks(x)
plt.ylabel('Loss')
#plt.title('Features list: Variant-All')
    
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

filename = "soft_wt_dnn_train_accs_rhorho_Variant-All_nc_21"
x = np.arange(1,len(train_accs)+1)
plt.plot(x,train_accs, 'o', label='training')
plt.plot(x,valid_accs, 'd', label='validation')
plt.legend()
#plt.ylim([0.0, 0.4])
plt.xlabel('Number of epochs')
plt.xticks(x)
plt.ylabel(r'Fraction $|\Delta_{class}|$ = 0')
#plt.title('Features list: Variant-All')
    
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

filename = "soft_wt_dnn_train_l1_deltas_rhorho_Variant-All_nc_21"
x = np.arange(1,len(train_l1_deltas)+1)
plt.plot(x,train_l1_deltas, 'o', label='training')
plt.plot(x,valid_l1_deltas, 'd', label='validation')
plt.legend()
#plt.ylim([0.010, 0.018])
plt.xlabel('Number of epochs')
plt.xticks(x)
plt.ylabel(r'$l_1$')
#plt.title('Features list: Variant-All')
    
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

filename = "soft_wt_dnn_train_l2_deltas_rhorho_Variant-All_nc_21"
x = np.arange(1,len(train_l2_deltas)+1)
plt.plot(x,train_l2_deltas, 'o', label='training')
plt.plot(x,valid_l2_deltas, 'd', label='validation')
plt.legend()
#plt.ylim([0.010, 0.025])
plt.xlabel('Number of epochs')
plt.xticks(x)
plt.ylabel(r'$l_2$')
#plt.title('Features list: Variant-All')
    
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

filename = "soft_wt_dnn_test_l2_deltas_rhorho_Variant-All_nc_21"
plt.plot(test_l2_deltas, 'o', label='testing')
plt.legend()
#plt.ylim([0.0, 0.005])
plt.xlabel('Count of updates')
plt.ylabel(r'$l_2$')
#plt.title('Features list: Variant-All')
    
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
