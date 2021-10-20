import sys
import os, errno
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from scipy import optimize

pathOUT = "figures/"

pathIN  = "../laptop_results_dropout=0/nn_rhorho_Variant-All_soft_c012s_hits_c0s_Unweighted_False_NO_NUM_CLASSES_21_epochs25/monit_npy/"
train_losses_c0         = np.load(pathIN+'train_losses_soft_c012s.npy')
validation_losses_c0    = np.load(pathIN+'valid_losses_soft_c012s.npy')

pathIN  = "../laptop_results_dropout=0/nn_rhorho_Variant-All_soft_c012s_hits_c1s_Unweighted_False_NO_NUM_CLASSES_21_epochs25/monit_npy/"
train_losses_c1         = np.load(pathIN+'train_losses_soft_c012s.npy')
validation_losses_c1    = np.load(pathIN+'valid_losses_soft_c012s.npy')

pathIN  = "../laptop_results_dropout=0/nn_rhorho_Variant-All_soft_c012s_hits_c2s_Unweighted_False_NO_NUM_CLASSES_21_epochs25/monit_npy/"
train_losses_c2         = np.load(pathIN+'train_losses_soft_c012s.npy')
validation_losses_c2    = np.load(pathIN+'valid_losses_soft_c012s.npy')

#----------------------------------------------------------------------------------

filename = "soft_c012s_dnn_train_loss_rhorho_Variant-All"
x = np.arange(1,len(train_losses_c0)+1)
plt.plot([0], marker='None',
           linestyle='None', label=r'Classification: $C_0, C_1, C_2$')
plt.plot(x,train_losses_c0, 'o', color = 'black', label=r'Training $C_0$')
plt.plot(x,train_losses_c1, 'd', color = 'black', label=r'Training $C_1$')
plt.plot(x,train_losses_c2, 'x', color = 'black', label=r'Training $C_2$')
plt.plot(x,validation_losses_c0, 'o', color = 'orange', label=r'Validation $C_0$')
plt.plot(x,validation_losses_c1, 'd', color = 'orange', label=r'Validation $C_1$')
plt.plot(x,validation_losses_c2, 'x', color = 'orange', label=r'Validation $C_2$')
plt.legend()
plt.ylim([0.8, 3.0])
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
    
#----------------------------------------------------------------------------------
plt.clf()
#----------------------------------------------------------------------------------
