import sys
import os, errno
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from scipy import optimize


pathIN  = "../laptop_results_dropout=0/nn_rhorho_Variant-All_regr_argmaxs_hits_c0s_Unweighted_False_NO_NUM_CLASSES_0_epochs25/monit_npy/"
pathOUT = "figures/"

train_losses    = np.load(pathIN+'train_losses_regr_argmaxs.npy')
valid_losses    = np.load(pathIN+'valid_losses_regr_argmax.npy')
print train_losses
print valid_losses

#----------------------------------------------------------------------------------

filename = "regr_argmaxs_dnn_train_loss_rhorho_Variant-All"
x = np.arange(1,len(train_losses)+1)
plt.plot([0], marker='None',
           linestyle='None', label=r'Regression: $\alpha^{CP}_{max}$')
plt.plot(x,train_losses, 'o', color = 'black', label='Training')
plt.plot(x,valid_losses, 'd', color = 'orange', label='Validation')
plt.legend()
plt.ylim([1.0, 2.0])
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
