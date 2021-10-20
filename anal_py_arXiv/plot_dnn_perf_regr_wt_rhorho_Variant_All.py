import sys
import os, errno
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from scipy import optimize


pathIN  = "../laptop_results_dropout=0/nn_rhorho_Variant-All_regr_weights_hits_c0s_Unweighted_False_NO_NUM_CLASSES_21_epochs25/monit_npy/"
pathOUT = "figures/"

train_losses    = np.load(pathIN+'train_losses_weights.npy')
valid_losses    = np.load(pathIN+'valid_losses_weights.npy')


#----------------------------------------------------------------------------------

filename = "regr_wt_dnn_train_loss_rhorho_Variant-All"
x = np.arange(1,len(train_losses)+1)
plt.plot([0], marker='None',
           linestyle='None', label=r'Regression: wt')
plt.plot(x,train_losses, 'o', color = 'black', label='Training')
plt.plot(x,valid_losses, 'd', color = 'orange', label='Validation')
plt.legend()
plt.ylim([0.0, 0.10])
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
