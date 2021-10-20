import sys
import os, errno
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from scipy import optimize


pathIN  = "../laptop_results/nn_rhorho_Variant-All_regr_popts_Unweighted_False_NO_NUM_CLASSES_0/monit_npy/"
pathOUT = "figures/"

train_losses    = np.load(pathIN+'train_losses.npy')


#----------------------------------------------------------------------------------

filename = "dnn_train_loss_rhorho_Variant-All_regr"
x = np.arange(1,len(train_losses)+1)
plt.plot(x,train_losses, 'o', label='training')
plt.legend()
#plt.ylim([2.9, 3.0])
plt.xlabel('Number of epochs')
plt.xticks(x)
plt.ylabel('Loss')
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
    
#----------------------------------------------------------------------------------
plt.clf()
#----------------------------------------------------------------------------------
