import sys
import os, errno
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
#


from scipy import optimize


pathIN  = "../temp_results/nn_rhorho_Variant-All_regr_c012s_hits_c1s_Unweighted_False_NO_NUM_CLASSES_21_SIZE_1000_LAYERS_6/monit_npy/"
pathOUT = "figures/"

train_losses    = np.load(pathIN+'train_losses_regr_c012s.npy')
valid_losses    = np.load(pathIN+'valid_losses_regr_c012ss.npy')


#----------------------------------------------------------------------------------

filename = "regr_c012s_dnn_train_loss_rhorho_Variant-All"
x = np.arange(1,len(train_losses)+1)
plt.plot([0], marker='None',
           linestyle='None', label=r'Regression: $C_0, C_1, C_2$')
plt.plot(x,train_losses, 'o', color = 'black', label='Training')
plt.plot(x,valid_losses, 'd', color = 'orange', label='Validation')
plt.legend()
plt.ylim([0, 0.016])
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
