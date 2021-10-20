import os, errno
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src_py.metrics_utils import  calculate_deltas_signed

pathIN  = "../temp_results/nn_rhorho_Variant-All_soft_argmaxs_hits_c0s_Unweighted_False_NO_NUM_CLASSES_51_SIZE_1000_LAYERS_6/monit_npy/"
pathOUT = "figures/"

calc_hits_argmaxs  = np.load(pathIN+'train_soft_calc_hits_argmaxs.npy')
preds_hits_argmaxs  = np.load(pathIN+'train_soft_preds_hits_argmaxs.npy')

print calc_hits_argmaxs[0]
print preds_hits_argmaxs[0]

data_len = calc_hits_argmaxs.shape[0]
preds_argmaxs = np.zeros((data_len, 1))
calc_argmaxs = np.zeros((data_len, 1))

for i in range(data_len):
    preds_argmaxs[i] = np.argmax(preds_hits_argmaxs[i]) #np.sum(preds_hits_argmaxs[i]*np.arange(51)) #np.argmax(preds_hits_argmaxs[i])
    calc_argmaxs[i] = np.argmax(calc_hits_argmaxs[i])


delt_argmaxs =  calculate_deltas_signed(np.argmax(preds_hits_argmaxs[:], axis=1), np.argmax(calc_hits_argmaxs[:], axis=1), 51)      

k2PI= 2* np.pi
#----------------------------------------------------------------------------------
filename = "soft_argmaxs_rhorho_Variant-All"

plt.hist(calc_argmaxs, histtype='step', bins=51,  color = 'black', linestyle='--', label = "Generated")
plt.hist(preds_argmaxs, histtype='step', bins=51,  color = 'red', label = "Classification: \alpha^{CP}_{max}")
#plt.step(np.arange(0,51),np.sum(calc_hits_argmaxs[1:4], axis = 0),  color = 'black', linestyle='--', label = "Generated")
#plt.step(np.arange(0,51),np.sum(preds_hits_argmaxs[1:4],axis = 0),  color = 'red', label = r"Classification: $\alpha^{CP}_{max}$")

#plt.xlim([0, 50])
plt.ylim([0,80000])
plt.xlabel('Class index [idx]')
#plt.title('Features list: Variant-All')
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
filename = "soft_argmaxs_delt_rhorho_Variant-All"

plt.hist(delt_argmaxs, histtype='step', bins=51,  color = 'black')
plt.xlabel(r'$\Delta_{class}$')
#plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_argmaxs)
std  = np.std(delt_argmaxs)
meanrad = np.mean(delt_argmaxs) * k2PI/51.0
stdrad  = np.std(delt_argmaxs) * k2PI/51.0

table_vals=[["mean", "= {:0.3f} [idx]".format(mean)],
            ["std", "= {:1.3f} [idx]".format(std)],
            ["", ""],
            ["mean", "= {:0.3f} [rad]".format(meanrad)],
            ["std", "= {:1.3f} [rad]".format(stdrad)]
            ]

table = plt.table(cellText=table_vals,
                  colWidths = [0.10, 0.22],
                  cellLoc="left",
                  loc='upper right')
table.set_fontsize(12)

for key, cell in table.get_celld().items():
    cell.set_linewidth(0)

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
