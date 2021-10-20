import os, errno
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src_py.metrics_utils import  calculate_deltas_signed
from scipy import optimize, stats

pathIN  = "../temp_results/nn_rhorho_Variant-All_soft_argmaxs_hits_c0s_Unweighted_False_NO_NUM_CLASSES_21_SIZE_1000_LAYERS_6/monit_npy/"
pathOUT = "figures/"

calc_hits_argmaxs  = np.load(pathIN+'valid_soft_calc_hits_argmaxs.npy')
preds_hits_argmaxs  = np.load(pathIN+'valid_soft_preds_hits_argmaxs.npy')

print calc_hits_argmaxs[0]
print preds_hits_argmaxs[0]

data_len = calc_hits_argmaxs.shape[0]
preds_argmaxs = np.zeros((data_len, 1))
calc_argmaxs = np.zeros((data_len, 1))

for i in range(data_len):
    preds_argmaxs[i] = np.argmax(preds_hits_argmaxs[i])
    calc_argmaxs[i] = np.argmax(calc_hits_argmaxs[i])


delt_argmaxs =  calculate_deltas_signed(np.argmax(preds_hits_argmaxs[:], axis=1), np.argmax(calc_hits_argmaxs[:], axis=1), 21)      


k2PI= 2* np.pi
#----------------------------------------------------------------------------------
filename = "soft_argmaxs_rhorho_Variant-All_nc21"

plt.hist(calc_argmaxs, histtype='step', bins=21,  color = 'black', linestyle='--', label = "Generated")
plt.hist(preds_argmaxs, histtype='step', bins=21,  color = 'red', label = r"Classification: $\alpha^{CP}_{max}$")
#plt.xlim([0, 50])
plt.ylim([0,5000])
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
filename = "soft_argmaxs_delt_rhorho_Variant-All_nc21"

plt.hist(delt_argmaxs, histtype='step', bins=21,  color = 'black')
plt.xlabel(r'$\Delta_{class}$ [idx]')
#plt.title('Features list: Variant-All')
plt.xlim(-5,5)

ax = plt.gca()
mean = np.mean(delt_argmaxs)
std  = np.std(delt_argmaxs)
meanerr = stats.sem(delt_argmaxs) 
meanrad = np.mean(delt_argmaxs) * k2PI/21.0
stdrad  = np.std(delt_argmaxs) * k2PI/21.0
meanerrrad = stats.sem(delt_argmaxs)* k2PI/21.0 

table_vals=[[r'Classification:$\alpha^{CP}_{max}$'],
            [" "],
            [r"mean = {:0.3f}$\pm$ {:1.3f} [idx]".format(mean, meanerr)],
            ["std = {:1.3f} [idx]".format(std)],
            [" "],
            [r"mean = {:0.3f}$\pm$ {:1.3f} [rad]".format(meanrad, meanerrrad)],
            ["std = {:1.3f} [rad]".format(stdrad)]
            ]

table = plt.table(cellText=table_vals,
                  colWidths = [0.40],
                  cellLoc="left",
                  loc='upper right')
table.set_fontsize(14)

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
