import os, errno
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from src_py.metrics_utils import  calculate_deltas_signed_pi



pathIN  = "../laptop_results_dropout=0/nn_rhorho_Variant-All_regr_argmaxs_hits_c0s_Unweighted_False_NO_NUM_CLASSES_0/monit_npy/"
pathOUT = "figures/"

calc_argmaxs  = np.load(pathIN+'test_regr_calc_argmaxs.npy')
preds_argmaxs = np.load(pathIN+'test_regr_preds_argmaxs.npy')


print calc_argmaxs
print preds_argmaxs
print calc_argmaxs - preds_argmaxs

delt_argmaxs = calc_argmaxs - preds_argmaxs
#poprawic!!! delta < 3.1415
delt_argmaxs = calculate_deltas_signed_pi(calc_argmaxs, preds_argmaxs)

k2PI = 2 * np.pi
#calc_argmaxs= calc_argmaxs/k2PI
#print calc_argmaxs

#----------------------------------------------------------------------------------
filename = "regr_argmaxs_calc_preds_argmax_rhorho_Variant-All"

plt.hist(calc_argmaxs, histtype='step', bins=50,  color = 'black', linestyle='--', label="Generated")
plt.hist(preds_argmaxs, histtype='step', bins=50, color = 'red', label=r"Regression: $\alpha^{CP}_{max}$")
plt.xlim([0, k2PI])
plt.ylim([0, 1700])
plt.xlabel(r'$\alpha^{CP}_{max}$[rad]')
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
filename = "regr_argmaxs_delt_argmax_rhorho_Variant-All"

plt.hist(delt_argmaxs, histtype='step', bins=50,  color = 'black')
plt.xlim([-3.2, 3.2])
plt.xlabel(r'$\Delta \alpha^{CP}_{max}$ [rad]')
#plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_argmaxs)
std  = np.std(delt_argmaxs)

table_vals=[[r"Regression: $\alpha^{CP}_{max}$"],
            [" "],
            ["mean = {:0.3f} [rad]".format(mean)],
            ["std = {:1.3f} [rad]".format(std)]
            ]

table = plt.table(cellText=table_vals,
                  colWidths = [0.30],
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
#----------------------------------------------------------------------------------
