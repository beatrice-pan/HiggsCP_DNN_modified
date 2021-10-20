import os, errno
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from src_py.metrics_utils import  calculate_deltas_signed

pathIN  = "../temp_results/nn_rhorho_Variant-All_regr_weights_hits_c0s_Unweighted_False_NO_NUM_CLASSES_21_SIZE_100_LAYERS_6_ZNOISE_0.0/monit_npy/"
# pathOUT = "figures/"
pathOUT = "nn_rhorho_Variant-All_regr_weights_hits_c0s_Unweighted_False_NO_NUM_CLASSES_21_SIZE_100_LAYERS_6_ZNOISE_0.0/"

calc_w  = np.load(pathIN+'test_regr_calc_weights.npy')
preds_w = np.load(pathIN+'test_regr_preds_weights.npy')

k2PI = 2*np.pi
#----------------------------------------------------------------------------------
                           
i = 1
filename = "regr_wt_calc_preds_rhorho_Variant-All_nc_21_event_1"
x = np.arange(1,22)
plt.plot(x,calc_w[i], 'o', label='Generated')
plt.plot(x,preds_w[i], 'd', label='Regression: wt')
plt.legend()
#plt.ylim([0.0, 0.125])
plt.xlabel('Class index [idx]')
plt.xticks(x)
plt.ylabel(r'$wt$')
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
                           
i = 10
filename = "regr_wt_calc_preds_rhorho_Variant-All_nc_21_event_10"
x = np.arange(1,22)
plt.plot(x,calc_w[i], 'o', label='Generated')
plt.plot(x,preds_w[i], 'd', label='Regression: wt')
plt.legend()
#plt.ylim([0.0, 0.125])
plt.xlabel('Class index [idx]')
plt.xticks(x)
plt.ylabel(r'$wt$')
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
                           
i = 1000
filename = "regr_wt_calc_preds_rhorho_Variant-All_nc_21_event_1000"
x = np.arange(1,22)
plt.plot(x,calc_w[i], 'o', label='Generated')
plt.plot(x,preds_w[i], 'd', label='Regression: wt')
plt.legend()
#plt.ylim([0.0, 0.1])
plt.xlabel('Class index [idx]')
plt.xticks(x)
plt.ylabel(r'$wt$')
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
                           
i = 2000
filename = "regr_wt_calc_preds_rhorho_Variant-All_nc_21_event_2000"
x = np.arange(1,22)
plt.plot(x,calc_w[i], 'o', label='Generated')
plt.plot(x,preds_w[i], 'd', label='Regression: wt')
plt.legend()
#plt.ylim([0.0, 0.1])
plt.xlabel('Class index [idx]')
plt.xticks(x)
plt.ylabel(r'$wt$')
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
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------

pathIN  = "../temp_results/nn_rhorho_Variant-All_regr_weights_hits_c0s_Unweighted_False_NO_NUM_CLASSES_21_SIZE_100_LAYERS_6_ZNOISE_0.0/monit_npy/"
# pathOUT = "figures/"
pathOUT = "nn_rhorho_Variant-All_regr_weights_hits_c0s_Unweighted_False_NO_NUM_CLASSES_21_SIZE_100_LAYERS_6_ZNOISE_0.0/"

calc_w_nc21  = np.load(pathIN+'test_regr_calc_weights.npy')
preds_w_nc21 = np.load(pathIN+'test_regr_preds_weights.npy')
delt_argmax_nc21 =  calculate_deltas_signed(np.argmax(preds_w_nc21[:], axis=1), np.argmax(calc_w_nc21[:], axis=1), 21)      

filename = "regr_wt_delt_argmax_rhorho_Variant-All_nc_21"

# plt.figure(figsize=(15,10))
sns.barplot(x=pd.DataFrame(delt_argmax_nc21)[0].value_counts().sort_index().index, y=pd.DataFrame(delt_argmax_nc21)[0].value_counts().sort_index(), color='salmon')
##### May have problem at delta = 0:
# plt.hist(delt_argmax_nc51, histtype='step', bins=51, color='black')

plt.xticks(rotation='vertical')
plt.xlabel(r'$\alpha^{CP}_{max}: \Delta_{class} [idx]$')
plt.ylabel('')
#plt.title('Features list: Variant-All')

ax = plt.gca()
nc21=21
mean = np.mean(delt_argmax_nc21, dtype=np.float64)
std  = np.std(delt_argmax_nc21, dtype=np.float64)
meanerr = stats.sem(delt_argmax_nc21)
meanrad = np.mean(delt_argmax_nc21, dtype=np.float64) * k2PI/nc21
stdrad  = np.std(delt_argmax_nc21, dtype=np.float64) * k2PI/nc21
meanerrrad = stats.sem(delt_argmax_nc21) * k2PI/nc21


table_vals=[[r"Regression: $wt$"],
            [" "],
            [r"mean = {:0.3f} $\pm$ {:1.3f}[idx] ".format(mean, meanerr)],
            ["std = {:1.3f} [idx]".format(std)],
            [" "],
            [r"mean = {:0.3f} $\pm$ {:1.3f}[rad]".format(meanrad, meanerrrad)],
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

#----------------------------------------------------------------------------------
