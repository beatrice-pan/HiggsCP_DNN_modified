import sys
import os, errno
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
#
#plt.ioff()

from scipy import optimize, stats
from src_py.metrics_utils import  calculate_deltas_signed

pathIN  = "../temp_results/nn_rhorho_Variant-All_soft_weights_hits_c0s_Unweighted_False_NO_NUM_CLASSES_21_SIZE_1000_LAYERS_6/monit_npy/"
pathOUT = "figures/"

calc_w  = np.load(pathIN+'softmax_calc_w.npy')
preds_w = np.load(pathIN+'softmax_preds_w.npy')

k2PI = 6.28
#----------------------------------------------------------------------------------
                           
i = 1
filename = "soft_wt_calc_preds_rhorho_Variant-All_nc_21_event_1"
x = np.arange(1,22)
plt.plot(x,calc_w[i]/sum(calc_w[i]), label='Generated')
plt.step(x,preds_w[i], label='Classification: wt')
plt.legend()
plt.ylim([0.0, 0.125])
plt.xlabel('Class index')
plt.xticks(x)
plt.ylabel(r'$wt^{norm}$')
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
filename = "soft_wt_calc_preds_rhorho_Variant-All_nc_21_event_10"
x = np.arange(1,22)
plt.plot(x,calc_w[i]/sum(calc_w[i]), 'o', label='Generated')
plt.plot(x,preds_w[i], 'd', label='predicted')
plt.legend()
plt.ylim([0.0, 0.125])
plt.xlabel('Class index')
plt.xticks(x)
plt.ylabel(r'$wt^{norm}$')
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
filename = "soft_wt_calc_preds_rhorho_Variant-All_nc_21_event_1000"
x = np.arange(1,22)
x2 = np.arange(0,22)
plt.plot(x,calc_w[i]/sum(calc_w[i]), label='Generated')
plt.step(x2+0.5,np.append(preds_w[i][0],preds_w[i]), label='Classification: wt')
plt.legend()
plt.ylim([0.0, 0.1])
plt.xlabel('Class index')
plt.xticks(x)
plt.ylabel(r'$wt^{norm}$')
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
filename = "soft_wt_calc_preds_rhorho_Variant-All_nc_21_event_2000"
x = np.arange(1,22)
x2 = np.arange(0,22)
plt.plot(x,calc_w[i]/sum(calc_w[i]), label='Generated')
plt.step(x2+0.5,np.append(preds_w[i][0],preds_w[i]), label='Classification: wt')
plt.legend()
plt.ylim([0.0, 0.1])
plt.xlabel('Class index')
plt.xticks(x)
plt.ylabel(r'$wt^{norm}$')
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

pathIN  = "../temp_results/nn_rhorho_Variant-All_soft_weights_hits_c0s_Unweighted_False_NO_NUM_CLASSES_5_SIZE_1000_LAYERS_6/monit_npy/"
pathOUT = "figures/"

calc_w_nc5  = np.load(pathIN+'softmax_calc_w.npy')
preds_w_nc5 = np.load(pathIN+'softmax_preds_w.npy')
delt_argmax_nc5 =  calculate_deltas_signed(np.argmax(preds_w_nc5[:], axis=1), np.argmax(calc_w_nc5[:], axis=1), 5)      

filename = "soft_wt_delt_argmax_rhorho_Variant-All_nc_5"
plt.hist(delt_argmax_nc5, histtype='step', color='black')
plt.xlabel(r'$\Delta_{class}$')
plt.ylabel('Entries')
#plt.title('Features list: Variant-All')

ax = plt.gca()
nc5=5
mean = np.mean(delt_argmax_nc5) 
std  = np.std(delt_argmax_nc5) 
meanerr = stats.sem(delt_argmax_nc5)
meanrad = np.mean(delt_argmax_nc5, dtype=np.float64) * 6.28/nc5
stdrad  = np.std(delt_argmax_nc5, dtype=np.float64) * 6.28/nc5
meanerrrad = stats.sem(delt_argmax_nc5) * 6.28/nc5

table_vals=[[r"Classification: $wt$"],
            [" "],
            [r"mean = {:0.3f} $\pm$ {:1.3f}[idx] ".format(mean, meanerr)],
            ["std = {:1.3f} [idx]".format(std)],
            [" "],
            [r"mean = {:0.3f} $\pm$ {:1.3f}[rad]".format(meanrad, meanerrrad)],
            ["std = {:1.3f} [rad]".format(stdrad)]
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
#----------------------------------------------------------------------------------

pathIN  = "../temp_results/nn_rhorho_Variant-All_soft_weights_hits_c0s_Unweighted_False_NO_NUM_CLASSES_11_SIZE_1000_LAYERS_6/monit_npy/"
pathOUT = "figures/"

calc_w_nc11  = np.load(pathIN+'softmax_calc_w.npy')
preds_w_nc11 = np.load(pathIN+'softmax_preds_w.npy')
preds_argmax_nc11 =  np.argmax(preds_w_nc11[:], axis=1) * k2PI/11.0    
calc_argmax_nc11 =  np.argmax(calc_w_nc11[:], axis=1) * k2PI/11.0    

delt_argmax_nc11 =  calculate_deltas_signed(np.argmax(preds_w_nc11[:], axis=1), np.argmax(calc_w_nc11[:], axis=1), 11)      

filename = "soft_wt_calc_argmax_rhorho_Variant-All_nc_11"
plt.hist(calc_argmax_nc11, histtype='step', color = "black", bins=100, label = "Generated")
plt.hist(preds_argmax_nc11, histtype='step', color = "red", bins=100, label = "Classification: wt")
#plt.ylim([0, 800])
plt.ylabel('Entries')
plt.xlabel(r'$\alpha^{CP}_{max}$')
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

filename = "soft_wt_delt_argmax_rhorho_Variant-All_nc_11"
plt.hist(delt_argmax_nc11, histtype='step', bins=11, color='black')
plt.ylabel('Entries')
plt.xlabel(r'$\Delta_{class}$')
#plt.title('Features list: Variant-All')

ax = plt.gca()
nc11=11
mean = np.mean(delt_argmax_nc11, dtype=np.float64)
std  = np.std(delt_argmax_nc11, dtype=np.float64)
meanerr = stats.sem(delt_argmax_nc11)
meanrad = np.mean(delt_argmax_nc11, dtype=np.float64) * 6.28/11.0
stdrad  = np.std(delt_argmax_nc11, dtype=np.float64) * 6.28/11.0
meanerrrad = stats.sem(delt_argmax_nc11) * 6.28/11

table_vals=[[r"mean", r"= {:0.3f} $\pm$ {:1.3f}[idx] ".format(mean, meanerr)],
            ["std", "= {:1.3f} [idx]".format(std)],
            ["", ""],
            ["mean", r"= {:0.3f} $\pm$ {:1.3f}[rad]".format(meanrad, meanerrrad)],
            ["std", "= {:1.3f} [rad]".format(stdrad)]
            ]

table = plt.table(cellText=table_vals,
                  colWidths = [0.08, 0.28],
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

pathIN  = "../temp_results/nn_rhorho_Variant-All_soft_weights_hits_c0s_Unweighted_False_NO_NUM_CLASSES_21_SIZE_1000_LAYERS_6/monit_npy/"
pathOUT = "figures/"

calc_w_nc21  = np.load(pathIN+'softmax_calc_w.npy')
preds_w_nc21 = np.load(pathIN+'softmax_preds_w.npy')
delt_argmax_nc21 =  calculate_deltas_signed(np.argmax(preds_w_nc21[:], axis=1), np.argmax(calc_w_nc21[:], axis=1), 21)      

filename = "soft_wt_delt_argmax_rhorho_Variant-All_nc_21"
plt.hist(delt_argmax_nc21, histtype='step', bins=21, color='black')
plt.xlabel(r'$\Delta_{class}$ [idx]')
#plt.title('Features list: Variant-All')
plt.xlim(-5,5)
ax = plt.gca()
nc21=21
mean = np.mean(delt_argmax_nc21, dtype=np.float64)
std  = np.std(delt_argmax_nc21, dtype=np.float64)
meanerr = stats.sem(delt_argmax_nc21)
meanrad = np.mean(delt_argmax_nc21, dtype=np.float64) * 6.28/21.0
stdrad  = np.std(delt_argmax_nc21, dtype=np.float64) * 6.28/21.0
meanerrrad = stats.sem(delt_argmax_nc21) * 6.28/21


table_vals=[[r"Classification: $wt$"],
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
#----------------------------------------------------------------------------------

pathIN  = "../temp_results/nn_rhorho_Variant-All_soft_weights_hits_c0s_Unweighted_False_NO_NUM_CLASSES_25_SIZE_1000_LAYERS_6/monit_npy/"
pathOUT = "figures/"

calc_w_nc25  = np.load(pathIN+'softmax_calc_w.npy')
preds_w_nc25 = np.load(pathIN+'softmax_preds_w.npy')
delt_argmax_nc25 =  calculate_deltas_signed(np.argmax(preds_w_nc25[:], axis=1), np.argmax(calc_w_nc25[:], axis=1), 25)      

filename = "soft_wt_delt_argmax_rhorho_Variant-All_nc_25"
plt.hist(delt_argmax_nc25, histtype='step', bins=50, color='black')
plt.xlabel(r'$\Delta_{class}$ [idx]')
plt.ylabel('Entries')
#plt.title('Features list: Variant-All')

ax = plt.gca()
nc25=25
mean = np.mean(delt_argmax_nc25, dtype=np.float64)
std  = np.std(delt_argmax_nc25, dtype=np.float64)
meanerr = stats.sem(delt_argmax_nc25)
meanrad = np.mean(delt_argmax_nc25, dtype=np.float64) * 6.28/25.0
stdrad  = np.std(delt_argmax_nc25, dtype=np.float64) * 6.28/25.0
meanerrrad = stats.sem(delt_argmax_nc25) * 6.28/25

table_vals=[[r"mean", r"= {:0.3f} $\pm$ {:1.3f}[idx] ".format(mean, meanerr)],
            ["std", "= {:1.3f} [idx]".format(std)],
            ["", ""],
            ["mean", r"= {:0.3f} $\pm$ {:1.3f}[rad]".format(meanrad, meanerrrad)],
            ["std", "= {:1.3f} [rad]".format(stdrad)]
            ]

table = plt.table(cellText=table_vals,
                  colWidths = [0.08, 0.28],
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
#----------------------------------------------------------------------------------

pathIN  = "../temp_results/nn_rhorho_Variant-All_soft_weights_hits_c0s_Unweighted_False_NO_NUM_CLASSES_51_SIZE_1000_LAYERS_6/monit_npy/"
pathOUT = "figures/"

calc_w_nc51  = np.load(pathIN+'softmax_calc_w.npy')
preds_w_nc51 = np.load(pathIN+'softmax_preds_w.npy')
delt_argmax_nc51 =  calculate_deltas_signed(np.argmax(preds_w_nc51[:], axis=1), np.argmax(calc_w_nc51[:], axis=1), 51)      

filename = "soft_wt_delt_argmax_rhorho_Variant-All_nc_51"
plt.hist(delt_argmax_nc51, histtype='step', bins=51, color='black')
plt.xlabel(r'$\Delta_{class}$ [idx]')
plt.ylabel('Entries')
#plt.title('Features list: Variant-All')
plt.xlim(-8,8)
ax = plt.gca()
nc51=51
mean = np.mean(delt_argmax_nc51, dtype=np.float64)
std  = np.std(delt_argmax_nc51, dtype=np.float64)
meanerr = stats.sem(delt_argmax_nc51)
meanrad = np.mean(delt_argmax_nc51, dtype=np.float64) * 6.28/51.0
stdrad  = np.std(delt_argmax_nc51, dtype=np.float64) * 6.28/51.0
meanerrrad = stats.sem(delt_argmax_nc51) * 6.28/51


table_vals=[[r"Classification: $wt$"],
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
#----------------------------------------------------------------------------------

'''pathIN  = "../temp_results/nn_rhorho_Variant-All_soft_weights_hits_c0s_Unweighted_False_NO_NUM_CLASSES_101/monit_npy/"
pathOUT = "figures/"

calc_w_nc101  = np.load(pathIN+'softmax_calc_w.npy')
preds_w_nc101 = np.load(pathIN+'softmax_preds_w.npy')
delt_argmax_nc101 = np.argmax(calc_w_nc101[:], axis=1) - np.argmax(preds_w_nc101[:], axis=1)
for i in range (len(delt_argmax_nc101)):
    if  delt_argmax_nc101[i] > 101.0/2.0 :
        delt_argmax_nc101[i] = 101.0 -  delt_argmax_nc101[i]
    if  delt_argmax_nc101[i] < - 101.0/2.0 :
        delt_argmax_nc101[i] = - 101.0 -  delt_argmax_nc101[i]

filename = "soft_wt_delt_argmax_rhorho_Variant-All_nc_101"
plt.hist(delt_argmax_nc101, histtype='step', bins=101, color='black')
plt.xlabel(r'$\Delta_{class}$')
plt.ylabel('Entries')
#plt.title('Features list: Variant-All')

ax = plt.gca()
nc101=101
mean = np.mean(delt_argmax_nc101, dtype=np.float64)
std  = np.std(delt_argmax_nc101, dtype=np.float64)
meanerr = stats.sem(delt_argmax_nc101)
meanrad = np.mean(delt_argmax_nc101, dtype=np.float64) * 6.28/101.0
stdrad  = np.std(delt_argmax_nc101, dtype=np.float64) * 6.28/101.0
meanerrrad = stats.sem(delt_argmax_nc101) * 6.28/101

table_vals=[[r"mean", r"= {:0.3f} $\pm$ {:1.3f}[idx] ".format(mean, meanerr)],
            ["std", "= {:1.3f} [idx]".format(std)],
            ["", ""],
            ["mean", r"= {:0.3f} $\pm$ {:1.3f}[rad]".format(meanrad, meanerrrad)],
            ["std", "= {:1.3f} [rad]".format(stdrad)]
            ]

table = plt.table(cellText=table_vals,
                  colWidths = [0.08, 0.28],
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
'''
#----------------------------------------------------------------------------------
classes = np.linspace(0, 2, 10) * np.pi
print classes
classes = np.linspace(0, 2, 11) * np.pi
print classes
