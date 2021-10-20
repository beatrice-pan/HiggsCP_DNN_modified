import os, errno
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scipy import optimize, stats

pathIN  = "../temp_results/nn_rhorho_Variant-All_soft_c012s_hits_c0s_Unweighted_False_NO_NUM_CLASSES_51_SIZE_1000_LAYERS_6/monit_npy/"
calc_hits_c0s   = np.load(pathIN+'train_soft_calc_hits_c012s.npy')
preds_hits_c0s  = np.load(pathIN+'train_soft_preds_hits_c012s.npy')

pathIN  = "../temp_results/nn_rhorho_Variant-All_soft_c012s_hits_c1s_Unweighted_False_NO_NUM_CLASSES_51_SIZE_1000_LAYERS_6/monit_npy/"
calc_hits_c1s   = np.load(pathIN+'train_soft_calc_hits_c012s.npy')
preds_hits_c1s  = np.load(pathIN+'train_soft_preds_hits_c012s.npy')

pathIN  = "../temp_results/nn_rhorho_Variant-All_soft_c012s_hits_c2s_Unweighted_False_NO_NUM_CLASSES_51_SIZE_1000_LAYERS_6/monit_npy/"
calc_hits_c2s   = np.load(pathIN+'train_soft_calc_hits_c012s.npy')
preds_hits_c2s  = np.load(pathIN+'train_soft_preds_hits_c012s.npy')

pathOUT = "figures/"


data_len = calc_hits_c0s.shape[0]
preds_c0s = np.zeros((data_len, 1))
calc_c0s = np.zeros((data_len, 1))
preds_c1s = np.zeros((data_len, 1))
calc_c1s = np.zeros((data_len, 1))
preds_c2s = np.zeros((data_len, 1))
calc_c2s = np.zeros((data_len, 1))

for i in range(data_len):
    preds_c0s[i] = np.argmax(preds_hits_c0s[i])
    calc_c0s[i] = np.argmax(calc_hits_c0s[i])
    preds_c1s[i] = np.argmax(preds_hits_c1s[i])
    calc_c1s[i] = np.argmax(calc_hits_c1s[i])
    preds_c2s[i] = np.argmax(preds_hits_c2s[i])
    calc_c2s[i] = np.argmax(calc_hits_c2s[i])
#    print i, " ",  preds_hits_c0s[i]
#    print i, " ",  preds_hits_c1s[i]
#    print preds_c0s[i], " ", preds_c1s[i]

    
print calc_hits_c0s[0]
print calc_c0s[0]
print preds_hits_c0s[0]
print preds_c0s[0]
    
print calc_hits_c1s[0]
print calc_c1s[0]
print preds_hits_c1s[0]
print preds_c1s[0]
    
print calc_hits_c2s[0]
print calc_c2s[0]
print preds_hits_c2s[0]
print preds_c2s[0]


delt_c0s =  np.argmax(preds_hits_c0s[:], axis=1) -  np.argmax(calc_hits_c0s[:], axis=1)     
delt_c1s =  np.argmax(preds_hits_c1s[:], axis=1) -  np.argmax(calc_hits_c1s[:], axis=1)     
delt_c2s =  np.argmax(preds_hits_c2s[:], axis=1) -  np.argmax(calc_hits_c2s[:], axis=1)     


k2PI= 2* np.pi
#----------------------------------------------------------------------------------
filename = "soft_c012s_c0s_rhorho_Variant-All_nc51"

plt.hist(calc_c0s, histtype='step', bins=51,  color = 'black', label = "Generated")
plt.hist(preds_c0s, histtype='step', bins=51,  color = 'red', label = r"Classification: $C_0, C_1, C_2$")
plt.xlim([0, 50.0])
plt.ylim([0, 8000])
plt.xlabel(r'$C_0$: Class index [idx]')
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
filename = "soft_c012s_delt_c0s_rhorho_Variant-All_nc51"

plt.hist(delt_c0s, histtype='step', bins=51,  color = 'black')
plt.xlabel(r'$C_0: \Delta_{class}$ [idx]')
#plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_c0s)
std  = np.std(delt_c0s)
meanerr = stats.sem(delt_c0s)
meanC = np.mean(delt_c0s) * 2.0/51.0
stdC  = np.std(delt_c0s) * 2.0/51.0
meanerrC = meanerr * 2.0/51.0

table_vals=[[r"mean", r"= {:0.3f}$\pm$ {:1.3f} [idx]".format(mean, meanerr)],
            ["std", "= {:1.3f} [idx]".format(std)],
            ["", ""],
            [r"mean", r"= {:0.3f}$\pm$ {:1.3f}".format(meanC, meanerrC)],
            ["std", "= {:1.3f}".format(stdC)]
            ]

table = plt.table(cellText=table_vals,
                  colWidths = [0.10, 0.30],
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
filename = "soft_c012s_c1s_rhorho_Variant-All_nc51"

plt.hist(calc_c1s, histtype='step', bins=51,  color = 'black', label = "Generated")
plt.hist(preds_c1s, histtype='step', bins=51,  color = 'red', label = r"Classification: $C_0, C_1, C_2$")
plt.xlim([0, 50.0])
plt.ylim([0, 5000])
plt.xlabel(r'$C_1$: Class index [idx]')
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
filename = "soft_c012s_delt_c1s_rhorho_Variant-All_nc51"

plt.hist(delt_c1s, histtype='step', bins=51,  color = 'black')
plt.xlabel(r'$C_1: \Delta_{class}$ [idx]')
#plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_c1s)
std  = np.std(delt_c1s)
meanerr = stats.sem(delt_c1s)
meanC = np.mean(delt_c1s) * 2.0/51.0
stdC  = np.std(delt_c1s) * 2.0/51.0
meanerrC = meanerr * 2.0/51.0

table_vals=[[r"mean", r"= {:0.3f}$\pm$ {:1.3f} [idx]".format(mean, meanerr)],
            ["std", "= {:1.3f} [idx]".format(std)],
            ["", ""],
            ["mean", "= {:0.3f}$\pm$ {:1.3f}".format(meanC, meanerrC)],
            ["std", "= {:1.3f}".format(stdC)]
            ]

table = plt.table(cellText=table_vals,
                  colWidths = [0.10, 0.30],
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
filename = "soft_c012s_c2s_rhorho_Variant-All_nc51"

plt.hist(calc_c2s, histtype='step', bins=51,  color = 'black', label = "Generated")
plt.hist(preds_c2s, histtype='step', bins=51,  color = 'red', label = r"Classification: $C_0, C_1, C_2$")
plt.xlim([0, 50.0])
plt.ylim([0, 5000])
plt.xlabel(r'$C_2$: Class index [idx]')
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
filename = "soft_c012s_delt_c2s_rhorho_Variant-All_nc51"

plt.hist(delt_c2s, histtype='step', bins=51,  color = 'black')
plt.xlabel(r'$C_2: \Delta_{class}$ [idx]')
#plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_c2s)
std  = np.std(delt_c2s)
meanerr = stats.sem(delt_c2s)
meanC = np.mean(delt_c2s) * 2.0/51.0
stdC  = np.std(delt_c2s) * 2.0/51.0
meanerrC = meanerr * 2.0/51.0

table_vals=[[r"mean", r"= {:0.3f}$\pm$ {:1.3f} [idx]".format(mean, meanerr)],
            ["std", "= {:1.3f} [idx]".format(std)],
            ["", ""],
            [r"mean", r"= {:0.3f}$\pm$ {:1.3f}".format(meanC, meanerrC)],
            ["std", "= {:1.3f}".format(stdC)]
            ]

table = plt.table(cellText=table_vals,
                  colWidths = [0.10, 0.30],
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
