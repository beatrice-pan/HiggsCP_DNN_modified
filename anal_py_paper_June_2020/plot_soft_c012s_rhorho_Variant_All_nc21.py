import os, errno
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from anal_utils import weight_fun, calc_weights

from scipy import optimize, stats
from src_py.metrics_utils import  calculate_deltas_signed

pathIN  = "../temp_results/nn_rhorho_Variant-All_soft_c012s_hits_c0s_Unweighted_False_NO_NUM_CLASSES_21_SIZE_1000_LAYERS_6/monit_npy/"
calc_hits_c0s   = np.load(pathIN+'train_soft_calc_hits_c012s.npy')
preds_hits_c0s  = np.load(pathIN+'train_soft_preds_hits_c012s.npy')
#calc_hits_c0s   = np.load(pathIN+'valid_soft_calc_c012s.npy')
#preds_hits_c0s  = np.load(pathIN+'valid_soft_preds_c012s.npy')

pathIN  = "../temp_results/nn_rhorho_Variant-All_soft_c012s_hits_c1s_Unweighted_False_NO_NUM_CLASSES_21_SIZE_1000_LAYERS_6/monit_npy/"
calc_hits_c1s   = np.load(pathIN+'train_soft_calc_hits_c012s.npy')
preds_hits_c1s  = np.load(pathIN+'train_soft_preds_hits_c012s.npy')
#calc_hits_c1s   = np.load(pathIN+'valid_soft_calc_c012s.npy')
#preds_hits_c1s  = np.load(pathIN+'valid_soft_preds_c012s.npy')

pathIN  = "../temp_results/nn_rhorho_Variant-All_soft_c012s_hits_c2s_Unweighted_False_NO_NUM_CLASSES_21_SIZE_1000_LAYERS_6/monit_npy/"
calc_hits_c2s   = np.load(pathIN+'train_soft_calc_hits_c012s.npy')
preds_hits_c2s  = np.load(pathIN+'train_soft_preds_hits_c012s.npy')
#calc_hits_c2s   = np.load(pathIN+'valid_soft_calc_c012s.npy')
#preds_hits_c2s  = np.load(pathIN+'valid_soft_preds_c012s.npy')

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

calc_c012s   = np.zeros((data_len, 3))
preds_c012s   = np.zeros((data_len, 3))
for i in range(data_len):
    calc_c012s[i][0] = calc_c0s[i] * (2./21.)
    calc_c012s[i][1] = calc_c1s[i] * (2./21.) -1.0
    calc_c012s[i][2] = calc_c2s[i] * (2./21.) -1.0

    preds_c012s[i][0] = preds_c0s[i] * (2./21.)
    preds_c012s[i][1] = preds_c1s[i] * (2./21.) -1.0
    preds_c012s[i][2] = preds_c2s[i] * (2./21.) -1.0

print calc_c012s[1:]
print "tu jestem"

k2PI= 2* np.pi


#----------------------------------------------------------------------------------
                           
i = 1
filename = "soft_c012s_calc_preds_rhorho_Variant-All_nc_21_event_1"
x = np.linspace(0, k2PI, 21)
plt.plot(x,weight_fun(x, *calc_c012s[i]), 'o', label='Generated')
plt.plot(x,weight_fun(x, *preds_c012s[i]), 'd', label=r'Classification: $C_0, C_1, C_2$')
plt.legend(loc='upper left')
plt.ylim([1.0, 2.2])
plt.xlim([-0.2, k2PI+0.2])
#plt.yticks(np.arange(0.0, 2.25, 0.25))
plt.xlabel(r'$\alpha^{CP}$ [rad]')
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


calc_w_nc21  =  calc_weights(21, calc_c012s)
preds_w_nc21 =  calc_weights(21, preds_c012s)
delt_argmax_nc21 =  calculate_deltas_signed(np.argmax(preds_w_nc21[:], axis=1), np.argmax(calc_w_nc21[:], axis=1), 21)      
nc21=21.0


filename = "soft_c012s_delt_argmax_rhorho_Variant-All_nc_21"
plt.hist(delt_argmax_nc21, histtype='step', bins=21, color = 'black')
plt.xlim([-5, 5])
plt.ylim([0.0, 45000])
plt.xlabel(r'$\Delta_{class}[idx]$')
#plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_argmax_nc21) 
std  = np.std(delt_argmax_nc21)
meanerr = stats.sem(delt_argmax_nc21)
meanrad = np.mean(delt_argmax_nc21) * k2PI/21.0
stdrad  = np.std(delt_argmax_nc21) * k2PI/21.0
meanraderr = stats.sem(delt_argmax_nc21) * k2PI/21.0

table_vals=[[r'Classification: $C_0, C_1, C_2$'],
            [" "],
            [r"mean = {:0.3f}$\pm$ {:1.3f} [idx]".format(mean, meanerr)],
            ["std = {:1.3f} [idx]".format(std)],
            [" "],
            [r"mean = {:0.3f}$\pm$ {:1.3f} [rad]".format(meanrad, meanraderr)],
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
filename = "soft_c012s_c0s_rhorho_Variant-All_nc21"

plt.hist(calc_c0s, histtype='step', bins=21,  color = 'black', linestyle='--', label = "Generated" )
plt.hist(preds_c0s, histtype='step', bins=21,  color = 'red', label = r"Classification: $C_0, C_1, C_2$")
#plt.xlim([-7, 7])
plt.ylim([0, 12000])
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
filename = "soft_c012s_delt_c0s_rhorho_Variant-All_nc21"
plt.hist(delt_c0s, histtype='step', bins=np.arange(-20,21)-0.5,  color = 'black')
plt.xlim([-5.0, 5.0])
plt.xlabel(r'$C_0: \Delta_{class}$ [idx]')
#plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_c0s)
std  = np.std(delt_c0s)
meanerr = stats.sem(delt_c0s)
meanC = np.mean(delt_c0s) * 2.0/21.0
stdC  = np.std(delt_c0s) * 2.0/21.0
meanerrC = meanerr * 2.0/21.0

table_vals=[[r'Classification: $C_0, C_1, C_2$'],
            [" "],
            [r"mean = {:0.3f}$\pm$ {:1.3f} [idx]".format(mean, meanerr)],
            ["std = {:1.3f} [idx]".format(std)],
            [" "],
            [r"mean = {:0.3f}$\pm$ {:1.3f}".format(meanC, meanerrC)],
            ["std = {:1.3f}".format(stdC)]
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
filename = "soft_c012s_c1s_rhorho_Variant-All_nc21"

plt.hist(calc_c1s, histtype='step', bins=21,  color = 'black', linestyle='--', label = "Generated")
plt.hist(preds_c1s, histtype='step', bins=21,  color = 'red', label = r"Classification: $C_0, C_1, C_2$")
#plt.xlim([0, 20.0])
plt.ylim([0, 6000])
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
filename = "soft_c012s_delt_c1s_rhorho_Variant-All_nc21"

plt.hist(delt_c1s, histtype='step', bins=np.arange(-20,21)-0.5,  color = 'black')
plt.xlim([-5.0, 5.0])
plt.xlabel(r'$C_1: \Delta_{class}$ [idx]')
#plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_c1s)
std  = np.std(delt_c1s)
meanerr = stats.sem(delt_c1s)
meanC = np.mean(delt_c1s) * 2.0/21.0
stdC  = np.std(delt_c1s) * 2.0/21.0
meanerrC = meanerr * 2.0/21.0


table_vals=[[r'Classification: $C_0, C_1, C_2$'],
            [" "],
            [r"mean = {:0.3f}$\pm$ {:1.3f} [idx]".format(mean, meanerr)],
            ["std = {:1.3f} [idx]".format(std)],
            [" "],
            [r"mean = {:0.3f}$\pm$ {:1.3f}".format(meanC, meanerrC)],
            ["std = {:1.3f}".format(stdC)]
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
filename = "soft_c012s_c2s_rhorho_Variant-All_nc21"

plt.hist(calc_c2s, histtype='step', bins=21,  color = 'black',  linestyle='--', label = "Generated")
plt.hist(preds_c2s, histtype='step', bins=21,  color = 'red', label = r"Classification: $C_0, C_1, C_2$")
#plt.xlim([0, 20.0])
plt.ylim([0, 6000])
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
filename = "soft_c012s_delt_c2s_rhorho_Variant-All_nc21"

plt.hist(delt_c2s, histtype='step', bins=np.arange(-20,21)-0.5,  color = 'black')
plt.xlim([-5.0, 5.0])
plt.xlabel(r'$C_2: \Delta_{class}$ [idx]')
#plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_c2s)
std  = np.std(delt_c2s)
meanerr = stats.sem(delt_c2s)
meanC = np.mean(delt_c2s) * 2.0/21.0
stdC  = np.std(delt_c2s) * 2.0/21.0
meanerrC = meanerr * 2.0/21.0



table_vals=[[r'Classification: $C_0, C_1, C_2$'],
            [" "],
            [r"mean = {:0.3f}$\pm$ {:1.3f} [idx]".format(mean, meanerr)],
            ["std = {:1.3f} [idx]".format(std)],
            [" "],
            [r"mean = {:0.3f}$\pm$ {:1.3f}".format(meanC, meanerrC)],
            ["std = {:1.3f}".format(stdC)]
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
