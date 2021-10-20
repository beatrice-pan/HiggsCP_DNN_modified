import numpy as np
from glob import glob
import os, errno

import matplotlib.pyplot as plt

from anal_utils import calculate_metrics_regr_c012s_from_file

filelist_rhorho_Variant_All=[]
filelist_rhorho_Variant_All.append('../laptop_results_dropout=0/nn_rhorho_Variant-All_regr_c012s_hits_c0s_Unweighted_False_NO_NUM_CLASSES_0/monit_npy/')


metrics_Variant_All = [calculate_metrics_regr_c012s_from_file(filelist_rhorho_Variant_All[0], 3),calculate_metrics_regr_c012s_from_file(filelist_rhorho_Variant_All[0], 5),
                       calculate_metrics_regr_c012s_from_file(filelist_rhorho_Variant_All[0], 7), calculate_metrics_regr_c012s_from_file(filelist_rhorho_Variant_All[0], 9),
                       calculate_metrics_regr_c012s_from_file(filelist_rhorho_Variant_All[0], 11), calculate_metrics_regr_c012s_from_file(filelist_rhorho_Variant_All[0], 13),
                       calculate_metrics_regr_c012s_from_file(filelist_rhorho_Variant_All[0], 15), calculate_metrics_regr_c012s_from_file(filelist_rhorho_Variant_All[0], 17),
                       calculate_metrics_regr_c012s_from_file(filelist_rhorho_Variant_All[0], 19), calculate_metrics_regr_c012s_from_file(filelist_rhorho_Variant_All[0], 21),
                       calculate_metrics_regr_c012s_from_file(filelist_rhorho_Variant_All[0], 23), calculate_metrics_regr_c012s_from_file(filelist_rhorho_Variant_All[0], 25),
                       calculate_metrics_regr_c012s_from_file(filelist_rhorho_Variant_All[0], 27), calculate_metrics_regr_c012s_from_file(filelist_rhorho_Variant_All[0], 29),
                       calculate_metrics_regr_c012s_from_file(filelist_rhorho_Variant_All[0], 31), calculate_metrics_regr_c012s_from_file(filelist_rhorho_Variant_All[0], 33),
                       calculate_metrics_regr_c012s_from_file(filelist_rhorho_Variant_All[0], 35), calculate_metrics_regr_c012s_from_file(filelist_rhorho_Variant_All[0], 37),
                       calculate_metrics_regr_c012s_from_file(filelist_rhorho_Variant_All[0], 39), calculate_metrics_regr_c012s_from_file(filelist_rhorho_Variant_All[0], 41),
                       calculate_metrics_regr_c012s_from_file(filelist_rhorho_Variant_All[0], 43), calculate_metrics_regr_c012s_from_file(filelist_rhorho_Variant_All[0], 45),
                       calculate_metrics_regr_c012s_from_file(filelist_rhorho_Variant_All[0], 47), calculate_metrics_regr_c012s_from_file(filelist_rhorho_Variant_All[0], 49),
                       calculate_metrics_regr_c012s_from_file(filelist_rhorho_Variant_All[0], 51)]

           
metrics_Variant_All = np.stack(metrics_Variant_All)

#binning for horisontal axis
x = np.array([3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51])


# Now start plotting metrics
# Make better plots here, add axes labels, add color labels, store into figures/*.eps, figures/*.pdf files
# Should we first convert to histograms (?)

#---------------------------------------------------------------------

pathOUT = "figures/"
filename = "regr_c012s_acc_rhorh0_Variant-All"
# example plt.plot(x, metrics_Variant_All[:, 0],'o', label=r'$\sigma$' )
plt.plot(x, metrics_Variant_All[:, 0],'o', label=r'$|\Delta_{class}| < 1$')
plt.plot(x, metrics_Variant_All[:, 1],'x', label=r'$|\Delta_{class}| < 2$')
plt.plot(x, metrics_Variant_All[:, 2],'d', label=r'$|\Delta_{class}| < 3$')
plt.plot(x, metrics_Variant_All[:, 3],'v', label=r'$|\Delta_{class}| < 4$')
plt.ylim([0.0, 1.5])
plt.xticks(x)
plt.legend()
plt.xlabel(r'$N_{class}$')
plt.ylabel('Fraction')
plt.title('Feautures list: Variant-All')

ax = plt.gca()
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

#---------------------------------------------------------------------
plt.clf()
#---------------------------------------------------------------------

pathOUT = "figures/"
filename = "regr_c012s_acc_alphaCP_rhorho_Variant-All"

# example plt.plot(x, metrics_Variant_All[:, 0],'o', label=r'$\sigma$' )
plt.plot(x, metrics_Variant_All[:, 8],'o', label=r'$|\Delta\alpha^{CP}| < 0.25[rad]$')
plt.plot(x, metrics_Variant_All[:, 9],'x', label=r'$|\Delta\alpha^{CP}| < 0.50[rad]$')
plt.plot(x, metrics_Variant_All[:, 10],'d', label=r'$|\Delta\alpha^{CP}| < 0.75[rad]$')
plt.plot(x, metrics_Variant_All[:, 11],'v', label=r'$|\Delta\alpha^{CP}| < 1.0[rad]$')
plt.ylim([0.0, 1.5])
plt.xticks(x)
plt.legend()
plt.xlabel(r'$N_{class}$')
plt.ylabel('Fraction')
plt.title('Feautures list: Variant-All')

ax = plt.gca()
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

#---------------------------------------------------------------------
plt.clf()
#---------------------------------------------------------------------

pathOUT = "figures/"
filename = "regr_c012s_meanDelt_class_rhorho_Variant-All"

plt.errorbar(x, metrics_Variant_All[:,4], yerr=metrics_Variant_All[:,14], label=r'$<\Delta_{class}> [idx]$', linestyle = '', marker = 'o')
plt.plot([3,51],[0,0],linestyle = "--", color = "black")

plt.ylim([-0.5, 0.5])
plt.xticks(x)
plt.legend()
plt.xlabel('Number of classes')
plt.ylabel(r'$<\Delta>$ classes')
plt.title('Feautures list: Variant-All')

ax = plt.gca()
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

#---------------------------------------------------------------------
plt.clf()
#---------------------------------------------------------------------

pathOUT = "figures/"
filename = "regr_c012s_meanDelt_alphaCP_rhorho_Variant-All"

plt.errorbar(x, metrics_Variant_All[:,7], yerr=metrics_Variant_All[:,15], label=r'$<\Delta \alpha^{CP}> [rad]$', linestyle = '', marker = 'o')
plt.plot([3,51],[0,0],linestyle = "--", color = "black")

plt.ylim([0.0, 0.5])
plt.xticks(x)
plt.legend()
#plt.ylim([-0.5, 0.5])
plt.ylim([-0.3, 0.3])
plt.xlabel(r'$N_{class}$')
plt.ylabel(r'$<\Delta \alpha^{CP}>$ [rad]')
plt.title('Feautures list: Variant-All')

ax = plt.gca()
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

#---------------------------------------------------------------------
plt.clf()
#---------------------------------------------------------------------

pathOUT = "figures/"
filename = "regr_c012s_L1delt_w_rhorho_Variant_All"
plt.plot(x, metrics_Variant_All[:, 12],'o', label=r'$l_1$ with $wt^{norm}$')
plt.plot(x, metrics_Variant_All[:, 5],'d', label=r'$l_1$ with $wt$')


plt.ylim([0.0, 0.2])
plt.xticks(x)
plt.legend()
plt.xlabel(r'$N_{class}$')
plt.ylabel(r'$l_1$')
plt.title('Feautures list: Variant-All')

ax = plt.gca()
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
    
#---------------------------------------------------------------------
plt.clf()
#---------------------------------------------------------------------


pathOUT = "figures/"
filename = "regr_c012s_L2delt_w_rhorho_Variant_All"

plt.plot([0], marker='None',
           linestyle='None', label=r'Regression: $C_0, C_1, C_2$')
plt.plot(x, metrics_Variant_All[:, 13]*x,'o', color = "black", label=r'$l_2$ with $wt^{norm}$')
plt.plot([3,51],[0.2,0.2],linestyle = "--", color = "black")
plt.plot(x, metrics_Variant_All[:, 6],'d', color = "orange", label=r'$l_2$ with $wt$')

plt.ylim([0.0, 0.4])
plt.xticks(x)
plt.legend()
plt.xlabel(r'$N_{class}$')
plt.ylabel(r'$l_2$')
#plt.title('Feautures list: Variant-All')

ax = plt.gca()
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
#---------------------------------------------------------------------
plt.clf()
#---------------------------------------------------------------------
 
