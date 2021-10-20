import numpy as np
import os, errno

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from anal_utils import calculate_metrics_from_file
from anal_utils import calculate_metrics_regr_c012s_from_file

filelist_rhorho_Variant_All = []

for i in range(3, 52, 2):
    filelist_rhorho_Variant_All.append('../laptop_results_dropout=0/nn_rhorho_Variant-All_soft_weights_hits_c0s_Unweighted_False_NO_NUM_CLASSES_{}/monit_npy/'.format(i))

metrics_softmax_Variant_All = [calculate_metrics_from_file(filelist_rhorho_Variant_All[0], 3),calculate_metrics_from_file(filelist_rhorho_Variant_All[1], 5),
                       calculate_metrics_from_file(filelist_rhorho_Variant_All[2], 7), calculate_metrics_from_file(filelist_rhorho_Variant_All[3], 9),
                       calculate_metrics_from_file(filelist_rhorho_Variant_All[4], 11), calculate_metrics_from_file(filelist_rhorho_Variant_All[5], 13),
                       calculate_metrics_from_file(filelist_rhorho_Variant_All[6], 15), calculate_metrics_from_file(filelist_rhorho_Variant_All[7], 17),
                       calculate_metrics_from_file(filelist_rhorho_Variant_All[8], 19), calculate_metrics_from_file(filelist_rhorho_Variant_All[9], 21),
                       calculate_metrics_from_file(filelist_rhorho_Variant_All[10], 23), calculate_metrics_from_file(filelist_rhorho_Variant_All[11], 25),
                       calculate_metrics_from_file(filelist_rhorho_Variant_All[12], 27), calculate_metrics_from_file(filelist_rhorho_Variant_All[13], 29),
                       calculate_metrics_from_file(filelist_rhorho_Variant_All[14], 31), calculate_metrics_from_file(filelist_rhorho_Variant_All[15], 33),
                       calculate_metrics_from_file(filelist_rhorho_Variant_All[16], 35), calculate_metrics_from_file(filelist_rhorho_Variant_All[17], 37),
                       calculate_metrics_from_file(filelist_rhorho_Variant_All[18], 39), calculate_metrics_from_file(filelist_rhorho_Variant_All[19], 41),
                       calculate_metrics_from_file(filelist_rhorho_Variant_All[20], 43), calculate_metrics_from_file(filelist_rhorho_Variant_All[21], 45),
                       calculate_metrics_from_file(filelist_rhorho_Variant_All[22], 47), calculate_metrics_from_file(filelist_rhorho_Variant_All[23], 49),
                       calculate_metrics_from_file(filelist_rhorho_Variant_All[24], 51)]
           
metrics_softmax_Variant_All = np.stack(metrics_softmax_Variant_All)


filelist_rhorho_regr_Variant_All=[]
filelist_rhorho_regr_Variant_All.append('../laptop_results_dropout=0/nn_rhorho_Variant-All_regr_c012s_hits_c0s_Unweighted_False_NO_NUM_CLASSES_0/monit_npy/')


metrics_regr_Variant_All = [calculate_metrics_regr_c012s_from_file(filelist_rhorho_regr_Variant_All[0], 3),calculate_metrics_regr_c012s_from_file(filelist_rhorho_regr_Variant_All[0], 5),
                       calculate_metrics_regr_c012s_from_file(filelist_rhorho_regr_Variant_All[0], 7), calculate_metrics_regr_c012s_from_file(filelist_rhorho_regr_Variant_All[0], 9),
                       calculate_metrics_regr_c012s_from_file(filelist_rhorho_regr_Variant_All[0], 11), calculate_metrics_regr_c012s_from_file(filelist_rhorho_regr_Variant_All[0], 13),
                       calculate_metrics_regr_c012s_from_file(filelist_rhorho_regr_Variant_All[0], 15), calculate_metrics_regr_c012s_from_file(filelist_rhorho_regr_Variant_All[0], 17),
                       calculate_metrics_regr_c012s_from_file(filelist_rhorho_regr_Variant_All[0], 19), calculate_metrics_regr_c012s_from_file(filelist_rhorho_regr_Variant_All[0], 21),
                       calculate_metrics_regr_c012s_from_file(filelist_rhorho_regr_Variant_All[0], 23), calculate_metrics_regr_c012s_from_file(filelist_rhorho_regr_Variant_All[0], 25),
                       calculate_metrics_regr_c012s_from_file(filelist_rhorho_regr_Variant_All[0], 27), calculate_metrics_regr_c012s_from_file(filelist_rhorho_regr_Variant_All[0], 29),
                       calculate_metrics_regr_c012s_from_file(filelist_rhorho_regr_Variant_All[0], 31), calculate_metrics_regr_c012s_from_file(filelist_rhorho_regr_Variant_All[0], 33),
                       calculate_metrics_regr_c012s_from_file(filelist_rhorho_regr_Variant_All[0], 35), calculate_metrics_regr_c012s_from_file(filelist_rhorho_regr_Variant_All[0], 37),
                       calculate_metrics_regr_c012s_from_file(filelist_rhorho_regr_Variant_All[0], 39), calculate_metrics_regr_c012s_from_file(filelist_rhorho_regr_Variant_All[0], 41),
                       calculate_metrics_regr_c012s_from_file(filelist_rhorho_regr_Variant_All[0], 43), calculate_metrics_regr_c012s_from_file(filelist_rhorho_regr_Variant_All[0], 45),
                       calculate_metrics_regr_c012s_from_file(filelist_rhorho_regr_Variant_All[0], 47), calculate_metrics_regr_c012s_from_file(filelist_rhorho_regr_Variant_All[0], 49),
                       calculate_metrics_regr_c012s_from_file(filelist_rhorho_regr_Variant_All[0], 51)]
         
metrics_regr_Variant_All = np.stack(metrics_regr_Variant_All)




# Now start plotting metrics
# Make better plots here, add axes labels, add color labels, store into figures/*.eps, figures/*.pdf files
# Should we first convert to histograms (?)

#binning for horisontal axis
x = np.array([3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51])

#---------------------------------------------------------------------

pathOUT = "figures/"
filename = "comp_soft_regr_acc_wt_rhorho_Variant-All_nc"

plt.plot([0], marker='None',
           linestyle='None', label='Classification: wt')
plt.plot(x, metrics_softmax_Variant_All[:, 0],'o', label=r'$|\Delta_{class}| < 1$')
plt.plot(x, metrics_softmax_Variant_All[:, 1],'x', label=r'$|\Delta_{class}| < 2$')
plt.plot(x, metrics_softmax_Variant_All[:, 2],'d', label=r'$|\Delta_{class}| < 3$')
plt.plot(x, metrics_softmax_Variant_All[:, 3],'v', label=r'$|\Delta_{class}| < 4$')
plt.plot([0], marker='None',
         linestyle='None', label=' ')
plt.plot([0], marker='None',
         linestyle='None', label=r'Regression: $C_0, C_1, C_2$')
plt.plot(x, metrics_regr_Variant_All[:, 0],'o', label=r'$|\Delta_{class}| < 1$')
plt.plot(x, metrics_regr_Variant_All[:, 1],'x', label=r'$|\Delta_{class}| < 2$')
plt.plot(x, metrics_regr_Variant_All[:, 2],'d', label=r'$|\Delta_{class}| < 3$')
plt.plot(x, metrics_regr_Variant_All[:, 3],'v', label=r'$|\Delta_{class}| < 4$')
plt.legend(loc='upper right', ncol=1)
plt.ylim([0.1, 1.1])
plt.yticks(np.arange(0.2, 1.1, 0.2))
#plt.xticks(np.arange(3, 52, 6))
plt.xlim([1, 53])
plt.xlabel(r'$N_{class}$')
plt.ylabel('Fraction')
#plt.title('Feautures list: Variant-All')

ax = plt.gca()

lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

if filename:
    try:
        os.makedirs(pathOUT)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.savefig(pathOUT + filename+".eps", bbox_extra_artists=(lgd,), bbox_inches='tight')
    print('Saved '+pathOUT + filename+".eps")
    plt.savefig(pathOUT + filename+".pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    print('Saved '+pathOUT + filename+".pdf")
else:
    plt.show()

#---------------------------------------------------------------------
plt.clf()
#---------------------------------------------------------------------
#---------------------------------------------------------------------

pathOUT = "figures/"
filename = "comp_soft_regr_acc_alphaCP_rhorho_Variant-All_nc"

plt.plot([0], marker='None',
           linestyle='None', label='Classification: wt')
plt.plot(x, metrics_softmax_Variant_All[:, 8],'o', label=r'$|\Delta\alpha^{CP}_{max}| < 0.25[rad]$')
plt.plot(x, metrics_softmax_Variant_All[:, 9],'x', label=r'$|\Delta\alpha^{CP}_{max}| < 0.50[rad]$')
plt.plot(x, metrics_softmax_Variant_All[:,10],'d', label=r'$|\Delta\alpha^{CP}_{max}| < 0.75[rad]$')
plt.plot(x, metrics_softmax_Variant_All[:,11],'v', label=r'$|\Delta\alpha^{CP}_{max}| < 1.0[rad]$')
plt.plot([0], marker='None',
         linestyle='None', label=' ')
plt.plot([0], marker='None',
         linestyle='None', label=r'Regression: $C_0, C_1, C_2$')
plt.plot(x, metrics_regr_Variant_All[:, 8],'o', label=r'$|\Delta\alpha^{CP}_{max}| < 0.25[rad]$')
plt.plot(x, metrics_regr_Variant_All[:, 9],'x', label=r'$|\Delta\alpha^{CP}_{max}| < 0.50[rad]$')
plt.plot(x, metrics_regr_Variant_All[:,10],'d', label=r'$|\Delta\alpha^{CP}_{max}| < 0.75[rad]$')
plt.plot(x, metrics_regr_Variant_All[:,11],'v', label=r'$|\Delta\alpha^{CP}_{max}| < 1.0[rad]$')
plt.legend(loc='upper right', ncol=1)
plt.ylim([0.1, 1.1])
plt.yticks(np.arange(0.2, 1.1, 0.2))
#plt.xticks(np.arange(3, 52, 6))
plt.xlim([1, 53])
plt.xlabel(r'$N_{class}$')
plt.ylabel('Fraction')
#plt.title('Feautures list: Variant-All')

ax = plt.gca()

lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

if filename:
    try:
        os.makedirs(pathOUT)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.savefig(pathOUT + filename+".eps", bbox_extra_artists=(lgd,), bbox_inches='tight')
    print('Saved '+pathOUT + filename+".eps")
    plt.savefig(pathOUT + filename+".pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    print('Saved '+pathOUT + filename+".pdf")
else:
    plt.show()

#---------------------------------------------------------------------
plt.clf()
#---------------------------------------------------------------------


pathOUT = "figures/"
filename = "comp_soft_regr_L1delt_wt_rhorho_Variant-All_nc"

plt.plot(x, metrics_softmax_Variant_All[:, 5],'o', label='Classification: wt')
plt.plot(x, metrics_regr_Variant_All[:, 12],'d', label=r'Regression: $C_0, C_1, C_2$')

plt.yticks(np.arange(0.0, 0.09, 0.02))
plt.ylim([-0.01, 0.09])
#plt.xticks(np.arange(3, 52, 6))
plt.legend()
plt.xlabel(r'$N_{class}$')
plt.ylabel(r'$l_1$ with $wt^{norm}')
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

#binning for horisontal axis
x = np.array([3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51])


pathOUT = "figures/"
filename = "comp_soft_regr_L2delt_wt_rhorho_Variant-All_nc"

plt.plot(x, metrics_softmax_Variant_All[:, 13]*x,'o', label='Classification: wt')
plt.plot([3,51],[0.2,0.2],linestyle = "--", color = "black")
plt.plot(x, metrics_regr_Variant_All[:, 13]*x,'d', label=r'Regression: $C_0, C_1, C_2$')


#plt.yticks(np.arange(0.0, 0.09, 0.02))
plt.ylim([-0.0, 0.40])
plt.xticks(x)
plt.legend()
plt.xlabel(r'$N_{class}$')
plt.ylabel(r'$l_2$ with $wt^{norm}$')
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
#---------------------------------------------------------------------

pathOUT = "figures/"
filename = "comp_soft_regr_wt_meanDelt_class_rhorho_Variant-All_nc"

plt.errorbar(x, metrics_softmax_Variant_All[:,4], yerr=metrics_softmax_Variant_All[:,14], label='Classification: wt', linestyle = '', marker = 'o')
plt.errorbar(x, metrics_regr_Variant_All[:,4], yerr=metrics_regr_Variant_All[:,14], label=r'Regression: $C_0, C_1, C_2$', linestyle = '', marker = 'd')
plt.plot([3,51],[0,0],linestyle = "--", color = "black")
plt.ylim([-0.5, 0.5])
plt.xticks(x)
plt.legend()
plt.xlabel(r'$N_{class}$')
plt.ylabel(r'$<\Delta_{class}>$ [idx]')
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

pathOUT = "figures/"
filename = "comp_soft_regr_wt_meanDelt_alphaCP_rhorho_Variant-All_nc"

plt.errorbar(x, metrics_softmax_Variant_All[:,7], yerr=metrics_softmax_Variant_All[:,15], label='Classification: wt', linestyle = '', marker = 'o')
plt.errorbar(x, metrics_regr_Variant_All[:,7], yerr=metrics_regr_Variant_All[:,15], label=r'Regression: $C_0, C_1, C_2$', linestyle = '', marker = 'd')
plt.plot([3,51],[0,0],linestyle = "--", color = "black")

#plt.ylim([0.0, 0.5])
plt.xticks(x)
plt.legend()
#plt.ylim([-0.5, 0.5])
plt.ylim([-0.2, 0.2])
plt.xlabel(r'$N_{class}$')
plt.ylabel(r'$<\Delta \alpha^{CP}_{max}>$ [rad]')
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
