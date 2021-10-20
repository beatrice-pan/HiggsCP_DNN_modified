import sys
import os, errno
import numpy as np
from scipy import optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def weight_fun(x, a, b, c):
    return a + b * np.cos(x) + c * np.sin(x)

pathOUT = "figures/"

#step function
ci = 0

cs_pred = np.zeros((52,5000,3))
cs_calc = np.zeros((52,5000,3))

x = np.linspace(0,2,10000)*np.pi
score = np.zeros(25)
for i in range(3,52,2):
    x_class = np.linspace(0,2,i)*np.pi
    calc = np.load("/net/people/plgerichter/HiggsCP_2019/CPmix-kacper/HiggsCP/temp_results/nn_rhorho_Variant-All_regr_weights_hits_c0s_Unweighted_False_NO_NUM_CLASSES_{0}_SIZE_1000_LAYERS_6/monit_npy/test_regr_calc_weights.npy".format(i))
    preds = np.load("/net/people/plgerichter/HiggsCP_2019/CPmix-kacper/HiggsCP/temp_results/nn_rhorho_Variant-All_regr_weights_hits_c0s_Unweighted_False_NO_NUM_CLASSES_{0}_SIZE_1000_LAYERS_6/monit_npy/test_regr_preds_weights.npy".format(i))
    x2 = np.linspace(0,2,i)*np.pi
    #print i
    #print (2./(i-1)*np.pi)
    
    for j in range(5000):
        
        cs_calc[int((i-3)/2),j], _ = optimize.curve_fit(weight_fun, x2, calc[j], p0=[1, 1, 1])
        cs_pred[int((i-3)/2),j], _ = optimize.curve_fit(weight_fun, x2, preds[j], p0=[1, 1, 1])
        calc_y = weight_fun(x,*cs_calc[int((i-3)/2),j])
        #calc_y *= i
        xx = ((x + 1/(i-1)*np.pi) // (2./(i-1)*np.pi)) * 2 *np.pi / (i-1)
        pred_y =  np.interp(xx, x_class, weight_fun(x_class,*cs_pred[int((i-3)/2),j]))
        #pred_y *= i
        #score[int((i-3)/2)] += sum((calc_y-pred_y)**2)*1e-8
        score[int((i-3)/2)] += np.sqrt(sum((calc_y-pred_y)**2)*2e-4*np.pi)/5000

pathOUT = "figures/"
filename = "regr_wt_L2delt_rhorho_Variant_All_nc_not_norm"

plt.plot([0], marker='None',
           linestyle='None', label=r'Regression: wt')
plt.semilogy(np.arange(3,52,2),score,'o', label=r'$l_2$', color = "black")

plt.xticks(np.arange(3,52,2))
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

