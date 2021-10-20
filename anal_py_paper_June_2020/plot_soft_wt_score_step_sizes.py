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

layers = ["3", "4","5", "6","7", "8"]
sizes = ["100", "200","400", "600","800", "1000", "1200"]

x_class = np.linspace(0,2,21)*np.pi
x2 = np.linspace(0,2,21)*np.pi

print "Nclass = 21"
x = np.linspace(0,2,10000)*np.pi
score = np.zeros(42)
i = 0
for l in layers:
    for s in sizes:
        calc = np.load("/net/people/plgerichter/HiggsCP_2019/CPmix-kacper/HiggsCP/temp_results/nn_rhorho_Variant-All_soft_weights_hits_c0s_Unweighted_False_NO_NUM_CLASSES_21_SIZE_{0}_LAYERS_{1}/monit_npy/softmax_calc_w.npy".format(s,l))
        preds = np.load("/net/people/plgerichter/HiggsCP_2019/CPmix-kacper/HiggsCP/temp_results/nn_rhorho_Variant-All_soft_weights_hits_c0s_Unweighted_False_NO_NUM_CLASSES_21_SIZE_{0}_LAYERS_{1}/monit_npy/softmax_preds_w.npy".format(s,l))
    
        for j in range(5000):
        
            cs_calc, _ = optimize.curve_fit(weight_fun, x2, calc[j], p0=[1, 1, 1])
            cs_pred, _ = optimize.curve_fit(weight_fun, x2, preds[j], p0=[1, 1, 1])
            calc_y = weight_fun(x,*cs_calc)
            calc_y *= 21
            xx = ((x + 1/(21-1)*np.pi) // (2./(21-1)*np.pi)) * 2 *np.pi / (21-1)
            pred_y =  np.interp(xx, x_class, weight_fun(x_class,*cs_pred))
            pred_y *= 21
            score[i] += np.sqrt(sum((calc_y-pred_y)**2)*2e-4*np.pi)/5000
        print score[i]
        i += 1






x_class = np.linspace(0,2,51)*np.pi
x2 = np.linspace(0,2,51)*np.pi

print "Nclass =	51"
x = np.linspace(0,2,10000)*np.pi
score = np.zeros(42)
i = 0
for l in layers:
    for	s in sizes:
        calc = np.load("/net/people/plgerichter/HiggsCP_2019/CPmix-kacper/HiggsCP/temp_results/nn_rhorho_Variant-All_soft_weights_hits_c0s_Unweighted_False_NO_NUM_CLASSES_51_SIZE_{0}_LAYERS_{1}/monit_npy/softmax_calc_w.npy".format(s,l))
        preds = np.load("/net/people/plgerichter/HiggsCP_2019/CPmix-kacper/HiggsCP/temp_results/nn_rhorho_Variant-All_soft_weights_hits_c0s_Unweighted_False_NO_NUM_CLASSES_51_SIZE_{0}_LAYERS_{1}/monit_npy/softmax_calc_w.npy".format(s,l))

        for j in range(5000):

            cs_calc, _ = optimize.curve_fit(weight_fun, x2, calc[j], p0=[1, 1, 1])
            cs_pred, _ = optimize.curve_fit(weight_fun, x2, preds[j], p0=[1, 1, 1])
            calc_y = weight_fun(x,*cs_calc)
            calc_y *= 51
            xx = ((x + 1/(51-1)*np.pi) // (2./(51-1)*np.pi)) * 2 *np.pi / (51-1)
            pred_y =  np.interp(xx, x_class, weight_fun(x_class,*cs_pred))
            pred_y *= 51
            score[i] += np.sqrt(sum((calc_y-pred_y)**2)*2e-4*np.pi)/5000
	print score[i]
        i += 1

