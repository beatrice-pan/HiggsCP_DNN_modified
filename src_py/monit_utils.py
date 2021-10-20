import matplotlib.pyplot as plt
from src_py.plot_utils import plot_one_TH1D, plot_two_TH1D
import os, errno
import numpy as np

def is_nan(x):
    return (x is np.nan or x != x)


def monit_plots(pathOUT, args, event, w_a, w_b):

    if args.PLOT_FEATURES == "FILTER":
        filt = [x==1 for x in event.cols[:,-1]]
        w_a = w_a[filt]
        w_b = w_b[filt]
    else:
        filt = [not is_nan(x) for x in data]
        w_a = w_a[filt]
        w_b = w_b[filt]

    filedir =  pathOUT 

    for i in range(len(event.cols[0,:])-1):
        plot_two_TH1D(event.cols[:,i], filedir, filename = event.labels[i], w_a = w_a, w_b = w_b, filt = filt)

        
    if args.FEAT == "Variant-1.1" and args.TYPE == "nn_rhorho": #acoangle depending on y1y2 sign
        y1y2_pos = np.array(event.cols[:,-3][filt]*event.cols[:,-2][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        plot_two_TH1D(event.cols[:,-4], filedir, filename = "aco_angle_y1y2_pos", w_a = w_a*y1y2_pos, w_b = w_b*y1y2_pos, filt = filt)
        plot_two_TH1D(event.cols[:,-4], filedir, filename = "aco_angle_y1y2_neg", w_a = w_a*y1y2_neg, w_b = w_b*y1y2_neg, filt = filt)
        
    if args.FEAT == "Variant-1.1" and args.TYPE == "nn_a1rho": #acoangle depending on y1y2 sign
        y1y2_pos = np.array(event.cols[:,-9][filt]*event.cols[:,-8][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        plot_two_TH1D(event.cols[:,-11], filedir, filename = "aco_angle_1_y1y2_pos", w_a = w_a*y1y2_pos, w_b = w_b*y1y2_pos, filt = filt)
        plot_two_TH1D(event.cols[:,-11], filedir, filename = "aco_angle_1_y1y2_neg", w_a = w_a*y1y2_neg, w_b = w_b*y1y2_neg, filt = filt)
        y1y2_pos = np.array(event.cols[:,-8][filt]*event.cols[:,-7][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        plot_two_TH1D(event.cols[:,-10], filedir, filename = "aco_angle_2_y1y2_pos", w_a = w_a*y1y2_pos, w_b = w_b*y1y2_pos, filt = filt)
        plot_two_TH1D(event.cols[:,-10], filedir, filename = "aco_angle_2_y1y2_neg", w_a = w_a*y1y2_neg, w_b = w_b*y1y2_neg, filt = filt)
        y1y2_pos = np.array(event.cols[:,-4][filt]*event.cols[:,-3][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        plot_two_TH1D(event.cols[:,-6], filedir, filename = "aco_angle_3_y1y2_pos", w_a = w_a*y1y2_pos, w_b = w_b*y1y2_pos, filt = filt)
        plot_two_TH1D(event.cols[:,-6], filedir, filename = "aco_angle_3_y1y2_neg", w_a = w_a*y1y2_neg, w_b = w_b*y1y2_neg, filt = filt)
        y1y2_pos = np.array(event.cols[:,-3][filt]*event.cols[:,-2][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        plot_two_TH1D(event.cols[:,-5], filedir, filename = "aco_angle_4_y1y2_pos", w_a = w_a*y1y2_pos, w_b = w_b*y1y2_pos, filt = filt)
        plot_two_TH1D(event.cols[:,-5], filedir, filename = "aco_angle_4_y1y2_neg", w_a = w_a*y1y2_neg, w_b = w_b*y1y2_neg, filt = filt)


        
    if args.FEAT == "Variant-1.1" and args.TYPE == "nn_a1a1": #acoangle depending on y1y2 sign
        y1y2_pos = np.array(event.cols[:,-45][filt]*event.cols[:,-44][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        plot_two_TH1D(event.cols[:,-49], filedir, filename = "aco_angle_1_y1y2_pos", w_a = w_a*y1y2_pos, w_b = w_b*y1y2_pos, filt = filt)
        plot_two_TH1D(event.cols[:,-49], filedir, filename = "aco_angle_1_y1y2_neg", w_a = w_a*y1y2_neg, w_b = w_b*y1y2_neg, filt = filt)
        y1y2_pos = np.array(event.cols[:,-43][filt]*event.cols[:,-42][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        plot_two_TH1D(event.cols[:,-48], filedir, filename = "aco_angle_2_y1y2_pos", w_a = w_a*y1y2_pos, w_b = w_b*y1y2_pos, filt = filt)
        plot_two_TH1D(event.cols[:,-48], filedir, filename = "aco_angle_2_y1y2_neg", w_a = w_a*y1y2_neg, w_b = w_b*y1y2_neg, filt = filt)
        y1y2_pos = np.array(event.cols[:,-41][filt]*event.cols[:,-40][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        plot_two_TH1D(event.cols[:,-47], filedir, filename = "aco_angle_3_y1y2_pos", w_a = w_a*y1y2_pos, w_b = w_b*y1y2_pos, filt = filt)
        plot_two_TH1D(event.cols[:,-47], filedir, filename = "aco_angle_3_y1y2_neg", w_a = w_a*y1y2_neg, w_b = w_b*y1y2_neg, filt = filt)
        y1y2_pos = np.array(event.cols[:,-39][filt]*event.cols[:,-38][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        plot_two_TH1D(event.cols[:,-46], filedir, filename = "aco_angle_4_y1y2_pos", w_a = w_a*y1y2_pos, w_b = w_b*y1y2_pos, filt = filt)
        plot_two_TH1D(event.cols[:,-46], filedir, filename = "aco_angle_4_y1y2_neg", w_a = w_a*y1y2_neg, w_b = w_b*y1y2_neg, filt = filt)
 
        y1y2_pos = np.array(event.cols[:,-33][filt]*event.cols[:,-32][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        plot_two_TH1D(event.cols[:,-37], filedir, filename = "aco_angle_5_y1y2_pos", w_a = w_a*y1y2_pos, w_b = w_b*y1y2_pos, filt = filt)
        plot_two_TH1D(event.cols[:,-37], filedir, filename = "aco_angle_5_y1y2_neg", w_a = w_a*y1y2_neg, w_b = w_b*y1y2_neg, filt = filt)
        y1y2_pos = np.array(event.cols[:,-31][filt]*event.cols[:,-30][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        plot_two_TH1D(event.cols[:,-36], filedir, filename = "aco_angle_6_y1y2_pos", w_a = w_a*y1y2_pos, w_b = w_b*y1y2_pos, filt = filt)
        plot_two_TH1D(event.cols[:,-36], filedir, filename = "aco_angle_6_y1y2_neg", w_a = w_a*y1y2_neg, w_b = w_b*y1y2_neg, filt = filt)
        y1y2_pos = np.array(event.cols[:,-29][filt]*event.cols[:,-28][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        plot_two_TH1D(event.cols[:,-35], filedir, filename = "aco_angle_7_y1y2_pos", w_a = w_a*y1y2_pos, w_b = w_b*y1y2_pos, filt = filt)
        plot_two_TH1D(event.cols[:,-35], filedir, filename = "aco_angle_7_y1y2_neg", w_a = w_a*y1y2_neg, w_b = w_b*y1y2_neg, filt = filt)
        y1y2_pos = np.array(event.cols[:,-27][filt]*event.cols[:,-26][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        plot_two_TH1D(event.cols[:,-34], filedir, filename = "aco_angle_8_y1y2_pos", w_a = w_a*y1y2_pos, w_b = w_b*y1y2_pos, filt = filt)
        plot_two_TH1D(event.cols[:,-34], filedir, filename = "aco_angle_8_y1y2_neg", w_a = w_a*y1y2_neg, w_b = w_b*y1y2_neg, filt = filt)

        y1y2_pos = np.array(event.cols[:,-21][filt]*event.cols[:,-20][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        plot_two_TH1D(event.cols[:,-25], filedir, filename = "aco_angle_9_y1y2_pos", w_a = w_a*y1y2_pos, w_b = w_b*y1y2_pos, filt = filt)
        plot_two_TH1D(event.cols[:,-25], filedir, filename = "aco_angle_9_y1y2_neg", w_a = w_a*y1y2_neg, w_b = w_b*y1y2_neg, filt = filt)
        y1y2_pos = np.array(event.cols[:,-19][filt]*event.cols[:,-18][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        plot_two_TH1D(event.cols[:,-24], filedir, filename = "aco_angle_10_y1y2_pos", w_a = w_a*y1y2_pos, w_b = w_b*y1y2_pos, filt = filt)
        plot_two_TH1D(event.cols[:,-24], filedir, filename = "aco_angle_10_y1y2_neg", w_a = w_a*y1y2_neg, w_b = w_b*y1y2_neg, filt = filt)
        y1y2_pos = np.array(event.cols[:,-17][filt]*event.cols[:,-16][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        plot_two_TH1D(event.cols[:,-23], filedir, filename = "aco_angle_11_y1y2_pos", w_a = w_a*y1y2_pos, w_b = w_b*y1y2_pos, filt = filt)
        plot_two_TH1D(event.cols[:,-23], filedir, filename = "aco_angle_11_y1y2_neg", w_a = w_a*y1y2_neg, w_b = w_b*y1y2_neg, filt = filt)
        y1y2_pos = np.array(event.cols[:,-15][filt]*event.cols[:,-14][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        plot_two_TH1D(event.cols[:,-22], filedir, filename = "aco_angle_12_y1y2_pos", w_a = w_a*y1y2_pos, w_b = w_b*y1y2_pos, filt = filt)
        plot_two_TH1D(event.cols[:,-22], filedir, filename = "aco_angle_12_y1y2_neg", w_a = w_a*y1y2_neg, w_b = w_b*y1y2_neg, filt = filt)

        y1y2_pos = np.array(event.cols[:,-9][filt]*event.cols[:,-8][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        plot_two_TH1D(event.cols[:,-13], filedir, filename = "aco_angle_13_y1y2_pos", w_a = w_a*y1y2_pos, w_b = w_b*y1y2_pos, filt = filt)
        plot_two_TH1D(event.cols[:,-13], filedir, filename = "aco_angle_13_y1y2_neg", w_a = w_a*y1y2_neg, w_b = w_b*y1y2_neg, filt = filt)
        y1y2_pos = np.array(event.cols[:,-7][filt]*event.cols[:,-6][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        plot_two_TH1D(event.cols[:,-12], filedir, filename = "aco_angle_14_y1y2_pos", w_a = w_a*y1y2_pos, w_b = w_b*y1y2_pos, filt = filt)
        plot_two_TH1D(event.cols[:,-12], filedir, filename = "aco_angle_14_y1y2_neg", w_a = w_a*y1y2_neg, w_b = w_b*y1y2_neg, filt = filt)
        y1y2_pos = np.array(event.cols[:,-5][filt]*event.cols[:,-4][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        plot_two_TH1D(event.cols[:,-11], filedir, filename = "aco_angle_15_y1y2_pos", w_a = w_a*y1y2_pos, w_b = w_b*y1y2_pos, filt = filt)
        plot_two_TH1D(event.cols[:,-11], filedir, filename = "aco_angle_15_y1y2_neg", w_a = w_a*y1y2_neg, w_b = w_b*y1y2_neg, filt = filt)
        y1y2_pos = np.array(event.cols[:,-3][filt]*event.cols[:,-2][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        plot_two_TH1D(event.cols[:,-10], filedir, filename = "aco_angle_16_y1y2_pos", w_a = w_a*y1y2_pos, w_b = w_b*y1y2_pos, filt = filt)
        plot_two_TH1D(event.cols[:,-10], filedir, filename = "aco_angle_16_y1y2_neg", w_a = w_a*y1y2_neg, w_b = w_b*y1y2_neg, filt = filt)

    if args.FEAT == "Variant-4.1": 
        for i in range(len(event.labels_suppl)):
            plot_one_TH1D(event.cols_suppl[:,i], filedir, filename = event.labels_suppl[i], w = w_a, filt = filt)
