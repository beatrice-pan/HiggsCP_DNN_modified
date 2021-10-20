import numpy as np
import os

from scipy import stats

from sklearn.metrics import roc_auc_score

from src_py.metrics_utils import calculate_deltas_unsigned, calculate_deltas_signed


def weight_fun(x, a, b, c):
    return a + b * np.cos(x) + c * np.sin(x)

def calc_weights(num_classes, coeffs):
    k2PI = 2* np.pi
    x = np.linspace(0, k2PI, num_classes)
    data_len = coeffs.shape[0]
    weights = np.zeros((data_len, num_classes))
    for i in range(data_len):
        weights[i] = weight_fun(x, *coeffs[i])
    return weights

def calc_argmaxs_distances(pred_arg_maxs, calc_arg_maxs, num_class):
    return calculate_deltas_signed(calc_arg_maxs, pred_arg_maxs, num_class)


def calculate_metrics_from_file(directory, num_classes):
#    calc_w = np.load(os.path.join(directory, 'softmax_calc_w.npy'))
#    preds_w = np.load(os.path.join(directory, 'softmax_preds_w.npy'))
    preds_w = np.load(os.path.join(directory, 'test_regr_preds_weights.npy'))
    calc_w = np.load(os.path.join(directory, 'test_regr_calc_weights.npy'))
    return calculate_metrics(calc_w, preds_w, num_classes)


def calculate_metrics(calc_w, preds_w, num_classes):
    pred_arg_maxs = np.argmax(preds_w, axis=1)
    calc_arg_maxs = np.argmax(calc_w, axis=1)

    k2PI = 2 * np.pi
    calc_pred_argmaxs_abs_distances = calculate_deltas_unsigned( pred_arg_maxs, calc_arg_maxs, num_classes)
    calc_pred_argmaxs_signed_distances = calculate_deltas_signed(pred_arg_maxs, calc_arg_maxs, num_classes)
    calc_pred_argmaxs_abs_distances_rad = calc_pred_argmaxs_abs_distances * k2PI/(1.0*num_classes)
    
    mean_deltas = np.mean(calc_pred_argmaxs_signed_distances, dtype=np.float64)
    mean_deltas_err = stats.sem(calc_pred_argmaxs_signed_distances)
    mean_deltas_rad = mean_deltas * k2PI/(1.0*num_classes)
    mean_deltas_err_rad = mean_deltas_err * k2PI/(1.0*num_classes)

    acc0 = (calc_pred_argmaxs_abs_distances <= 0).mean()
    acc1 = (calc_pred_argmaxs_abs_distances <= 1).mean()
    acc2 = (calc_pred_argmaxs_abs_distances <= 2).mean()
    acc3 = (calc_pred_argmaxs_abs_distances <= 3).mean()

    acc0_rad = (calc_pred_argmaxs_abs_distances_rad <= 0.25).mean()
    acc1_rad = (calc_pred_argmaxs_abs_distances_rad <= 0.50).mean()
    acc2_rad = (calc_pred_argmaxs_abs_distances_rad <= 0.75).mean()
    acc3_rad = (calc_pred_argmaxs_abs_distances_rad <= 1.00).mean()

    l1_delta_w = np.mean(np.abs(calc_w - preds_w), dtype=np.float64)
    l2_delta_w = np.sqrt(np.mean((calc_w - preds_w)**2), dtype=np.float64)
    # problem with format, should not be array
    l2_delta_w_err = stats.sem((calc_w - preds_w)**2)

    # calc_w, preds_w normalisation to probability
    calc_w_norm = calc_w / np.sum(calc_w, axis=1)[:, np.newaxis]
    preds_w_norm = preds_w / np.sum(preds_w, axis=1)[:, np.newaxis]
 
    l1_delta_w_norm = np.mean(np.abs(calc_w_norm - preds_w_norm), dtype=np.float64)
    l2_delta_w_norm = np.sqrt(np.mean((calc_w_norm - preds_w_norm)**2), dtype=np.float64)
    # problem with format, should not be array
    l2_delta_w_norm_err = stats.sem((calc_w_norm - preds_w_norm)**2)
  
    
    return np.array([acc0, acc1, acc2, acc3, mean_deltas, l1_delta_w, l2_delta_w, mean_deltas_rad, acc0_rad, acc1_rad, acc2_rad, acc3_rad,l1_delta_w_norm, l2_delta_w_norm, mean_deltas_err, mean_deltas_err_rad, l2_delta_w_err, l2_delta_w_norm_err ]) 


def calculate_metrics_regr_c012s_from_file(directory, num_classes):
    calc_c012s = np.load(os.path.join(directory,'test_regr_calc_c012s.npy'))
    pred_c012s = np.load(os.path.join(directory,'test_regr_preds_c012s.npy'))

    return calculate_metrics_regr_c012s(calc_c012s, pred_c012s, num_classes)


def calculate_metrics_regr_c012s(calc_c012s, pred_c012s, num_classes):
    calc_w  = calc_weights(num_classes, calc_c012s)
    preds_w = calc_weights(num_classes, pred_c012s)

    return calculate_metrics(calc_w, preds_w, num_classes)


def get_filename_for_class(pathIN, class_num, subset=None):
    d = '../monit_npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_{class_num}'
    if subset:
        d += "_WEIGHTS_SUBS" + str(subset)
    return d


# The primary versions of three methods below were
#  evaluate from tf_model.py
#  evaluate2 from tf_model.py
#  both using 
#  evaluate_preds  from tf_model.py
# when extending to multi-class something is not correctly
# implemented for handling numpy arrays. 


def evaluate_roc_auc(preds, wa, wb):
    n = len(preds)
    true_labels = np.concatenate([np.ones(n), np.zeros(n)])
    preds = np.concatenate([preds, preds])
    weights = np.concatenate([wa, wb])
    
    return roc_auc_score(true_labels, preds, sample_weight=weights)


def calculate_roc_auc(preds_w, calc_w, index_a, index_b):
    n, num_classes = calc_w.shape
    true_labels = np.concatenate([np.ones(n), np.zeros(n)])
    preds = np.concatenate([preds_w[:, index_a], preds_w[:, index_a]])
    weights = np.concatenate([calc_w[:, index_a], calc_w[:, index_b]])

    return roc_auc_score(true_labels, preds, sample_weight=weights)

# binary classification
def test_roc_auc(directory, num_class):
    calc_w = np.load(os.path.join(directory, 'softmax_calc_w.npy'))
    preds_w = np.load(os.path.join(directory, 'softmax_preds_w.npy'))
    
    oracle_roc_auc = []
    preds_roc_auc  = []
    
    for i in range(0, num_class):
         oracle_roc_auc  += [calculate_roc_auc(calc_w, calc_w, 0, i)]
         preds_roc_auc   += [calculate_roc_auc(preds_w, calc_w, 0, i)]
         print(i,
                  'oracle_roc_auc: {}'.format(calculate_roc_auc(calc_w, calc_w, 0, i)),
                  'preds_roc_auc: {}'.format(calculate_roc_auc(preds_w, calc_w, 0, i)))

    return oracle_roc_auc, preds_roc_auc
