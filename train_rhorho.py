import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import os, errno

from src_py.cpmix_utils import preprocess_data
from src_py.download_data_rhorho import download_data
from src_py.rhorho import RhoRhoEvent
from src_py.data_utils import EventDatasets
from src_py.tf_model import total_train, NeuralNetwork


# from src_py.monit_utils import monit_plots


def run(args):
    num_classes = args.NUM_CLASSES

    print("Downloading data")
    download_data(args)

    print("Preprocessing data")
    data, weights, argmaxs, perm, c012s, hits_argmaxs, hits_c012s = preprocess_data(args)

    print("Processing data")
    event = RhoRhoEvent(data, args)
    points = EventDatasets(event, weights, argmaxs, perm, c012s=c012s, hits_argmaxs=hits_argmaxs, hits_c012s=hits_c012s,
                           miniset=args.MINISET, unweighted=args.UNWEIGHTED)
    num_features = points.train.x.shape[1]
    print("Prepared %d features" % num_features)

    pathOUT = "temp_results/" + args.TYPE + "_" + args.FEAT + "_" + args.TRAINING_METHOD + "_" + args.HITS_C012s + "_Unweighted_" + str(
        args.UNWEIGHTED) + "_" + args.PLOT_FEATURES + "_NUM_CLASSES_" + str(args.NUM_CLASSES) + "_SIZE_" + str(
        args.SIZE) + "_LAYERS_" + str(args.LAYERS) + "_ZNOISE_" + str(args.Z_NOISE_FRACTION) + "/"
    if pathOUT:
        try:
            os.makedirs(pathOUT)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    pathOUT_npy = pathOUT + 'monit_npy/'
    if pathOUT_npy:
        try:
            os.makedirs(pathOUT_npy)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    pathOUT_plots = pathOUT + 'monit_plots/'
    if pathOUT_plots:
        try:
            os.makedirs(pathOUT_plots)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    if args.PLOT_FEATURES is not "NO":
        w_a = weights[:, 0]
        w_b = weights[:, num_classes / 2]
        monit_plots(pathOUT_plots, args, event, w_a, w_b)

    print("Initializing model")
    with tf.variable_scope("model1") as vs:
        model = NeuralNetwork(num_features, num_classes,
                              num_layers=args.LAYERS, size=args.SIZE,
                              keep_prob=(1 - args.DROPOUT), optimizer=args.OPT,
                              tloss=args.TRAINING_METHOD)

    with tf.variable_scope("model1", reuse=True) as vs:
        emodel = NeuralNetwork(num_features, num_classes,
                               num_layers=args.LAYERS, size=args.SIZE,
                               keep_prob=(1 - args.DROPOUT), optimizer=args.OPT,
                               tloss=args.TRAINING_METHOD)

    tf.global_variables_initializer().run()

    print("Training")
    total_train(pathOUT_npy, model, points, args, emodel=emodel, batch_size=128, epochs=args.EPOCHS)


def start(args):
    tf.reset_default_graph()
    sess = tf.Session()
    np.random.seed(781)
    tf.set_random_seed(781)
    with sess.as_default():
        run(args)


if __name__ == "__main__":
    start(args={})
