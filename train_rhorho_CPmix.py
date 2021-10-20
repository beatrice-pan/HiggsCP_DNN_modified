import tensorflow as tf
from rhorho import RhoRhoEvent
from data_utils import read_np, EventDatasets
from tf_model import total_train, NeuralNetwork
import os


def run(args):
    data_path = args.IN

    print "Loading data"
    data = read_np(os.path.join(data_path, "rhorho_raw.data.npy"))
    w_00 = read_np(os.path.join(data_path, "rhorho_raw.w_00.npy"))
    w_01 = read_np(os.path.join(data_path, "rhorho_raw.w_01.npy"))
    w_02 = read_np(os.path.join(data_path, "rhorho_raw.w_02.npy"))
    w_03 = read_np(os.path.join(data_path, "rhorho_raw.w_03.npy"))
    w_04 = read_np(os.path.join(data_path, "rhorho_raw.w_04.npy"))
    w_05 = read_np(os.path.join(data_path, "rhorho_raw.w_05.npy"))
    w_06 = read_np(os.path.join(data_path, "rhorho_raw.w_06.npy"))
    w_07 = read_np(os.path.join(data_path, "rhorho_raw.w_07.npy"))
    w_08 = read_np(os.path.join(data_path, "rhorho_raw.w_08.npy"))
    w_09 = read_np(os.path.join(data_path, "rhorho_raw.w_09.npy"))
    w_10 = read_np(os.path.join(data_path, "rhorho_raw.w_10.npy"))
    perm = read_np(os.path.join(data_path, "rhorho_raw.perm.npy"))
    print "Read %d events" % data.shape[0]

    print "Processing data"
    event = RhoRhoEvent(data, args)
    
    w1 = read_np(os.path.join(data_path, "rhorho_raw.{0}.npy".format(args.W1))
    w2 = read_np(os.path.join(data_path, "rhorho_raw.{0}.npy".format(args.W2))
    points = EventDatasets(event, w1, w2, perm, miniset=args.MINISET, unweighted=args.UNWEIGHTED, smear_polynomial=(args.BETA>0), filtered=True)

    num_features = points.train.x.shape[1]
    print "Generated %d features" % num_features

    print "Initializing model"
    with tf.variable_scope("model1") as vs:
        model = NeuralNetwork(num_features, num_layers=args.LAYERS, size=args.SIZE, keep_prob=(1-args.DROPOUT), optimizer=args.OPT)

    with tf.variable_scope("model1", reuse=True) as vs:
        emodel = NeuralNetwork(num_features, num_layers=args.LAYERS, size=args.SIZE, keep_prob=(1-args.DROPOUT), optimizer=args.OPT)

    tf.global_variables_initializer().run()

    print "Training"
    total_train(model, points, emodel=emodel, batch_size=128, epochs=args.EPOCHS)


def start(args):
    sess = tf.Session()
    with sess.as_default():
        run(args)

if __name__ == "__main__":
    start(args = {})
