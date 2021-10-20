import sys
import numpy as np
import tensorflow as tf
from rhorho import RhoRhoEvent
from data_utilsZ import read_np, EventDatasetsZ
from tf_model import NeuralNetwork
from sklearn.metrics import roc_auc_score
import os


def train(model, dataset, batch_size=128, noise_fraction=0.2):
    sess = tf.get_default_session()
    epoch_size = dataset.n_ab / batch_size
    losses = []

    for i in range(epoch_size):
        x, wa, wb = dataset.next_batch(batch_size, noise_fraction=noise_fraction)
        # print x.shape, wa.shape, wb.shape
        loss, _ = sess.run([model.loss, model.train_op],
                           {model.x: x, model.wa: wa, model.wb: wb})
        losses.append(loss)

        if i % (epoch_size / 10) == 5:
            sys.stdout.write(". %.3f " % np.mean(losses))
            sys.stdout.flush()
    return np.mean(losses)


def total_train(model, data, emodel=None, batch_size=128, epochs=25, noise_fraction=0.2, saver=None):
    if emodel is None:
        emodel = model
    aucs = []
    max_perf = evaluate2(emodel, data.valid)
    print max_perf

    for i in range(epochs):
        sys.stdout.write("EPOCH: %d " % (i + 1))
        loss = train(model, data.train, batch_size, noise_fraction)
        train_auc = evaluate(emodel, data.train, noise_fraction=noise_fraction, at_most=100000)
        valid_auc = evaluate(emodel, data.valid, noise_fraction=noise_fraction)
        msg_str = "TRAIN LOSS: %.3f AUC: %.3f VALID AUC: %.3f" % (loss, train_auc, valid_auc)
        print msg_str
        tf.logging.info(msg_str)
        aucs += [valid_auc]

        # saver.save(tf.get_default_session(), "checkpoints/rhorho_model_%.1f.ckpt" % noise_fraction, i)
    return aucs


def predictions(model, dataset, noise_fraction=0.2, at_most=None):
    sess = tf.get_default_session()
    x = dataset.x_ab
    wa = dataset.w_a
    wb = dataset.w_b

    if at_most is not None:
        x = x[:at_most]
        wa = wa[:at_most]
        wb = wb[:at_most]

    n_ab = x.shape[0]
    n_c = int(noise_fraction * n_ab)
    if n_c > dataset.n_c:
        print("Not enough points of type C. %d < %d" % (dataset.n_c, n_c))
        n_c = dataset.n_c

    if n_c > 0:
        x = np.concatenate([
                x,
                dataset.x_c[:n_c]
            ])

        wa = np.concatenate([
                wa,
                dataset.w_c[:n_c]
            ])
        wb = np.concatenate([
                wb,
                dataset.w_c[:n_c]
            ])

    p = sess.run(model.p, {model.x: x})
    if n_c > 0:
        return x[:n_ab], p[:n_ab], wa[:n_ab], wb[:n_ab]

    return x, p, wa, wb


def evaluate(model, dataset, noise_fraction=0.2, at_most=None):
    _, ps, was, wbs = predictions(model, dataset, noise_fraction, at_most)

    return evaluate_preds(ps, was, wbs)
    # replace by line below to get max sensitivity
    # return evaluate_preds(was/(was+wbs), was, wbs)


def evaluate2(model, dataset, at_most=None):
    _, ps, was, wbs = predictions(model, dataset, 0.0, at_most)

    return evaluate_preds(was / (was + wbs), was, wbs)


def evaluate_preds(preds, wa, wb):
    n = len(preds)
    true_labels = np.concatenate([np.ones(n), np.zeros(n)])
    preds = np.concatenate([preds, preds])
    weights = np.concatenate([wa, wb])

    return roc_auc_score(true_labels, preds, sample_weight=weights)


def load_rhorho(args):
    data_path = args.IN

    print "Loading data"
    data_ab = read_np(os.path.join(data_path, "rhorho_raw.data.npy"))
    w_a = read_np(os.path.join(data_path, "rhorho_raw.w_a.npy"))
    w_b = read_np(os.path.join(data_path, "rhorho_raw.w_b.npy"))
    perm_ab = read_np(os.path.join(data_path, "rhorho_raw.perm.npy"))
    print "Read %d events" % data_ab.shape[0]

    data_c = read_np(os.path.join(data_path, "Z_65_155.rhorho_raw.data.npy"))
    w_c = read_np(os.path.join(data_path, "Z_65_155.rhorho_raw.w_a.npy"))
    perm_c = read_np(os.path.join(data_path, "Z_65_155.rhorho_raw.perm.npy"))
    print "Read %d C events" % data_c.shape[0]

    print "Processing data"
    event_ab = RhoRhoEvent(data_ab, args)
    data_ab = event_ab.cols[:, :-1]  # Don't use last column for training, which is a filter
    filt_ab = event_ab.cols[:, -1] == 1  # Filter out events with last column values.
    # perm_ab = perm_ab[event_ab.cols[:, -1] == 1]
    print "Num filtered points: %d" % data_ab.shape[0]

    event_c = RhoRhoEvent(data_c, args)
    data_c = event_c.cols[:, :-1]  # Don't use last column for training, which is a filter
    filt_c = event_c.cols[:, -1] == 1
    # data_c = data_c[event_c.cols[:, -1] == 1]  # Filter out events with last column values.
    # perm_c = perm_c[event_c.cols[:, -1] == 1]
    print "Num filtered C points: %d" % data_c.shape[0]

    points = EventDatasetsZ(data_ab, data_c, w_a, w_b, w_c, perm_ab, perm_c, filt_ab, filt_c)
    return points


def run(args):
    points = load_rhorho(args)
    num_features = points.train.x_ab.shape[1]
    print "Generated %d features" % num_features

    print "Initializing model"
    with tf.variable_scope("model1") as vs:
        model = NeuralNetwork(num_features, num_layers=args.LAYERS, size=args.SIZE, keep_prob=(1-args.DROPOUT), optimizer=args.OPT)

    with tf.variable_scope("model1", reuse=True) as vs:
        emodel = NeuralNetwork(num_features, num_layers=args.LAYERS, size=args.SIZE, keep_prob=(1-args.DROPOUT), optimizer=args.OPT)

    # saver = tf.train.Saver()
    tf.global_variables_initializer().run()

    print "Training"
    # total_train(model, points, emodel=emodel, batch_size=128, epochs=5, noise_fraction=0.5, saver=saver)
    total_train(model, points, emodel=emodel, batch_size=128, epochs=args.EPOCHS, noise_fraction=args.Z_NOISE_FRACTION)



def start(args):
    sess = tf.Session()
    with sess.as_default():
        run(args)

if __name__ == "__main__":
    start(args = {})