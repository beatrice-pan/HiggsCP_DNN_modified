import numpy as np
from prepare_utils import read_raw_root
import argparse
import os


def read_raw_all(kind, args):
    print "Reading %s" % kind

    data_path = args.IN

    all_data = []
    all_weights = []
    for letter in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k"][:args.DATASETS]:
        name = os.path.join(data_path, "pythia.H.a1rho.1M.%s.%s.outTUPLE_labFrame" % (letter, kind))
        print letter, name
        data, weights = read_raw_root(name, num_particles=8)
        all_data += [data]
        all_weights += [weights]
    all_data = np.concatenate(all_data)
    all_weights = np.concatenate(all_weights)
    return all_data, all_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train classifier')
    parser.add_argument("-d", "--datasets", dest="DATASETS", default='2', type=int, help="number of datasets to prepare")
    parser.add_argument("-i", "--input", dest="IN", default=os.environ["A1RHO_DATA"])
    args = parser.parse_args()

    data_a, weights_a = read_raw_all("scalar", args)
    data_b, weights_b = read_raw_all("pseudoscalar", args)

    print "In total: %d examples." % len(weights_a)
    np.testing.assert_array_almost_equal(data_a, data_b)

    np.random.seed(123)
    perm = np.random.permutation(len(weights_a))

    data_path = args.IN

    np.save(os.path.join(data_path, "a1rho_raw.data.npy"), data_a)
    np.save(os.path.join(data_path, "a1rho_raw.w_a.npy"), weights_a)
    np.save(os.path.join(data_path, "a1rho_raw.w_b.npy"), weights_b)
    np.save(os.path.join(data_path, "a1rho_raw.perm.npy"), perm)
    