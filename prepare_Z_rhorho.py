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
        name = os.path.join(data_path, 
            "pythia.Z_65_155.rhorho.1M.%s.outTUPLE_labFrame" % (letter))
        print letter, name
        data, weights = read_raw_root(name, num_particles=7)
        all_data += [data]
        all_weights += [weights]
    all_data = np.concatenate(all_data)
    all_weights = np.concatenate(all_weights)
    return all_data, all_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare rhorhoZ data')
    parser.add_argument("-d", "--datasets", dest="DATASETS", type=int, default='2', 
        help="number of datasets to prepare")
    parser.add_argument("-i", "--input", dest="IN", default=os.environ["RHORHO_DATA"])
    args = parser.parse_args()

    data_a, weights_a = read_raw_all("vector", args)

    print "In total: %d examples." % len(weights_a)

    np.random.seed(123)
    perm = np.random.permutation(len(weights_a))

    data_path = args.IN

    np.save(os.path.join(data_path, "Z_65_155.rhorho_raw.data.npy"), data_a)
    np.save(os.path.join(data_path, "Z_65_155.rhorho_raw.w_a.npy"), weights_a)
    np.save(os.path.join(data_path, "Z_65_155.rhorho_raw.perm.npy"), perm)