import numpy as np
from prepare_utils import read_raw_root
import argparse
import os


def read_raw_all(args, letters):

    data_path = args.IN

    all_data = []
    all_weights = []
    for i,letter in enumerate(letters):
        name = os.path.join(data_path, "pythia.Z_115_135.rhorho.1M.%s.outTUPLE_labFrame" % (letter))
        print letter, name
        data, weights = read_raw_root(name, num_particles=7)
        all_data += [data]
    all_data = np.concatenate(all_data)
    return all_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train classifier')
    parser.add_argument("-d", "--datasets", dest="DATASETS", default=2, type=int, help="number of datasets to prepare")
    parser.add_argument("-i", "--input", dest="IN", default=os.environ["RHORHO_DATA"])
    args = parser.parse_args()

    data = read_raw_all( args, ["a","b","c","d","e", "f", "g", "h", "i", "k"])

    data_path = args.IN

    np.save(os.path.join(data_path, "rhorhoZ_raw.data.npy"), data)

