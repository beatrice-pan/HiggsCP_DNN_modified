import numpy as np
from src_py.prepare_utils import read_raw_root
import argparse
import os


def read_raw_all(kind, args):
    print "Reading %s" % kind

    data_path = args.IN

    all_data = []
    all_weights = []
    for letter in ["b","c","d","e","f","g","h","i","k"][:args.DATASETS]:
        name = os.path.join(data_path, "pythia.H.a1a1.1M.%s.%s.outTUPLE_labFrame" % (letter, kind))
        print letter, name
        data, weights = read_raw_root(name, num_particles=9)
        all_data += [data]
        all_weights += [weights]
    all_data = np.concatenate(all_data)
    all_weights = np.concatenate(all_weights)
    return all_data, all_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train classifier')
    parser.add_argument("-d", "--datasets", dest="DATASETS", default=9, type=int, help="number of datasets to prepare")
    parser.add_argument("-i", "--input", dest="IN", default=os.environ["A1A1_DATA"])
    args = parser.parse_args()

    data_00, weights_00 = read_raw_all("CPmix_00", args)
    data_02, weights_02 = read_raw_all("CPmix_02", args)
    data_04, weights_04 = read_raw_all("CPmix_04", args)
    data_06, weights_06 = read_raw_all("CPmix_06", args)
    data_08, weights_08 = read_raw_all("CPmix_08", args)
    data_10, weights_10 = read_raw_all("CPmix_10", args)
    data_12, weights_12 = read_raw_all("CPmix_12", args)
    data_14, weights_14 = read_raw_all("CPmix_14", args)
    data_16, weights_16 = read_raw_all("CPmix_16", args)
    data_18, weights_18 = read_raw_all("CPmix_18", args)
    data_20, weights_20 = read_raw_all("CPmix_20", args)

    print "In total: prepared %d events." % len(weights_00)
    np.testing.assert_array_almost_equal(data_00, data_02)
    np.testing.assert_array_almost_equal(data_00, data_04)
    np.testing.assert_array_almost_equal(data_00, data_06)
    np.testing.assert_array_almost_equal(data_00, data_08)
    np.testing.assert_array_almost_equal(data_00, data_10)
    np.testing.assert_array_almost_equal(data_00, data_12)
    np.testing.assert_array_almost_equal(data_00, data_14)
    np.testing.assert_array_almost_equal(data_00, data_16)
    np.testing.assert_array_almost_equal(data_00, data_18)
    np.testing.assert_array_almost_equal(data_00, data_20)

    np.random.seed(123)
    perm = np.random.permutation(len(weights_00))

    data_path = args.IN

    np.save(os.path.join(data_path, "a1a1_raw.data.npy"), data_00)
    np.save(os.path.join(data_path, "a1a1_raw.w_00.npy"), weights_00)
    np.save(os.path.join(data_path, "a1a1_raw.w_02.npy"), weights_02)
    np.save(os.path.join(data_path, "a1a1_raw.w_04.npy"), weights_04)
    np.save(os.path.join(data_path, "a1a1_raw.w_06.npy"), weights_06)
    np.save(os.path.join(data_path, "a1a1_raw.w_08.npy"), weights_08)
    np.save(os.path.join(data_path, "a1a1_raw.w_10.npy"), weights_10)
    np.save(os.path.join(data_path, "a1a1_raw.w_12.npy"), weights_12)
    np.save(os.path.join(data_path, "a1a1_raw.w_14.npy"), weights_14)
    np.save(os.path.join(data_path, "a1a1_raw.w_16.npy"), weights_16)
    np.save(os.path.join(data_path, "a1a1_raw.w_18.npy"), weights_18)
    np.save(os.path.join(data_path, "a1a1_raw.w_20.npy"), weights_20)
    np.save(os.path.join(data_path, "a1a1_raw.perm.npy"), perm)
