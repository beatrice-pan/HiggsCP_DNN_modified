import numpy as np
from prepare_utils import read_raw_root
import argparse
import os
from scipy import optimize

def weight_fun(x, a, b, c):
    return a + b * np.cos(x) + c * np.sin(x)


def read_raw_all(kinds, args, letters):

    data_path = args.IN

    all_data = []
    all_weights = []
    for i,letter in enumerate(letters):
        name = os.path.join(data_path, "pythia.H.a1rho.1M.%s.CPmix_%s.outTUPLE_labFrame" % (letter, kinds[i]))
        print letter, name
        data, weights = read_raw_root(name, num_particles=8)
        all_data += [data]
        all_weights += [weights]
    all_data = np.concatenate(all_data)
    all_weights = np.concatenate(all_weights)
    return all_data, all_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train classifier')
    parser.add_argument("-d", "--datasets", dest="DATASETS", default=2, type=int, help="number of datasets to prepare")
    parser.add_argument("-i", "--input", dest="IN", default=os.environ["A1RHO_DATA"])
    args = parser.parse_args()

    #data_1, weights_1 = read_raw_all(["00","00","00","00","00","00","06","00","00","04",], args, ["f","b","c","d","e"])#,"a","g","h","i","k",])
    #data_2, weights_2 = read_raw_all(["04","04","04","06","06","06","10","04","04","10",], args, ["f","b","c","d","e"])#,"a","g","h","i","k",])
    #data_3, weights_3 = read_raw_all(["10","10","10","10","12","10","20","10","10","20",], args, ["f","b","c","d","e"])#,"a","g","h","i","k",])
    data, weights_1 = read_raw_all(["00","00","00","00","00"], args, ["f","b","c","d","e"])#,"a","g","h","i","k",])
    data, weights_2 = read_raw_all(["04","04","04","06","06"], args, ["f","b","c","d","e"])#,"a","g","h","i","k",])
    data, weights_3 = read_raw_all(["10","10","10","10","12"], args, ["f","b","c","d","e"])#,"a","g","h","i","k",])

    x = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0]) * np.pi
    #xs = [[0,0.4,1], [0,0.4,1], [0,0.4,1], [0,0.6,1], [0,0.6,1.2], [0,0.6,1], [0.6,1,2], [0,0.4,1], [0,0.4,1], [0.4,1,2]]
    xs = [[0,0.4,1], [0,0.4,1], [0,0.4,1], [0,0.6,1], [0,0.6,1.2]]
    weights = np.zeros((len(weights_1), 11))
    
    for j in range(5):
        for i in range(len(weights_1)//5):
            coeff, ccov = optimize.curve_fit(weight_fun, np.array(xs[j])*np.pi, [weights_1[(len(weights_1)//5)*j + i], weights_2[(len(weights_1)//5)*j + i], weights_3[(len(weights_1)//5)*j + i]], p0=[1, 1, 1])
            weights[(len(weights_1)//5)*j + i,:] = weight_fun(x, *coeff)

    print "In total: prepared %d events." % len(weights_1)
    #np.testing.assert_array_almost_equal(data_1, data_2)
    #np.testing.assert_array_almost_equal(data_1, data_3)

    np.random.seed(123)
    perm = np.random.permutation(len(weights_1))

    data_path = args.IN

    np.save(os.path.join(data_path, "a1rho_raw.data.npy"), data)
    #np.save(os.path.join(data_path, "a1rho_raw.data2.npy"), data_2)
    #np.save(os.path.join(data_path, "a1rho_raw.data3.npy"), data_3)    
    np.save(os.path.join(data_path, "a1rho_raw.w.npy"), weights.T)
    np.save(os.path.join(data_path, "a1rho_raw.w_00.npy"), weights[:,0])
    np.save(os.path.join(data_path, "a1rho_raw.w_02.npy"), weights[:,1])
    np.save(os.path.join(data_path, "a1rho_raw.w_04.npy"), weights[:,2])
    np.save(os.path.join(data_path, "a1rho_raw.w_06.npy"), weights[:,3])
    np.save(os.path.join(data_path, "a1rho_raw.w_08.npy"), weights[:,4])
    np.save(os.path.join(data_path, "a1rho_raw.w_10.npy"), weights[:,5])
    np.save(os.path.join(data_path, "a1rho_raw.w_12.npy"), weights[:,6])
    np.save(os.path.join(data_path, "a1rho_raw.w_14.npy"), weights[:,7])
    np.save(os.path.join(data_path, "a1rho_raw.w_16.npy"), weights[:,8])
    np.save(os.path.join(data_path, "a1rho_raw.w_18.npy"), weights[:,9])
    np.save(os.path.join(data_path, "a1rho_raw.w_20.npy"), weights[:,10])
    np.save(os.path.join(data_path, "a1rho_raw.perm.npy"), perm)

