import numpy as np


class DatasetZ(object):
    def __init__(self, x, w):
        self.x = x
        self.w = w

        self.n = x.shape[0]
        self._next_id = 0
        self.shuffle()

    def shuffle(self):
        perm = np.arange(self.n)
        np.random.shuffle(perm)
        self.x = self.x[perm]
        self.w = self.w[perm]
        self._next_id = 0

    def next_batch(self, batch_size):
        if self._next_id + batch_size >= self.n:
            self.shuffle()

        cur_id = self._next_id
        self._next_id += batch_size
        return (self.x[cur_id:cur_id+batch_size],
                self.w[cur_id:cur_id+batch_size])


class ThreeDatasets(object):
    def __init__(self, x_ab, w_a, w_b, x_c, w_c):
        self.x_ab = x_ab
        self.w_a = w_a
        self.w_b = w_b
        self.x_c = x_c
        self.w_c = w_c

        self.n_ab = x_ab.shape[0]
        self.n_c = x_c.shape[0]

        self._next_id_ab = 0
        self._next_id_c = 0
        self.shuffle()

    def shuffle(self):
        perm_ab = np.arange(self.n_ab)
        np.random.shuffle(perm_ab)
        self.x_ab = self.x_ab[perm_ab]
        self.w_a = self.w_a[perm_ab]
        self.w_b = self.w_b[perm_ab]

        perm_c = np.arange(self.n_c)
        np.random.shuffle(perm_c)
        self.x_c = self.x_c[perm_c]
        self.w_c = self.w_c[perm_c]

        self._next_id_ab = 0
        self._next_id_c = 0

    def next_batch(self, batch_size, noise_fraction):
        num_c = int(batch_size * noise_fraction)
        num_ab = batch_size - num_c
        # print "requested ", batch_size, noise_fraction, num_ab, num_c
        if self._next_id_ab + num_ab >= self.n_ab or self._next_id_c + num_c >= self.n_c:
            self.shuffle()

        cur_id_ab = self._next_id_ab
        cur_id_c = self._next_id_c
        self._next_id_ab += num_ab
        self._next_id_c += num_c

        if num_c == 0:
            return (self.x_ab[cur_id_ab:cur_id_ab+num_ab],
                    self.w_a[cur_id_ab:cur_id_ab+num_ab],
                    self.w_b[cur_id_ab:cur_id_ab+num_ab])

        x = np.concatenate([
            self.x_ab[cur_id_ab:cur_id_ab+num_ab],
            self.x_c[cur_id_c:cur_id_c+num_c]])

        wa = np.concatenate([
            self.w_a[cur_id_ab:cur_id_ab+num_ab],
            self.w_c[cur_id_c:cur_id_c+num_c],
        ])

        wb = np.concatenate([
            self.w_b[cur_id_ab:cur_id_ab+num_ab],
            self.w_c[cur_id_c:cur_id_c+num_c],
        ])

        return x, wa, wb

def read_np(filename):
    with open(filename) as f:
        return np.load(f)


class EventDatasetsZ(object):

    def __init__(self, data_ab, data_c, w_a, w_b, w_c, perm_ab, perm_c, filt_ab, filt_c, raw=False):
        good_ids_ab = set([idx for idx in range(len(filt_ab)) if filt_ab[idx]])
        good_ids_c = set([idx for idx in range(len(filt_c)) if filt_c[idx]])
        print(len(good_ids_ab), len(good_ids_c))
        def filter_good_ab(arr):
            return np.array([idx for idx in arr if idx in good_ids_ab])
        def filter_good_c(arr):
            return np.array([idx for idx in arr if idx in good_ids_c])

        train_ids_ab = filter_good_ab(perm_ab[:-200000])
        train_ids_c = filter_good_c(perm_c[:-20000])
        valid_ids_ab = filter_good_ab(perm_ab[-200000:-100000])
        valid_ids_c = filter_good_c(perm_c[-20000:-10000])
        test_ids_ab = filter_good_ab(perm_ab[-100000:])
        test_ids_c = filter_good_c(perm_c[-10000:])

        # Hack! Remove when we have more data points of type C.
        valid_ids_c = train_ids_c

        print("Data sizes: train={} valid={} test={}".format(train_ids_ab.shape, valid_ids_ab.shape, test_ids_ab.shape))
        print("Data sizes: train={} valid={} test={}".format(train_ids_c.shape, valid_ids_c.shape, test_ids_c.shape))
        print("Data sizes: train={} valid={} test={}".format(w_a.shape, w_b.shape, w_c.shape))

        if not raw:
            print "SCALE!!"
            train_ab = data_ab[train_ids_ab]
            train_c = data_c[train_ids_c]
            all_train_data = np.concatenate([train_ab, train_c])
            means = all_train_data.mean(0)
            stds = all_train_data.std(0)
            data_ab = (data_ab - means) / stds
            data_c = (data_c - means) / stds

        self.train = ThreeDatasets(data_ab[train_ids_ab], w_a[train_ids_ab], w_b[train_ids_ab],
                                   data_c[train_ids_c], w_c[train_ids_c])
        self.valid = ThreeDatasets(data_ab[valid_ids_ab], w_a[valid_ids_ab], w_b[valid_ids_ab],
                                   data_c[valid_ids_c], w_c[valid_ids_c])
        self.test = ThreeDatasets(data_ab[test_ids_ab], w_a[test_ids_ab], w_b[test_ids_ab],
                                  data_c[test_ids_c], w_c[test_ids_c])