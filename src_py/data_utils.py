import numpy as np
import random

class Dataset(object):
    def __init__(self, x, weights, argmaxs, c012s, hits_argmaxs, hits_c012s):
        self.x = x[:, :-1]
        self.filt = x[:, -1]
        self.weights = weights
        self.argmaxs = argmaxs
        self.c012s = c012s
        self.hits_argmaxs = hits_argmaxs
        self.hits_c012s = hits_c012s

        self.n = x.shape[0]
        self._next_id = 0
        self.mask = np.ones(self.n) == 1
        self.shuffle()

    def shuffle(self):
        perm = np.arange(self.n)
        np.random.shuffle(perm)
        self.x = self.x[perm]
        self.weights = self.weights[perm]
        self.argmaxs = self.argmaxs[perm]
        self.c012s = self.c012s[perm]
        self.hits_argmaxs = self.hits_argmaxs[perm]
        self.hits_c012s = self.hits_c012s[perm]
        self.filt = self.filt[perm]
        self._next_id = 0

    def next_batch(self, batch_size):
        if self._next_id + batch_size >= self.n:
            self.shuffle()

        cur_id = self._next_id
        self._next_id += batch_size
        return (self.x[cur_id:cur_id+batch_size],
                self.weights[cur_id:cur_id+batch_size], self.argmaxs[cur_id:cur_id+batch_size], self.c012s[cur_id:cur_id+batch_size],
                self.hits_argmaxs[cur_id:cur_id+batch_size], self.hits_c012s[cur_id:cur_id+batch_size], self.filt[cur_id:cur_id+batch_size])

def unweight(x):
    return 0 if x < random.random() * 2 else 1


class UnweightedDataset(object):
    def __init__(self, x, weights, argmaxs, c012s, hits_argmaxs, hits_c012s):
        self.x = x[:, :-1]
        self.filt = x[:, -1]
        self.weights = weights
        self.argmaxs = argmaxs
        self.c012s = c012s
        self.hits_argmaxs = hits_argmaxs
        self.hits_c012s = hits_c012s

        self.n = x.shape[0]
        self._next_id = 0
        self.mask = np.ones(self.n)==1
        self.shuffle()

    def weight(self, w_ind):
        if w_ind:
            self.w_ind = w_ind
            self.mask = np.array(map(unweight, self.weights[:, w_ind]))
            self.mask = self.mask > 0
            self.n = self.mask.sum()
        else:
            self.n = self.x.shape[0]
            self.mask = np.ones(self.n) == 1

    def shuffle(self):
        perm = np.arange(self.n)
        np.random.shuffle(perm)
        self.x = self.x[perm]
        self.weights = self.weights[perm]
        self.argmaxs = self.argmaxs[perm]
        self.c012s = self.c012s[perm]
        self.hits_argmaxs = self.hits_argmaxs[perm]
        self.hits_c012s = self.hits_c012s[perm]
        self.filt = self.filt[perm]
        self._next_id = 0

    def next_batch(self, batch_size):
        if self._next_id + batch_size >= self.n:
            self.shuffle()

        cur_id = self._next_id
        self._next_id += batch_size
        return (self.x[self.mask][cur_id:cur_id+batch_size],
                self.weights[self.mask][cur_id:cur_id+batch_size], self.argmaxs[self.mask][cur_id:cur_id+batch_size],
                self.c012s[self.mask][cur_id:cur_id+batch_size],
                self.hits_argmaxs[self.mask][cur_id:cur_id+batch_size], self.hits_c012s[self.mask][cur_id:cur_id+batch_size],
                self.filt[self.mask][cur_id:cur_id+batch_size])


def read_np(filename):
    # with open(filename) as f:
        # return np.load(f)
    return np.load(filename)
# :100000


class EventDatasets(object):

    def __init__(self, event, weights, argmaxs, perm, c012s, hits_argmaxs, hits_c012s, filtered=False, raw=False, miniset=False,  unweighted=False):
        data = event.cols[:, :-1]
        filt = event.cols[:, -1]

        if miniset:
            print("Miniset")
            train_ids = perm[-300000:-200000]
            print(len(train_ids))
            valid_ids = perm[-200000:-100000]
            test_ids = perm[-100000:]
            # train_ids, valid_ids, test_ids = perm[:int(len(perm) * 0.8)], perm[int(len(perm) * 0.8):int(len(perm) * 0.9)], perm[int(len(perm) * 0.9):]
        else:
            # train_ids = perm[:-200000]
            # valid_ids = perm[-200000:-100000]
            # test_ids = perm[-100000:]
            train_ids, valid_ids, test_ids = perm[:int(len(perm) * 0.8)], perm[int(len(perm) * 0.8):int(len(perm) * 0.9)], perm[int(len(perm) * 0.9):]

        if not raw:
            print ("SCALE!!")
            means = data[train_ids].mean(0)
            stds = data[train_ids].std(0)
            data = (data - means) / stds

        if filtered:
            train_ids = train_ids[filt[train_ids] == 1]
            valid_ids = valid_ids[filt[valid_ids] == 1]
            test_ids = test_ids[filt[test_ids] == 1]

        data = np.concatenate([data, filt.reshape([-1, 1])], 1)

        def unweight(x):
            return 0 if x < random.random()*2 else 1

        # if unweighted:
        #     w_a = np.array(map(unweight, w_a))
        #     w_b = np.array(map(unweight, w_b))
        self.train = Dataset(data[train_ids], weights[train_ids, :], argmaxs[train_ids], c012s[train_ids], hits_argmaxs[train_ids], hits_c012s[train_ids])
        self.valid = Dataset(data[valid_ids], weights[valid_ids, :], argmaxs[valid_ids], c012s[valid_ids], hits_argmaxs[valid_ids], hits_c012s[valid_ids])
        self.test = Dataset(data[test_ids], weights[test_ids, :], argmaxs[test_ids], c012s[test_ids], hits_argmaxs[test_ids], hits_c012s[test_ids])
        self.unweightedtest = UnweightedDataset(data[test_ids], weights[test_ids, :], argmaxs[test_ids], c012s[test_ids], hits_argmaxs[test_ids], hits_c012s[test_ids])
