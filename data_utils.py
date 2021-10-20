import numpy as np
import random

class Dataset(object):
    def __init__(self, x, wa, wb):
        self.x = x[:, :-1]
        self.filt = x[:, -1]
        self.wa = wa
        self.wb = wb

        self.n = x.shape[0]
        self._next_id = 0
        self.shuffle()

    def shuffle(self):
        perm = np.arange(self.n)
        np.random.shuffle(perm)
        self.x = self.x[perm]
        self.wa = self.wa[perm]
        self.wb = self.wb[perm]
        self.filt = self.filt[perm]
        self._next_id = 0

    def next_batch(self, batch_size):
        if self._next_id + batch_size >= self.n:
            self.shuffle()

        cur_id = self._next_id
        self._next_id += batch_size
        return (self.x[cur_id:cur_id+batch_size],
                self.wa[cur_id:cur_id+batch_size],
                self.wb[cur_id:cur_id+batch_size],
                self.filt[cur_id:cur_id+batch_size])


def read_np(filename):
    with open(filename) as f:
        return np.load(f)


class EventDatasets(object):

    def __init__(self, event, w_a, w_b, perm, filtered=False, raw=False, miniset=False,  unweighted=False, smear_polynomial=False):
        data = event.cols[:, :-1]
        filt = event.cols[:, -1]

        if miniset:
            print("Miniset")
            train_ids = perm[-300000:-200000]
            print(len(train_ids))
            valid_ids = perm[-200000:-100000]
            test_ids = perm[-100000:]
        else:
            train_ids = perm[:-200000]
            valid_ids = perm[-200000:-100000]
            test_ids = perm[-100000:]
	
	    if filtered:
            train_ids = train_ids[filt[train_ids] == 1]
            valid_ids = valid_ids[filt[valid_ids] == 1]
            test_ids = test_ids[filt[test_ids] == 1]

        if not raw:
            print "SCALE!!"
            means = data[train_ids].mean(0)
            stds = data[train_ids].std(0)
            data = (data - means) / stds

        data = np.concatenate([data, filt.reshape([-1, 1])], 1)

        def unweight(x):
            return 0 if x < random.random()*2 else 1

        if unweighted:
            w_a = np.array(map(unweight, w_a))
            w_b = np.array(map(unweight, w_b))

        self.train = Dataset(data[train_ids], w_a[train_ids], w_b[train_ids])
        self.valid = Dataset(data[valid_ids], w_a[valid_ids], w_b[valid_ids])
        self.test = Dataset(data[test_ids], w_a[test_ids], w_b[test_ids])

	if smear_polynomial:

		data[:,-5:-1]=np.transpose(event.valid_cols)
		self.valid = Dataset(data[valid_ids], w_a[valid_ids], w_b[valid_ids])
		self.test = Dataset(data[test_ids], w_a[test_ids], w_b[test_ids])
