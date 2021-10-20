import os
import urllib

import numpy as np

DATA_URL = 'http://th-www.if.uj.edu.pl/~erichter/forHiggsCP/HiggsCP_data/a1a1/'


def download_data(args):
    data_path = args.IN
    if os.path.exists(data_path) is False:
        os.mkdir(data_path)
    download_weights(args)
    download_data_files(args)


def download_weights(args):
    data_path = args.IN
    # CPmix_index = 0 (scalar), 10 (pseudoscalar), 20 (scalar)
    CPmix_index = ['00', '02', '04', '06', '08', '10', '12', '14', '16', '18', '20']
    weights = []
    output_weight_file = os.path.join(data_path, 'a1a1_raw.w.npy')
    if os.path.exists(output_weight_file) and not args.FORCE_DOWNLOAD:
        print('Output weights file exists. Downloading data cancelled. If you want to force download use --force_download option')
        return
    for index in CPmix_index:
        filename = 'a1a1_raw.w_' + index + '.npy'
        print('Downloading ' + filename)
        filepath = os.path.join(data_path, filename)
        urllib.request.urlretrieve(DATA_URL + filename, filepath)
        weights.append(np.load(filepath))
    weights = np.stack(weights)
    np.save(output_weight_file, weights)


def download_data_files(args):
    data_path = args.IN
    files = ['a1a1_raw.data.npy', 'a1a1_raw.perm.npy', 'arg_maxs.npy', 'pcovs.npy', 'popts.npy', 'weigths.npy']
    for file in files:
        file_path = os.path.join(data_path, file)
        if os.path.exists(file_path) and not args.FORCE_DOWNLOAD:
            print('File ' + file_path + ' exists. Downloading data cancelled. If you want to force download use --force_download option')
        else:
            print('Downloading ' + file)
            urllib.request.urlretrieve(DATA_URL + file, file_path)
