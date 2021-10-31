import numpy as np
import argparse
import os
import glob
from tqdm.notebook import tqdm

def find_first_line(lines, phrase):
    for i, line in enumerate(lines):
        if phrase in line: return i


def filter_Analysed(lines,phrase):
    index=[]
    for i, line in enumerate(lines):
        if phrase not in line:
            index.append(i)
        if 'total' in line:
            index.append(i)
    return np.array(index)

def read_raw_root(file, decaymode, num_features):
    with open(file) as f:
        lines = f.readlines()
    data = np.array(lines[find_first_line(lines, "TUPLE"):find_first_line(lines, "Analysed in total:")])
    lines=data[filter_Analysed(data,phrase='Analysed')].tolist()
    ids = np.array([int(idx) for idx, line in enumerate(lines) if line.startswith("TUPLE")])
    interval_ids=np.array(list(range(0, num_features[decaymode] * len(ids), num_features[decaymode])))
    lines = [line.strip() for line in lines]
    num_examples = len(ids)
    weights = [float(lines[num_features[decaymode] * i].strip().split()[1]) for i in range(num_examples)]
    weights = np.array(weights)

    values=[list(map(float, " ".join(lines[num_features[decaymode] * i + 1: num_features[decaymode] * (i + 1)]).split())) for i in range(num_examples)]
    values = np.array(values)

    return values, weights

def convert_bkgd_raw(bkgd_raw_path):
    num_features={'a1a1':9,'a1rho':8,'rhorho':7}
    for i in range(len(num_features)):
        all_data = []
        decaymode = list(num_features.keys())[i]
        for file_path in tqdm(glob.glob(bkgd_raw_path%(decaymode))): # return the file path
            data, weights = read_raw_root(file_path, decaymode, num_features)
            all_data += [data]
        all_data = np.concatenate(all_data)
        np.save('HiggsCP_data/'+decaymode+'/'+decaymode+'Z_raw.data.npy',all_data)
    print('Ztt raw data already converted into npy file :)')


if __name__ == "__main__":
    bkgd_raw_path = args.ZPATH
    convert_Ztt_raw(bkgd_raw_path)