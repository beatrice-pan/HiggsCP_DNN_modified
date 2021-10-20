import os
import argparse

parser = argparse.ArgumentParser(description='Download rhorhoZ data')
parser.add_argument("-d", "--datasets", dest="DATASETS", type=int, default='2', help="number of datasets to download")
parser.add_argument("-o", "--output", dest="OUT", default=os.environ["RHORHO_DATA"])
parser.add_argument("-s", "--source", dest="SOURCE", default=os.environ["DATA_SOURCE"])

args = parser.parse_args()

prefix = args.SOURCE
rhorho_data_path = args.OUT

for letter in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k"][:args.DATASETS]:
    name = "pythia.Z_65_155.rhorho.1M.%s.outTUPLE_labFrame" % (letter)
    if os.path.exists(os.path.join(rhorho_data_path, name)):
        print name, " is already downloaded! Skipping."
        continue
    link = prefix + name
    print link
    os.system("wget -P " + rhorho_data_path + " " + link)
