# import os
# import yaml
# import zipfile
# import sys

# params = yaml.safe_load(open('params.yaml'))['ingest']
# data_path = os.path.join('data', 'prepared', f"v{params['dcount']}")
# origimg_path = os.path.join('data', 'store', f"v{params['dcount']}")
# # print(data_path)
# os.makedirs(data_path, exist_ok=True)
# os.makedirs(origimg_path, exist_ok=True)
# print("-------------------------------")
# print("Extracting data.....")
# print("-------------------------------")
# sys.path.append('../')
# with zipfile.ZipFile(f'buffer/dataset{params["dcount"]}.zip',"r") as zipf:
#     zipf.extractall(data_path)
#     zipf.extractall(origimg_path)

import os,sys
from distutils.dir_util import copy_tree
import yaml

params = yaml.safe_load(open('params.yaml'))
data_path = os.path.join('data', 'prepared', f"v{params['ingest']['dcount']}")
origimg_path = os.path.join('data', 'store', f"v{params['ingest']['dcount']}")
print(data_path,'\n',origimg_path)

input_dir = params["dataset"]["path"]
os.makedirs(data_path, exist_ok = True)
os.makedirs(origimg_path, exist_ok = True)

print("-------------------------------")
print("Ingesting Data.....")
print("-------------------------------")

copy_tree(input_dir, data_path)
copy_tree(input_dir, origimg_path)