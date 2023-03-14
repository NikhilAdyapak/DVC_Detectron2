import os
import sys
import yaml
import pickle
from tqdm import tqdm
import cv2
import fnmatch
import math

import splitfolders


if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/split.py data/prepared data/split\n'
    )
    sys.exit(1)


def numberOfFiles(params):
    dir_path = os.path.join(sys.argv[1],f"v{params['ingest']['dcount']}","Annotations")
    count = len(fnmatch.filter(os.listdir(dir_path), '*.*'))
    files = os.listdir(dir_path)
    print(files)
    return count


def makeBatches(path):
    batch_list = ['Infer','Batch1','Batch2','Batch3']
    for li in batch_list:
        parent_path = os.path.join(path,li)
        split_image_path = os.path.join(parent_path,"Annotations")
        split_annot_path = os.path.join(parent_path,"Images")
        os.makedirs(split_image_path,exist_ok=True)
        os.makedirs(split_annot_path,exist_ok=True)
    return


def split():
    return


def main():
    params = yaml.safe_load(open('params.yaml'))
    outputsplit = os.path.join(sys.argv[2],f"v{params['ingest']['dcount']}")

    input_path = os.path.join(sys.argv[1],f"v{params['ingest']['dcount']}")
    
    infer_batch = params['split']['val']
    train_batch = params['split']['train']
    # print(infer_batch)
    # print(train_batch)
    os.makedirs(outputsplit, exist_ok = True)

    print("\n\n\n")
    print("-------------------------------")
    print("Splitting.....")
    print("-------------------------------")
    print("\n\n\n")

    splitfolders.ratio(input_path, output = outputsplit, ratio = (train_batch, infer_batch), group_prefix = None, move = False)

    print("\n\n\n")
    print("-------------------------------")
    print("Splitting Completed....")
    print("-------------------------------")
    print("\n\n\n")

    
if __name__ == '__main__':
    main()
