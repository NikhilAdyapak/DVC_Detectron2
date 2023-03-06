import os
import sys
import yaml
import pickle
from tqdm import tqdm
import cv2

if len(sys.argv) != 4:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/split.py data/prepared data/transformed data/split\n'
    )
    sys.exit(1)

params = yaml.safe_load(open('params.yaml'))
print("-------------------------------")
print("Splitting.....")
print("-------------------------------")

outputsplit = os.path.join(sys.argv[3],f"v{params['ingest']['dcount']}")
os.makedirs(outputsplit, exist_ok = True)