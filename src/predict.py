import os,sys
from distutils.dir_util import copy_tree
import yaml

if len(sys.argv) != 4:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/predict.py data/train data/predict data/store\n'
    )
    sys.exit(1)


params = yaml.safe_load(open('params.yaml'))

print("-------------------------------")
print("Predicting.....")
print("-------------------------------")

outputpred = os.path.join(sys.argv[2],f"v{params['ingest']['dcount']}")
os.makedirs(outputpred, exist_ok = True)

input_dir = os.path.join(sys.argv[1],f"v{params['ingest']['dcount']}")

copy_tree(input_dir, outputpred)