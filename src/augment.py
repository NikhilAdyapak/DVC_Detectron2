import os,sys
from distutils.dir_util import copy_tree
import yaml

if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/augment.py data/prepared data/augmented\n'
    )
    sys.exit(1)

params = yaml.safe_load(open('params.yaml'))
print("-------------------------------")
print("Augmenting.....")
print("-------------------------------")

outputaug = os.path.join(sys.argv[2],f"v{params['ingest']['dcount']}")
os.makedirs(outputaug, exist_ok = True)