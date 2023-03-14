import os,sys
from distutils.dir_util import copy_tree
import yaml

if len(sys.argv) != 4:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/exit.py data/infer data/predict data/store\n'
    )
    sys.exit(1)

params = yaml.safe_load(open('params.yaml'))

print("\n\n\n")
print("-------------------------------")
print("Exiting.....")
print("-------------------------------")
print("\n\n\n")

datastore = os.path.join(sys.argv[3],f"v{params['ingest']['dcount']}")
pred = os.path.join(sys.argv[2],f"v{params['ingest']['dcount']}")
infer = os.path.join(sys.argv[1],f"v{params['ingest']['dcount']}")

copy_tree(pred, datastore)
copy_tree(infer, datastore)

print(datastore)
print("\n\n\nPipeline Executed-------------\n\n\n")