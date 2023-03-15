import os,sys
from distutils.dir_util import copy_tree
import yaml

if len(sys.argv) != 2:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/exit.py data/store\n'
    )
    sys.exit(1)

params = yaml.safe_load(open('params.yaml'))

print("\n\n\n")
print("-------------------------------")
print("Exiting.....")
print("-------------------------------")
print("\n\n\n")

datastore = os.path.join(sys.argv[1],f"v{params['ingest']['dcount']}")

dcount = params['ingest']['dcount'] + 1
params.pop("ingest")
params.update({"ingest":{"dcount":dcount}})
with open('params.yaml', 'w') as file:
    outputs = yaml.dump(params, file)

print(datastore)
print("\n\n\nPipeline Executed-------------\n\n\n")