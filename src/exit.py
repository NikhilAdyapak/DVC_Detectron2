import os,sys
from distutils.dir_util import copy_tree
import shutil
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

runs_path = "runs"
os.makedirs(runs_path, exist_ok = True)

train_path = os.path.join(runs_path,"train/expt{}".format(params['ingest']['dcount']))
os.makedirs(train_path, exist_ok = True)
train_weights_path = os.path.join(train_path,"weights")
os.makedirs(train_weights_path, exist_ok = True)
val_path = os.path.join(runs_path,"val/expt{}".format(params['ingest']['dcount']))
os.makedirs(val_path, exist_ok = True)

if os.path.exists("data/train/v{}/model_final.pth".format(params['ingest']['dcount'])):
    shutil.copy("data/train/v{}/model_final.pth".format(params['ingest']['dcount']), train_weights_path)

shutil.copy("data/train/v{}/metrics.json".format(params['ingest']['dcount']), train_path)
shutil.copy("data/predict/v{}/predict_metrics.json".format(params['ingest']['dcount']), train_path)
shutil.copy("data/predict/v{}/det2_hyperparamters.json".format(params['ingest']['dcount']), train_path)

shutil.copy("data/predict/v{}/predict_metrics.json".format(params['ingest']['dcount']), val_path)
shutil.copy("data/predict/v{}/det2_hyperparamters.json".format(params['ingest']['dcount']), val_path)


dcount = params['ingest']['dcount'] + 1
params.pop("ingest")
params.update({"ingest":{"dcount":dcount}})
with open('params.yaml', 'w') as file:
    outputs = yaml.dump(params, file)

print(datastore)
print("\n\n\nPipeline Executed-------------\n\n\n")