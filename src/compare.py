import yaml,shutil
import json
import sys,os
import re 
from distutils.dir_util import copy_tree

if len(sys.argv) != 4:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/compare.py data/infer data/predict data/store\n'
    )
    sys.exit(1)


params = yaml.safe_load(open('params.yaml'))

infer_path = os.path.join(sys.argv[1],f"v{params['ingest']['dcount']}")
predict_path = os.path.join(sys.argv[2],f"v{params['ingest']['dcount']}")
datastore = os.path.join(sys.argv[3],f"v{params['ingest']['dcount']}")
os.makedirs(datastore, exist_ok = True)

best_model = params['version']['best']


def compare(metrics_best,metrics_new):
    best_f1 = metrics_best["F1"]
    best_ap = metrics_best["AP"]
    new_f1 = metrics_new["F1"]
    new_ap = metrics_new["AP"]
    print("best_f1",best_f1)
    print("new_f1",new_f1)
    print("best_ap",best_ap)
    print("new_ap",new_ap)
    # if best_f1 > new_f1 and best_ap > new_ap:
    #     if infer_flag:
    #         print("Best model", best_model + " Pre-trained weights")
    #         best_model = best_model + "-infer"
    #     else:
    #         print("Best model", best_model)
    #         best_model = best_model
    # else:
    #     print("New Best model - ", f"v{params['ingest']['dcount']}")
    #     best_model = "v{}".format(params['ingest']['dcount'])
    
    pop_flag = True
    if best_f1 > new_f1 and best_ap > new_ap:
        if best_model == 'v0-infer' or best_model == "VD":
            best_model = "VD"
            print("Best Model - Detectron PreLoaded Weights: ",params['detectron_parameters']['config_file'])
        else:
            print("Best Model still is - ", best_model)
            pop_flag = False
    else:
        print("New Best Model - ", params['ingest']['dcount'])
        best_model = "v{}".format(params['ingest']['dcount'])

    if pop_flag:
        params.pop("version")
        params.update({"version":{"best":best_model}})
        with open('params.yaml', 'w') as file:
            outputs = yaml.dump(params, file)

    f = open(os.path.join(datastore,"best_model.txt"), "w")
    f.write(best_model)
    f.close()


if __name__ == "__main__":
    
    # infer_flag = False
    # pattern = re.compile(r'v\d-infer')
    # if pattern.search(best_model):
    #     dig = re.findall(r'\d+', best_model)[0]
    #     metrics_best_path = os.path.join(sys.argv[2],"v{}".format(dig),"metrics_{}.json".format(dig))
    #     infer_flag = True
    # else:
    #     best_model_path = os.path.join(sys.argv[2],best_model)
    #     metrics_best_path = os.path.join(best_model_path,"metrics_{}.json".format(params['ingest']['dcount']))

    # metrics_new_path = os.path.join(predict_path,"metrics_{}.json".format(params['ingest']['dcount']))

    metrics_best_path = os.path.join(infer_path,"predict_metrics.json")
    metrics_new_path = os.path.join(predict_path,"predict_metrics.json")


    f1 = open (metrics_best_path, "r")
    metrics_best = json.loads(f1.read())

    f2 = open (metrics_new_path, "r")
    metrics_new = json.loads(f2.read())

    print("\n\n\n")
    print("-------------------------------")
    print("Comparing.....")
    print("-------------------------------")
    print("\n\n\n")

    # compare(metrics_best, metrics_new, infer_flag)
    compare(metrics_best, metrics_new)

    datastore_predict = os.path.join(datastore,"predict")
    datastore_infer = os.path.join(datastore,"infer")
    os.makedirs(datastore_predict, exist_ok = True)
    os.makedirs(datastore_infer, exist_ok = True)
    copy_tree(predict_path, datastore_predict)
    copy_tree(infer_path, datastore_infer)

    print("\n\n\n")
    print("-------------------------------")
    print("Comparing Completed.....")
    print("-------------------------------")
    print("\n\n\n")