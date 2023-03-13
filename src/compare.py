import yaml,shutil
import json
import sys,os

if len(sys.argv) != 4:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/compare.py data/infer data/predict data/store\n'
    )
    sys.exit(1)


params = yaml.safe_load(open('params.yaml'))

infer_path = os.path.join(sys.argv[1],f"v{params['ingest']['dcount']}")
predict_path = os.path.join(sys.argv[2],f"v{params['ingest']['dcount']}")
store_path = os.path.join(sys.argv[3],f"v{params['ingest']['dcount']}")
os.makedirs(store_path, exist_ok = True)

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
    if best_f1 > new_f1 and best_ap > new_ap:
        print("Best model is same - ",best_model)
    else:
        print("New Best model - ", f"v{params['ingest']['dcount']}")



if __name__ == "__main__":
    # best_model_path = os.path.json(store_path,best_model)

    metrics_best_path = os.path.join(infer_path,"metrics_{}.json".format(params['ingest']['dcount']))
    # metrics_best_path = os.path.join(best_model_path,"metrics.json")
    metrics_new_path = os.path.join(predict_path,"metrics_{}.json".format(params['ingest']['dcount']))

    # metrics_best = json.loads(metrics_best_path)
    # metrics_new = json.load(metrics_new_path)

    f1 = open (metrics_best_path, "r")
    metrics_best = json.loads(f1.read())

    f2 = open (metrics_new_path, "r")
    metrics_new = json.loads(f2.read())

    print("-------------------------------")
    print("Comparing.....")
    print("-------------------------------")

    compare(metrics_best, metrics_new)

    print("-------------------------------")
    print("Comparing Completed.....")
    print("-------------------------------")