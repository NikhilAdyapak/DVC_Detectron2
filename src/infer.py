import os
import sys
import yaml
import pickle
from tqdm import tqdm
import cv2

import warnings as wr
wr.filterwarnings("ignore")

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random, glob
import re
import xml.etree.ElementTree as ET
import pandas as pd
import torch

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import build_detection_test_loader
from detectron2.data import *
from detectron2.structures import Boxes

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

from detectron2.evaluation import inference_on_dataset

import yaml,shutil
from tqdm import tqdm

from helper.xml_to_df import *
from helper.custom_evaluate import *
from detectron2.data.datasets import register_coco_instances

if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/infer.py data/transform data/infer\n'
    )
    sys.exit(1)

params = yaml.safe_load(open('params.yaml'))

outputinfer = os.path.join(sys.argv[2],f"v{params['ingest']['dcount']}")
os.makedirs(outputinfer, exist_ok = True)

base_path = os.path.join(sys.argv[1],f"v{params['ingest']['dcount']}")


def custom_detectron(cache,params,metrics):
    # torch.cuda.empty_cache()

    cfg = cache["cfg"]
    predictor = cache["predictor"]
    dataset_name_train = cache["train"]
    dataset_name_val = cache["val"]
    results = cache["results"]

    # dataset_name_train = params['dataset']['name'] + "train"
    # dataset_name_val = params['dataset']['name'] + "val"
    dataset_name_train = "my_dataset_train"
    dataset_name_val = "my_dataset_val"

    register_coco_instances("my_dataset_train", {}, "/home/yln1kor/Dataset/karthika95/Train/_annotations.coco.json", "/home/yln1kor/Dataset/karthika95/Train")
    register_coco_instances("my_dataset_val", {}, "/home/yln1kor/Dataset/karthika95/Val/_annotations.coco.json", "/home/yln1kor/Dataset/karthika95/Val")

    my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
    dataset_dicts = DatasetCatalog.get("my_dataset_train")


    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow("out",vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)
    cv2.destroyAllWindows()


    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(params['parameters']['config_file']))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(params['parameters']['config_file'])
    predictor = DefaultPredictor(cfg)
    # # Detectron Evaluator
    evaluator = COCOEvaluator(dataset_name_val, cfg, False, output_dir = params["output"]["base_dir"])
    val_loader = build_detection_test_loader(cfg, dataset_name_val)
    try:
        eval_results = inference_on_dataset(predictor.model, val_loader, evaluator)
    except AssertionError:
        print("AssertionError Occurred and Handled")

    # d1 = next(iter(eval_results.items()))
    # d2 = next(iter(eval_results.values()))

    # d = {'AP':d2["AP"],'AP50':d2["AP50"],'AP75':d2["AP75"],'APs':d2['APs']}
    # s1 = json.dumps(d)
    # results = json.loads(s1)

    cache = {"cfg":cfg,"predictor":predictor,"train":dataset_name_train,"val":dataset_name_val,"results":results}
    (cache,metrics) = custom_evaluator(cache,params,metrics)
    metrics.update(d)
    cache = {"cfg":cfg,"predictor":predictor,"train":dataset_name_train,"val":dataset_name_val,"results":results}
    return (cache,metrics)


def custom_evaluator(cache,params,metrics):
    # # Custom Evaluator

    cfg = cache["cfg"]
    predictor = cache["predictor"]
    dataset_name_train = cache["train"]
    dataset_name_val = cache["val"]
    results = cache["results"]

    dataframe = pd.read_pickle(os.path.join(base_path, f"v{params['ingest']['dcount']}" + "_val.pkl"))
    annot_path = dataframe["name"]

    gt_df = creatingInfoData(annot_path)

    for filename in tqdm(os.listdir(annot_path)):
        # ext = filename.split(".")[1]
        # if ext == "xml":
        #     img = os.path.join(img_path, filename).replace("xml","jpg")
        # else:
        #     img = os.path.join(img_path, filename).replace("txt","jpg")
        img = filename.replace("Annotations","Images") + ".jpg"
        img = cv2.imread(img)
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        boxes = v._convert_boxes(outputs["instances"].pred_boxes.to('cpu'))
        # print(boxes,type(boxes))
        gt_df_sub = gt_df[gt_df["name"] == filename]
        gt_boxes = []
        for index, row in gt_df_sub.iterrows():
            gt_boxes.append([row["xmin"],row["ymin"],row["xmax"],row["ymax"]])
        # print(gt_boxes, type(gt_boxes))
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        for box in boxes:
            metrics = iou_mapping(box,gt_boxes,metrics)

    # for index, row in dataframe.iterrows():
    #     annot_path = row["name"]
    #     img = annot_path.replace("Annotations","Images") + ".jpg"
    #     img = cv2.imread(img)
    #     outputs = predictor(img)
    #     v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    #     boxes = v._convert_boxes(outputs["instances"].pred_boxes.to('cpu'))
    #     # print(boxes,type(boxes))
    #     gt_df_sub = dataframe

    metrics = evaluate(metrics)
    cache = {"cfg":cfg,"predictor":predictor,"train":dataset_name_train,"val":dataset_name_val,"results":results}
    return (cache,metrics)


def results_logger_pretrained(cache,params,metrics):
    # base_path = params['output']['results_pretrained']
    # results_old = max(glob.glob(os.path.join(base_path, '*/')), key = os.path.getmtime)
    # run_num = int((results_old.split("/")[1]).split("_")[1]) + 1
    # new_results = os.path.join(base_path,"results_" + str(run_num))
    # print(new_results)

    hyper_parameters = {
        "config_file" : params['parameters']['config_file'],
    }

    metrics = {
        "tp" : metrics["TP"],
        "fp" : metrics["FP"],
        "fn" : metrics["FN"],
        "Precision" : metrics["Precision"],
        "Recall" : metrics["Recall"],
        "F1" : metrics["F1"],
        "Avg_IOU" : metrics["Avg_IOU"],
        'AP':metrics["AP"],
        'AP50':metrics["AP50"],
        'AP75':metrics["AP75"],
        'APs':metrics['APs']
    }

    # os.mkdir(new_results)

    with open(os.path.join(outputinfer,'hyperparamters_{}.txt'.format(params['ingest']['dcount'])), 'w') as fout:
        fout.write(json.dumps(hyper_parameters, indent = len(hyper_parameters)))

    with open(os.path.join(outputinfer,'metrics_{}.txt'.format(params['ingest']['dcount'])), 'w') as fout:
        fout.write(json.dumps(metrics, indent = len(metrics)))


if __name__ == "__main__":
    print("-------------------------------")
    print("Inferencing.....")
    print("-------------------------------")

    cache = dict()
    metrics = dict()
    cache = {"cfg":None,"predictor":None,"train":None,"val":None,"results":None}
    metrics = {"TP":0,"FP":0,"FN":0,"IOU":[]}
    (cache,metrics) = custom_detectron(cache,params,metrics)
    results_logger_pretrained(cache,params,metrics)

    # df = pd.read_pickle(file_name)