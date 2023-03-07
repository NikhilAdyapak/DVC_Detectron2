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

if len(sys.argv) != 5:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/infer.py data/prepared data/transformed data/infer data/store\n'
    )
    sys.exit(1)

params = yaml.safe_load(open('params.yaml'))
outputinfer = os.path.join(sys.argv[3],f"v{params['ingest']['dcount']}")
os.makedirs(outputinfer, exist_ok = True)

base_path = os.path.join(sys.argv[1],f"v{params['ingest']['dcount']}")


def custom_dataset_function_train():
    # file_name, height, width, image_id
    #[{'file_name': '/home/samjith/0000180.jpg', 'height': 788, 'width': 1400, 'image_id': 1, 
    #   'annotations': [{'bbox': [250.0, 675.0, 23.0, 17.0], 'bbox_mode': <BoxMode.XYWH_ABS: 1>, 'area': 391.0, 'segmentation': [],
    #        'category_id': 0}, {'bbox': [295.0, 550.0, 21.0, 20.0], 'bbox_mode': <BoxMode.XYWH_ABS: 1>, 'area': 420.0, 'segmentation': [], 'category_id': 0},..

    # annot_path = params['train']['annot_path']
    # img_path = params['train']['img_path']

    annot_path = os.path.join(base_path, params['train']['annot_path'])
    img_path = os.path.join(base_path, params['train']['img_path'])

    dataframe = creatingInfoData(annot_path)
    old_fname = os.path.join(img_path, dataframe["name"][0].split('.')[0] + ".jpg")
    annotations = []
    dataset = []
    for index,row in dataframe.iterrows():
        fname = os.path.join(img_path, row["name"].split('.')[0] + ".jpg")
        xmin = row["xmin"]
        ymin = row["ymin"]
        xmax = row["xmax"]
        ymax = row["ymax"]

        if old_fname != fname:
            img = cv2.imread(old_fname)
            dataset.append(
                        {"file_name":old_fname , 
                        "height":img.shape[0], 
                        "width":img.shape[1],
                        "image_id":re.findall(r'\d+', old_fname)[0],
                        "annotations":annotations})
            annotations = []

        annotations.append(
            {"bbox":[xmin,ymin,xmax,ymax],
            'bbox_mode': 0, 
            'area': (xmax - xmin) * (ymax - ymin), 
            'segmentation': [],
            'category_id':0})
     
        old_fname = fname

    img = cv2.imread(old_fname)
    dataset.append(
                        {"file_name":old_fname , 
                        "height":img.shape[0], 
                        "width":img.shape[1],
                        "image_id":re.findall(r'\d+', old_fname)[0],
                        "annotations":annotations})

    return dataset


def custom_dataset_function_test():
    # file_name, height, width, image_id
    #[{'file_name': '/home/samjith/0000180.jpg', 'height': 788, 'width': 1400, 'image_id': 1, 
    #   'annotations': [{'bbox': [250.0, 675.0, 23.0, 17.0], 'bbox_mode': <BoxMode.XYWH_ABS: 1>, 'area': 391.0, 'segmentation': [],
    #        'category_id': 0}, {'bbox': [295.0, 550.0, 21.0, 20.0], 'bbox_mode': <BoxMode.XYWH_ABS: 1>, 'area': 420.0, 'segmentation': [], 'category_id': 0},..

    annot_path = os.path.join(base_path, params['test']['annot_path'])
    img_path = os.path.join(base_path, params['test']['img_path'])

    dataframe = creatingInfoData(annot_path)
    old_fname = os.path.join(img_path, dataframe["name"][0].split('.')[0] + ".jpg")
    annotations = []
    dataset = []
    for index,row in dataframe.iterrows():
        fname = os.path.join(img_path, row["name"].split('.')[0] + ".jpg")
        xmin = row["xmin"]
        ymin = row["ymin"]
        xmax = row["xmax"]
        ymax = row["ymax"]
        if old_fname != fname:
            img = cv2.imread(old_fname)
            dataset.append(
                        {"file_name":old_fname , 
                        "height":img.shape[0], 
                        "width":img.shape[1],
                        "image_id":re.findall(r'\d+', old_fname)[0],
                        "annotations":annotations})
            annotations = []
        annotations.append(
            {"bbox":[xmin,ymin,xmax,ymax],
            'bbox_mode': 0, 
            'area': (xmax - xmin) * (ymax - ymin), 
            'segmentation': [],
            'category_id':0})
        old_fname = fname

    img = cv2.imread(old_fname)
    dataset.append(
                        {"file_name":old_fname , 
                        "height":img.shape[0], 
                        "width":img.shape[1],
                        "image_id":re.findall(r'\d+', old_fname)[0],
                        "annotations":annotations})

    return dataset


def custom_detectron(cache,params,metrics):
    # torch.cuda.empty_cache()

    cfg = cache["cfg"]
    predictor = cache["predictor"]
    dataset_name_train = cache["train"]
    dataset_name_test = cache["test"]
    results = cache["results"]

    dataset_name_train = params['train']['dataset_name']
    dataset_name_test = params['test']['dataset_name']

    DatasetCatalog.register(dataset_name_train, custom_dataset_function_train)
    MetadataCatalog.get(dataset_name_train).set(thing_classes = ["person"])
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(params['parameters']['config_file']))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(params['parameters']['config_file'])
    predictor = DefaultPredictor(cfg)
    DatasetCatalog.register(dataset_name_test, custom_dataset_function_test)
    MetadataCatalog.get(dataset_name_test).set(thing_classes = ["person"])

    my_dataset_test_metadata = MetadataCatalog.get(dataset_name_test).set(thing_classes = ["person"])
    # from detectron2.utils.visualizer import ColorMode
    dataset_dicts = DatasetCatalog.get(dataset_name_test)
    for d in random.sample(dataset_dicts, 5):    
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], metadata = my_dataset_test_metadata, scale = 0.5)
        # vis = visualizer.draw_dataset_dict(d)
        out = v.draw_instance_predictions(outputs["instances"][outputs['instances'].pred_classes == 0].to("cpu"))
        # cv2_imshow(vis.get_image()[:, :, ::-1])
        cv2.imshow("output",out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
    cv2.destroyAllWindows()


    # # Detectron Evaluator
    evaluator = COCOEvaluator(dataset_name_test, cfg, False, output_dir = params["output"]["base_dir"])
    val_loader = build_detection_test_loader(cfg, dataset_name_test)
    eval_results = inference_on_dataset(predictor.model, val_loader, evaluator)

    d1 = next(iter(eval_results.items()))
    d2 = next(iter(eval_results.values()))

    d = {'AP':d2["AP"],'AP50':d2["AP50"],'AP75':d2["AP75"],'APs':d2['APs']}
    s1 = json.dumps(d)
    results = json.loads(s1)

    cache = {"cfg":cfg,"predictor":predictor,"train":dataset_name_train,"test":dataset_name_test,"results":results}
    (cache,metrics) = custom_evaluator(cache,params,metrics)
    metrics.update(d)
    cache = {"cfg":cfg,"predictor":predictor,"train":dataset_name_train,"test":dataset_name_test,"results":results}
    return (cache,metrics)


def custom_evaluator(cache,params,metrics):
    # # Custom Evaluator

    cfg = cache["cfg"]
    predictor = cache["predictor"]
    dataset_name_train = cache["train"]
    dataset_name_test = cache["test"]
    results = cache["results"]

    annot_path = params['test']['annot_path']
    img_path = params['test']['img_path']

    annot_path = os.path.join(base_path, params['test']['annot_path'])
    img_path = os.path.join(base_path, params['test']['img_path'])

    gt_df = creatingInfoData(annot_path)

    for filename in tqdm(os.listdir(annot_path)):
        img = os.path.join(img_path, filename).replace("xml","jpg")
        img = cv2.imread(img)
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        boxes = v._convert_boxes(outputs["instances"][outputs["instances"].pred_classes == 0].pred_boxes.to('cpu'))
        # print(boxes,type(boxes))
        gt_df_sub = gt_df[gt_df["name"] == filename]
        gt_boxes = []
        for index, row in gt_df_sub.iterrows():
            gt_boxes.append([row["xmin"],row["ymin"],row["xmax"],row["ymax"]])
        # print(gt_boxes, type(gt_boxes))
        out = v.draw_instance_predictions(outputs["instances"][outputs['instances'].pred_classes == 0].to("cpu"))

        for box in boxes:
            metrics = iou_mapping(box,gt_boxes,metrics)

    metrics = evaluate(metrics)
    cache = {"cfg":cfg,"predictor":predictor,"train":dataset_name_train,"test":dataset_name_test,"results":results}
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


print("-------------------------------")
print("Inferencing.....")
print("-------------------------------")

cache = dict()
metrics = dict()
cache = {"cfg":None,"predictor":None,"train":None,"test":None,"results":None}
metrics = {"TP":0,"FP":0,"FN":0,"IOU":[]}
(cache,metrics) = custom_detectron(cache,params,metrics)
results_logger_pretrained(cache,params,metrics)