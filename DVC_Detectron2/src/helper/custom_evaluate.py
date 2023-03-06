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

import yaml
from tqdm import tqdm

from helper.xml_to_df import *


def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def iou_mapping(pred_box,gt_boxes,metrics):
    tp = metrics["TP"]
    fp = metrics["FP"]
    fn = metrics["FN"]
    iou_list = metrics["IOU"]
    overall_iou = []
    max_iou = 0
    for i in gt_boxes:
        single_iou = bb_intersection_over_union(pred_box,i)
        overall_iou.append(single_iou)
    if len(overall_iou) == 0:
        fn += 1
    else:
        max_iou = max(overall_iou)
        ind = overall_iou.index(max_iou)
        gt = gt_boxes[ind]
        gt_boxes.pop(ind)
    if max_iou == 0:
        fn += 1
    else:
        if max_iou > 0.7:
            tp += 1
        else:
            fp += 1
    iou_list.append(max_iou)
    return {"TP":tp,"FP":fp,"FN":fn,"IOU":iou_list}


def evaluate(metrics):
    tp = metrics["TP"]
    fp = metrics["FP"]
    fn = metrics["FN"]
    iou_list = metrics["IOU"]

    f1_score = 0
    prec = 0
    recall = 0
    iou_avg = 0
    try:
        iou_avg = sum(iou_list) / len(iou_list)
        prec = tp / float(tp + fp)
        recall = tp / float(tp + fn)
        f1_score = 2*prec*recall/(prec+recall)
    except ZeroDivisionError:
        print("ZeroDivisionError Occurred and Handled")
    
    return {"TP":tp,"FP":fp,"FN":fn,"Precision":prec,"Recall":recall,"F1":f1_score,"Avg_IOU":iou_avg}


def custom_evaluator(cache,params,metrics):
    # # Custom Evaluator

    cfg = cache["cfg"]
    predictor = cache["predictor"]
    dataset_name_train = cache["train"]
    dataset_name_test = cache["test"]
    results = cache["results"]

    annot_path = params['test']['annot_path']
    img_path = params['test']['img_path']
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


# params = yaml.safe_load(open('params.yaml'))

# annot_path = params['test']['annot_path']
# img_path = params['test']['img_path']

# dataframe = creatingInfoData(annot_path)
# dataset_name_train = params['train']['dataset_name']
# dataset_name_test = params['test']['dataset_name']


# cfg = get_cfg()
# # cfg.MODEL.DEVICE = 'cpu'
# cfg.merge_from_file(model_zoo.get_config_file(params['parameters']['config_file']))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(params['parameters']['config_file'])
# predictor = DefaultPredictor(cfg)

# gt_df = creatingInfoData(annot_path)

# metrics = {"TP":0,"FP":0,"FN":0,"IOU":[]}

# for filename in tqdm(os.listdir(annot_path)):
#     img = os.path.join(img_path, filename).replace("xml","jpg")
#     img = cv2.imread(img)
#     outputs = predictor(img)
#     outputs = predictor(img)
#     v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
#     boxes = v._convert_boxes(outputs["instances"][outputs["instances"].pred_classes == 0].pred_boxes.to('cpu'))
#     # print(boxes,type(boxes))
#     gt_df_sub = gt_df[gt_df["name"] == filename]
#     gt_boxes = []
#     for index, row in gt_df_sub.iterrows():
#         gt_boxes.append([row["xmin"],row["ymin"],row["xmax"],row["ymax"]])
#     # print(gt_boxes, type(gt_boxes))
#     out = v.draw_instance_predictions(outputs["instances"][outputs['instances'].pred_classes == 0].to("cpu"))
#     # cv2.imshow("img",out.get_image())
#     # cv2.waitKey(0)
#     for box in boxes:
#         metrics = iou_mapping(box,gt_boxes,metrics)

# evaluate(metrics)