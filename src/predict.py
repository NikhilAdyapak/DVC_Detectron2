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
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

import random
from detectron2.utils.visualizer import Visualizer

import yaml,shutil
from tqdm import tqdm

from helper.xml_to_df import *
from helper.my_evaluate import *
from helper.my_logger import *

if len(sys.argv) != 4:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/predict.py data/train data/transform data/predict\n'
    )
    sys.exit(1)


params = yaml.safe_load(open('params.yaml'))

train_path = os.path.join(sys.argv[1],f"v{params['ingest']['dcount']}")
transform_path = os.path.join(sys.argv[2],f"v{params['ingest']['dcount']}")
output_pred = os.path.join(sys.argv[3],f"v{params['ingest']['dcount']}")
os.makedirs(output_pred, exist_ok = True)


def predict():

    register_coco_instances("my_dataset_val", {}, os.path.join(transform_path,"_annotations_val.coco.json"), os.path.join("data/split",f"v{params['ingest']['dcount']}","val/Images"))

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(params["detectron_parameters"]["config_file"]))
    cfg.OUTPUT_DIR = train_path
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = params["detectron_parameters"]["SCORE_THRESH_TEST"]
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir = "./output/")
    val_loader = build_detection_test_loader(cfg, "my_dataset_val")
    # eval_results = inference_on_dataset(predictor.model, val_loader, evaluator)

    # for d in glob.glob(os.path.join("data/split",f"v{params['ingest']['dcount']}","val/Images/*jpg")):    
    #     img = cv2.imread(d)
    #     outputs = predictor(img)
    #     v = Visualizer(img[:, :, ::-1], metadata = test_metadata, scale = 0.5)
    #     # vis = visualizer.draw_dataset_dict(d)
    #     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #     # cv2_imshow(vis.get_image()[:, :, ::-1])
    #     cv2.imshow("output",out.get_image()[:, :, ::-1])
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # d1 = next(iter(eval_results.items()))
    # d2 = next(iter(eval_results.values()))
    # det_metrics = {'AP':d2["AP"],'AP50':d2["AP50"],'AP75':d2["AP75"],'APs':d2['APs']}
    det_metrics = {'AP':0,'AP50':0,'AP75':0,'APs':0}
    # s1 = json.dumps(d)
    # results = json.loads(s1)


    annot_path = os.path.join("data/split",f"v{params['ingest']['dcount']}","val/Annotations")
    img_path = os.path.join("data/split",f"v{params['ingest']['dcount']}","val/Images")
    
    gt_df = creatingInfoData(annot_path)
    gt_df["name"] = [x["name"].split("/")[-1] for index,x in gt_df.iterrows()]

    metrics = {"TP":0,"FP":0,"FN":0,"IOU":[]}

    for filename in tqdm(os.listdir(annot_path)):
        img = os.path.join(img_path, filename).replace("xml","jpg")
        img = cv2.imread(img)
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        boxes = v._convert_boxes(outputs["instances"][outputs["instances"].pred_classes == 0].pred_boxes.to('cpu'))
        gt_df_sub = gt_df[gt_df["name"] == filename.split(".")[0]]
        gt_boxes = []
        for index, row in gt_df_sub.iterrows():
            gt_boxes.append([row["xmin"],row["ymin"],row["xmax"],row["ymax"]])
        out = v.draw_instance_predictions(outputs["instances"][outputs['instances'].pred_classes == 0].to("cpu"))

        for box in boxes:
            metrics = iou_mapping(box,gt_boxes,metrics)

    my_metrics = evaluate(metrics)
    results_logger(det_metrics,my_metrics,output_pred,params)


if __name__ == "__main__":

    print("-------------------------------")
    print("Predicting.....")
    print("-------------------------------")

    predict()

    print("-------------------------------")
    print("Predicting Completed.....")
    print("-------------------------------")