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

# from helper.xml_to_df import *
from helper.custom_evaluate import *
from helper.my_logger import *

from detectron2.data.datasets import register_coco_instances

if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/infer.py data/transform data/infer\n'
    )
    sys.exit(1)

params = yaml.safe_load(open('params.yaml'))

transform_path = os.path.join(sys.argv[1],f"v{params['ingest']['dcount']}")
output_infer = os.path.join(sys.argv[2],f"v{params['ingest']['dcount']}")
os.makedirs(output_infer, exist_ok = True)
annot_path = os.path.join("data/split",f"v{params['ingest']['dcount']}","val/Annotations")
img_path = os.path.join("data/split",f"v{params['ingest']['dcount']}","val/Images")
val_path = params['validation']['path']


def creatingInfoData(Annotpath):
    xml_list = []
    for xml_file in sorted(glob.glob(str(Annotpath+'/*.xml*'))):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            file_name = xml_file.split('/')[-1][0:-4]
            value = (
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text),
                     file_name,
                     "person",
                     )
            xml_list.append(value)
    column_name = ['xmin', 'ymin', 'xmax', 'ymax', 'name', 'label']
    xml_df = pd.DataFrame(xml_list, columns = column_name)
    return xml_df


def custom_dataset_function_test():
    # file_name, height, width, image_id
    #[{'file_name': '/home/samjith/0000180.jpg', 'height': 788, 'width': 1400, 'image_id': 1, 
    #   'annotations': [{'bbox': [250.0, 675.0, 23.0, 17.0], 'bbox_mode': <BoxMode.XYWH_ABS: 1>, 'area': 391.0, 'segmentation': [],
    #        'category_id': 0}, {'bbox': [295.0, 550.0, 21.0, 20.0], 'bbox_mode': <BoxMode.XYWH_ABS: 1>, 'area': 420.0, 'segmentation': [], 'category_id': 0},..

    annot_path = os.path.join("data/split",f"v{params['ingest']['dcount']}","val/Annotations")
    img_path = os.path.join("data/split",f"v{params['ingest']['dcount']}","val/Images")

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


def custom_dataset_function_val():
    # file_name, height, width, image_id
    #[{'file_name': '/home/samjith/0000180.jpg', 'height': 788, 'width': 1400, 'image_id': 1, 
    #   'annotations': [{'bbox': [250.0, 675.0, 23.0, 17.0], 'bbox_mode': <BoxMode.XYWH_ABS: 1>, 'area': 391.0, 'segmentation': [],
    #        'category_id': 0}, {'bbox': [295.0, 550.0, 21.0, 20.0], 'bbox_mode': <BoxMode.XYWH_ABS: 1>, 'area': 420.0, 'segmentation': [], 'category_id': 0},..

    annot_path = os.path.join(val_path,"Annotations")
    img_path = os.path.join(val_path,"Images")
    
    dataframe = creatingInfoData(annot_path)

    old_fname = os.path.join(img_path, dataframe["name"][0]) + ".jpg"
    annotations = []
    dataset = []
    for index,row in dataframe.iterrows():
        fname = os.path.join(img_path, row["name"]) + ".jpg"
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


def detectron_custom_infer():
    # register_coco_instances("my_dataset_train", {}, os.path.join(transform_path,"_annotations_train.coco.json"), os.path.join("data/split",f"v{params['ingest']['dcount']}","train/Images"))
    # register_coco_instances("my_dataset_val", {}, os.path.join(transform_path,"_annotations_val.coco.json"), os.path.join("data/split",f"v{params['ingest']['dcount']}","val/Images"))

    # my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
    # dataset_dicts = DatasetCatalog.get("my_dataset_train")

    # for d in random.sample(dataset_dicts, 3):
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img[:, :, ::-1], metadata = my_dataset_train_metadata, scale = 0.5)
    #     vis = visualizer.draw_dataset_dict(d)
    #     cv2.imshow("out",vis.get_image()[:, :, ::-1])
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()

    DatasetCatalog.register("my_dataset_val", custom_dataset_function_test)
    MetadataCatalog.get("my_dataset_val").set(thing_classes = ["person"])

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(params['detectron_parameters']['config_file']))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(params['detectron_parameters']['config_file'])
    predictor = DefaultPredictor(cfg)

    test_metadata = MetadataCatalog.get("my_dataset_val")
    dataset_dicts = DatasetCatalog.get("my_dataset_val")
    for d in random.sample(dataset_dicts, 10):
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], metadata = test_metadata, scale = 0.5)
        # vis = visualizer.draw_dataset_dict(d)
        out = v.draw_instance_predictions(outputs["instances"][outputs['instances'].pred_classes == 0].to("cpu"))
        # cv2_imshow(vis.get_image()[:, :, ::-1])
        cv2.imshow("output",out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
    cv2.destroyAllWindows()

    # cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file(params['detectron_parameters']['config_file']))
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(params['detectron_parameters']['config_file'])
    # # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    # # MetadataCatalog.get("my_dataset_val").set(thing_classes = ['persons', 'person', 'person-like'])
    
    # predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir = output_infer)
    val_loader = build_detection_test_loader(cfg, "my_dataset_val")
    eval_results = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(eval_results)

    d1 = next(iter(eval_results.items()))
    d2 = next(iter(eval_results.values()))
    det_metrics = {'AP':d2["AP"],'AP50':d2["AP50"],'AP75':d2["AP75"],'APs':d2['APs'],'APm':d2['APm'],'APl':d2['APl']}
    # det_metrics = {'AP':0,'AP50':0,'AP75':0,'APs':0}
    # s1 = json.dumps(d)
    # results = json.loads(s1)

    annot_path = os.path.join("data/split",f"v{params['ingest']['dcount']}","val/Annotations")
    
    gt_df = creatingInfoData(annot_path)
    gt_df["name"] = [x["name"].split("/")[-1] for index,x in gt_df.iterrows()]
    # print(gt_df)

    metrics = {"TP":0,"FP":0,"FN":0,"IOU":[]}

    for filename in tqdm(os.listdir(annot_path)):
        img = os.path.join(img_path, filename).replace("xml","jpg")
        img = cv2.imread(img)
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        boxes = v._convert_boxes(outputs["instances"].pred_boxes.to('cpu'))
        gt_df_sub = gt_df[gt_df["name"] == filename.split(".")[0]]
        gt_boxes = []
        for index, row in gt_df_sub.iterrows():
            gt_boxes.append([row["xmin"],row["ymin"],row["xmax"],row["ymax"]])
        for box in boxes:
            metrics = iou_mapping(box,gt_boxes,metrics)

    my_metrics = evaluate(metrics)

    # Master Validation for compare

    DatasetCatalog.register("master_dataset_val", custom_dataset_function_val)
    MetadataCatalog.get("master_dataset_val").set(thing_classes = ["person"])

    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("master_dataset_val", cfg, False, output_dir = output_infer)
    val_loader = build_detection_test_loader(cfg, "master_dataset_val")
    eval_results = inference_on_dataset(predictor.model, val_loader, evaluator)

    d1 = next(iter(eval_results.items()))
    d2 = next(iter(eval_results.values()))
    det_metrics_val = {'AP':d2["AP"],'AP50':d2["AP50"],'AP75':d2["AP75"],'APs':d2['APs'],'APm':d2['APm'],'APl':d2['APl']}

    annot_path = os.path.join(val_path,"Annotations")
    img_path_val = os.path.join(val_path,"Images")
    for filename in tqdm(os.listdir(annot_path)):
        img = os.path.join(img_path_val, filename).replace("xml","jpg")
        img = cv2.imread(img)
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        boxes = v._convert_boxes(outputs["instances"].pred_boxes.to('cpu'))
        gt_df_sub = gt_df[gt_df["name"] == filename.split(".")[0]]
        gt_boxes = []
        for index, row in gt_df_sub.iterrows():
            gt_boxes.append([row["xmin"],row["ymin"],row["xmax"],row["ymax"]])
        for box in boxes:
            metrics = iou_mapping(box,gt_boxes,metrics)

    my_metrics_val = evaluate(metrics)

    results_logger(det_metrics,my_metrics,det_metrics_val,my_metrics_val,output_infer,params)


if __name__ == "__main__":
    print("-------------------------------")
    print("Inferencing.....")
    print("-------------------------------")
    detectron_custom_infer()
    print("\n\n\n")
    print("-------------------------------")
    print("Inferencing Completed.....")
    print("-------------------------------")


    # df = pd.read_pickle(file_name)