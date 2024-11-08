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

from helper.custom_evaluate import *
# from helper.txt_to_df import *

if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/train.py data/transform data/train\n'
    )
    sys.exit(1)

params = yaml.safe_load(open('params.yaml'))
transform_path = os.path.join(sys.argv[1],f"v{params['ingest']['dcount']}")
output_train = os.path.join(sys.argv[2],f"v{params['ingest']['dcount']}")
os.makedirs(output_train, exist_ok = True)


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


def aug_img_df(Annotpath):
    aug_list = []
    for files in sorted(glob.glob(str(Annotpath+'/*.txt*'))):
        with open(files, "r") as f:
            bbox = (f.read()).split('\n')
        for data in bbox[0:-1]:
            data = data.split()
            value = (
                float(data[0]),
                float(data[1]),
                float(data[2]),
                float(data[3]),
                files.split(".")[0],
                "person",
            )
            aug_list.append(value)
    column_name = ['xmin', 'ymin', 'xmax', 'ymax', 'name', 'label']
    aug_df = pd.DataFrame(aug_list, columns = column_name)
    return aug_df


def custom_dataset_function_train():
    # file_name, height, width, image_id
    #[{'file_name': '/home/samjith/0000180.jpg', 'height': 788, 'width': 1400, 'image_id': 1, 
    #   'annotations': [{'bbox': [250.0, 675.0, 23.0, 17.0], 'bbox_mode': <BoxMode.XYWH_ABS: 1>, 'area': 391.0, 'segmentation': [],
    #        'category_id': 0}, {'bbox': [295.0, 550.0, 21.0, 20.0], 'bbox_mode': <BoxMode.XYWH_ABS: 1>, 'area': 420.0, 'segmentation': [], 'category_id': 0},..

    annot_path = os.path.join("data/split",f"v{params['ingest']['dcount']}","train/Annotations")
    img_path = os.path.join("data/split",f"v{params['ingest']['dcount']}","train/Images")
    aug_annot_path = os.path.join("data/augmented",f"v{params['ingest']['dcount']}","Annotations")
    aug_img_path = os.path.join("data/augmented",f"v{params['ingest']['dcount']}","Images")

    # df1 = creatingInfoData(annot_path)
    df1 = pd.read_pickle(os.path.join(transform_path,"v{}_train.pkl".format(params['ingest']['dcount'])))
    df2 = aug_img_df(aug_annot_path)
    # dataframe = pd.concat([df1,df2])
    dataframe = df1
    dataframe["name"] = [x.replace("Annotations","Images") for x in dataframe["name"]]
    # print(df1)
    
    # old_fname = os.path.join(img_path, dataframe["name"][0]) + ".jpg"
    old_fname = os.path.join(dataframe["name"][0]) + ".jpg"
    annotations = []
    dataset = []
    for index,row in dataframe.iterrows():
        # fname = os.path.join(img_path, row["name"]) + ".jpg"
        fname = os.path.join(row["name"]) + ".jpg"
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

    annot_path = os.path.join("data/split",f"v{params['ingest']['dcount']}","val/Annotations")
    img_path = os.path.join("data/split",f"v{params['ingest']['dcount']}","val/Images")
    
    # dataframe = creatingInfoData(annot_path)
    dataframe = pd.read_pickle(os.path.join(transform_path,"v{}_val.pkl".format(params['ingest']['dcount'])))
    dataframe["name"] = [x.replace("Annotations","Images") for x in dataframe["name"]]
    # print(dataframe)
    
    # old_fname = os.path.join(img_path, dataframe["name"][0]) + ".jpg"
    old_fname = os.path.join(dataframe["name"][0]) + ".jpg"
    annotations = []
    dataset = []
    for index,row in dataframe.iterrows():
        # fname = os.path.join(img_path, row["name"]) + ".jpg"
        fname = os.path.join(row["name"]) + ".jpg"
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


def detectron_custom_train():
    # Custom Training
    # register_coco_instances("my_dataset_train", {}, os.path.join(transform_path,"_annotations_train.coco.json"), os.path.join("data/split",f"v{params['ingest']['dcount']}","train/Images"))
    # register_coco_instances("my_dataset_val", {}, os.path.join(transform_path,"_annotations_val.coco.json"), os.path.join("data/split",f"v{params['ingest']['dcount']}","val/Images"))

    # register_coco_instances("my_dataset_train", {}, os.path.join(transform_path,"_annotations_train.coco.json"), os.path.join("/home/yln1kor/nikhil-test/Datasets/kar_train"))
    # register_coco_instances("my_dataset_val", {}, os.path.join(transform_path,"_annotations_val.coco.json"), os.path.join("home/yln1kor/nikhil-test/Datasets/kar_val"))

    DatasetCatalog.register("my_dataset_train", custom_dataset_function_train)
    MetadataCatalog.get("my_dataset_train").set(thing_classes = ["person"])

    DatasetCatalog.register("my_dataset_val", custom_dataset_function_test)
    MetadataCatalog.get("my_dataset_val").set(thing_classes = ["person"])

    my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
    dataset_dicts = DatasetCatalog.get("my_dataset_train")

    # for d in random.sample(dataset_dicts, 3):
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
    #     vis = visualizer.draw_dataset_dict(d)
    #     cv2.imshow("out",vis.get_image()[:, :, ::-1])
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cfg = get_cfg()
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(params["detectron_parameters"]["config_file"])
    cfg.merge_from_file(model_zoo.get_config_file(params["detectron_parameters"]["config_file"]))
    if params['version']['best'] == "v0-infer" or params['version']['best'] == "VD":
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(params["detectron_parameters"]["config_file"])
    else:
        # cfg.MODEL.WEIGHTS = os.path.join(sys.argv[2],params['version']['best'],"model_final.pth")
        cfg.MODEL.WEIGHTS = os.path.join("runs/train/expt{}".format((params['version']['best'])[1]),"weights/model_final.pth")

    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = params["detectron_parameters"]["NUM_WORKERS"]
    cfg.SOLVER.IMS_PER_BATCH = params["detectron_parameters"]["IMS_PER_BATCH"]
    cfg.SOLVER.BASE_LR = params["detectron_parameters"]["BASE_LR"]
    cfg.SOLVER.WARMUP_ITERS = params["detectron_parameters"]["WARM_UP_ITERS"]
    cfg.SOLVER.MAX_ITER = params["detectron_parameters"]["MAX_ITER"] #adjust up if val mAP is still rising, adjust down if overfit
    # cfg.SOLVER.STEPS = (1000, 1500)
    cfg.SOLVER.GAMMA = params["detectron_parameters"]["GAMMA"]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = params["detectron_parameters"]["BATCH_SIZE_PER_IMAGE"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = params["detectron_parameters"]["NUM_CLASSES"]
    cfg.TEST.EVAL_PERIOD = params["detectron_parameters"]["EVAL_PERIOD"]
    cfg.OUTPUT_DIR = output_train

    os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume = False)
    trainer.train()


if __name__ == "__main__":

    print("\n\n\n")
    print("-------------------------------")
    print("Training.....")
    print("-------------------------------")
    print("\n\n\n")

    detectron_custom_train()
    print("\n\n\n")
    print("-------------------------------")
    print("Training Completed.....")
    print("-------------------------------")
    print("\n\n\n")