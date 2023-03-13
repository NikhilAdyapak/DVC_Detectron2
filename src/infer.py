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
            'category_id':1})
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
    MetadataCatalog.get("my_dataset_val").set(thing_classes = ["persons"])

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(params['detectron_parameters']['config_file']))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(params['detectron_parameters']['config_file'])
    predictor = DefaultPredictor(cfg)

    my_dataset_test_metadata = MetadataCatalog.get("my_dataset_val").set(thing_classes = ["person","person-like"])
    # from detectron2.utils.visualizer import ColorMode
    dataset_dicts = DatasetCatalog.get("my_dataset_val")
    for d in random.sample(dataset_dicts, 5):    
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], metadata = my_dataset_test_metadata, scale = 0.5)
        # vis = visualizer.draw_dataset_dict(d)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
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
    # evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir = output_infer)
    # val_loader = build_detection_test_loader(cfg, "my_dataset_val")
    # eval_results = inference_on_dataset(predictor.model, val_loader, evaluator)
    # print(eval_results)




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