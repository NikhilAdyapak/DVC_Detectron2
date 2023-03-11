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
from helper.custom_evaluate import *

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


def detectron_custom_train():
    # Custom Training
    register_coco_instances("my_dataset_train", {}, os.path.join(transform_path,"_annotations_train.coco.json"), os.path.join("data/split",f"v{params['ingest']['dcount']}","train/Images"))
    register_coco_instances("my_dataset_val", {}, os.path.join(transform_path,"_annotations_val.coco.json"), os.path.join("data/split",f"v{params['ingest']['dcount']}","val/Images"))

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
    cfg.merge_from_file(model_zoo.get_config_file(params["detectron_parameters"]["config_file"]))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_test",)

    cfg.DATALOADER.NUM_WORKERS = params["detectron_parameters"]["NUM_WORKERS"]
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(params["detectron_parameters"]["config_file"])
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

    print("-------------------------------")
    print("Training.....")
    print("-------------------------------")

    detectron_custom_train()
    print("\n\n\n")
    print("-------------------------------")
    print("Training Completed.....")
    print("-------------------------------")
    print("\n\n\n")