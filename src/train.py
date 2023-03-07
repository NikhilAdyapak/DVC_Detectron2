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
from helper.txt_to_df import *

if len(sys.argv) != 5:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/train.py data/prepared data/transformed data/train data/predict\n'
    )
    sys.exit(1)

params = yaml.safe_load(open('params.yaml'))
outputpred = os.path.join(sys.argv[3],f"v{params['ingest']['dcount']}")
os.makedirs(outputpred, exist_ok = True)

base_path = os.path.join(sys.argv[1],f"v{params['ingest']['dcount']}")


def custom_dataset_function_train():
    # file_name, height, width, image_id
    #[{'file_name': '/home/samjith/0000180.jpg', 'height': 788, 'width': 1400, 'image_id': 1, 
    #   'annotations': [{'bbox': [250.0, 675.0, 23.0, 17.0], 'bbox_mode': <BoxMode.XYWH_ABS: 1>, 'area': 391.0, 'segmentation': [],
    #        'category_id': 0}, {'bbox': [295.0, 550.0, 21.0, 20.0], 'bbox_mode': <BoxMode.XYWH_ABS: 1>, 'area': 420.0, 'segmentation': [], 'category_id': 0},..

    annot_path = params['train']['annot_path']
    img_path = params['train']['img_path']
    aug_annot_path = params['train']['aug_annot_path']
    aug_img_path = params['train']['aug_img_path']

    df1 = creatingInfoData(annot_path)
    # df2 = aug_img_df(aug_annot_path)
    df2 = None
    dataframe = pd.concat([df1,df2])
    
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

    annot_path = params['test']['annot_path']
    img_path = params['test']['img_path']
    aug_annot_path = params['test']['aug_annot_path']
    aug_img_path = params['test']['aug_img_path']

    df1 = creatingInfoData(annot_path)
    # df2 = aug_img_df(aug_annot_path)
    df2 = None
    dataframe = pd.concat([df1,df2])

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


def detectron_data_load(cache,params):
    # torch.cuda.empty_cache()

    cfg = cache["cfg"]
    predictor = cache["predictor"]
    dataset_name_train = cache["train"]
    dataset_name_test = cache["test"]
    results = cache["results"]

    dataset_name_train = params['train']['dataset_name']
    dataset_name_test = params['test']['dataset_name']
    # # Change Name

    DatasetCatalog.register(dataset_name_train, custom_dataset_function_train)
    MetadataCatalog.get(dataset_name_train).set(thing_classes = ["person"])

    DatasetCatalog.register(dataset_name_test, custom_dataset_function_test)
    MetadataCatalog.get(dataset_name_test).set(thing_classes = ["person"])
    # # Change thing_classes

    # from detectron2.data.datasets import register_coco_instances
    # register_coco_instances("my_dataset_train", {}, "/content/train/_annotations.coco.json", "/content/train")
    # register_coco_instances("my_dataset_val", {}, "/content/valid/_annotations.coco.json", "/content/valid")
    # register_coco_instances("my_dataset_test", {}, "/content/test/_annotations.coco.json", "/content/test")

    # dataset_dicts = DatasetCatalog.get(dataset_name_train)
    # print(dataset_dicts[0],len(dataset_dicts))
    # if dataset_name in DatasetCatalog.list():
    #     print(True)
    #     dataset = DatasetCatalog.get(dataset_name)

    # my_dataset_train_metadata = MetadataCatalog.get(dataset_name_train).set(thing_classes = ["person"])
    # dataset_dicts = DatasetCatalog.get(dataset_name_train)
    # custom_visualization(dataset_dicts,my_dataset_train_metadata)

    cache = {"cfg":cfg,"predictor":predictor,"train":dataset_name_train,"test":dataset_name_test,"results":results}
    return cache


def detectron_custom_train(cache,params):
    # Custom Training

    cfg = cache["cfg"]
    predictor = cache["predictor"]
    dataset_name_train = cache["train"]
    dataset_name_test = cache["test"]
    results = cache["results"]

    # # Old Custom Training
    # cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    # # # Change yaml model file

    # cfg.DATASETS.TRAIN = (dataset_name_train,)
    # # cfg.DATASETS.TEST = (dataset_name_test,)

    # cfg.DATALOADER.NUM_WORKERS = 2
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    # # # Change yaml model file
    
    # cfg.SOLVER.IMS_PER_BATCH = 1
    # cfg.SOLVER.BASE_LR = 0.001


    # # cfg.SOLVER.WARMUP_ITERS = 500
    # cfg.SOLVER.MAX_ITER = 300 #adjust up if val mAP is still rising, adjust down if overfit
    # # cfg.SOLVER.STEPS = (100, 150)
    # cfg.SOLVER.GAMMA = 0.05

    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 4
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 #your number of classes + 1

    # # cfg.TEST.EVAL_PERIOD = 200
    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # # uncomment below to train
    # trainer = DefaultTrainer(cfg) 
    # trainer.resume_or_load(resume=False)
    # trainer.train()

    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = CocoTrainer(cfg)
    # trainer.resume_or_load(resume=False)
    # trainer.train()

    # # Roboflow custom Training Blood Dataset
    # cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    # cfg.DATASETS.TRAIN = ("my_dataset_train",)
    # cfg.DATASETS.TEST = ("my_dataset_val",)

    # cfg.DATALOADER.NUM_WORKERS = 4
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.SOLVER.IMS_PER_BATCH = 4
    # cfg.SOLVER.BASE_LR = 0.001


    # cfg.SOLVER.WARMUP_ITERS = 1000
    # cfg.SOLVER.MAX_ITER = 1500 #adjust up if val mAP is still rising, adjust down if overfit
    # cfg.SOLVER.STEPS = (1000, 1500)
    # cfg.SOLVER.GAMMA = 0.05

    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4 #your number of classes + 1

    # cfg.TEST.EVAL_PERIOD = 500


    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = CocoTrainer(cfg)
    # trainer.resume_or_load(resume=False)
    # trainer.train()

    # Kaggle custom dataset Training
    #from detectron2.engine import DefaultTrainer

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(params['parameters']['config_file']))
    cfg.DATASETS.TRAIN = (dataset_name_train,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = params['parameters']['NUM_WORKERS']
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(params['parameters']['config_file'])  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = params['parameters']['IMS_PER_BATCH']
    cfg.SOLVER.BASE_LR = params['parameters']['BASE_LR']  # pick a good LR
    cfg.SOLVER.MAX_ITER = params['parameters']['MAX_ITER']    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = params['parameters']['BATCH_SIZE_PER_IMAGE']   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = params['parameters']['NUM_CLASSES']  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    # cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    # cfg.DATASETS.TRAIN = (dataset_name_train,)
    # cfg.DATASETS.TEST = ()
    # cfg.DATALOADER.NUM_WORKERS = 4
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.SOLVER.IMS_PER_BATCH = 4
    # cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    # cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # uncomment below to train
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    predictor = DefaultPredictor(cfg)
    cache = {"cfg":cfg,"predictor":predictor,"train":dataset_name_train,"test":dataset_name_test,"results":results}
    return cache


def detectron_visualize(cache,params):
    # # Detectron2 Visualizer

    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained

    cfg = cache["cfg"]
    predictor = cache["predictor"]
    dataset_name_train = cache["train"]
    dataset_name_test = cache["test"]
    results = cache["results"]
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(params['parameters']['config_file']))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,"model_final.pth")
    # cfg.MODEL.WEIGHTS = params['test']['model']
    predictor = DefaultPredictor(cfg)

    my_dataset_test_metadata = MetadataCatalog.get(dataset_name_test).set(thing_classes = ["person"])
    dataset_dicts = DatasetCatalog.get(dataset_name_test)
    for d in random.sample(dataset_dicts, 5):    
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)  
        v = Visualizer(img[:, :, ::-1], metadata = my_dataset_test_metadata, scale = 0.5)
        out = v.draw_instance_predictions(outputs["instances"][outputs['instances'].pred_classes == 0].to("cpu"))
        cv2.imshow("output",out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cache = {"cfg":cfg,"predictor":predictor,"train":dataset_name_train,"test":dataset_name_test,"results":results}
    return cache


def detectron_evaluator(cache,params):
    # # Detectron Evaluator

    cfg = cache["cfg"]
    predictor = cache["predictor"]
    dataset_name_train = cache["train"]
    dataset_name_test = cache["test"]
    results = cache["results"]

    evaluator = COCOEvaluator(dataset_name_test, cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, dataset_name_test)
    eval_results = inference_on_dataset(predictor.model, val_loader, evaluator)
    # OrderedDict([('bbox', {'AP': 57.332991072547934, 'AP50': 80.06881315052435, 'AP75': 65.82521254362214, 'APs': 62.61058631793035, 'APm': nan, 'APl': nan})])

    d1 = next(iter(eval_results.items()))
    d2 = next(iter(eval_results.values()))

    d = {'AP':d2["AP"],'AP50':d2["AP50"],'AP75':d2["AP75"],'APs':d2['APs']}
    s1 = json.dumps(d)
    results = json.loads(s1)

    cache = {"cfg":cfg,"predictor":predictor,"train":dataset_name_train,"test":dataset_name_test,"results":results}
    return cache


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

    annot_path = os.path.join(base_path, params['test']['annot_path'])
    img_path = os.path.join(base_path, params['test']['img_path'])
    
    gt_df = creatingInfoData(annot_path)

    for filename in tqdm(os.listdir(annot_path)):
        ext = filename.split(".")[1]
        if ext == "xml":
            img = os.path.join(img_path, filename).replace("xml","jpg")
        else:
            img = os.path.join(img_path, filename).replace("txt","jpg")
        img = cv2.imread(img)
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        boxes = v._convert_boxes(outputs["instances"][outputs["instances"].pred_classes == 0].pred_boxes.to('cpu'))
        gt_df_sub = gt_df[gt_df["name"] == filename]
        gt_boxes = []
        for index, row in gt_df_sub.iterrows():
            gt_boxes.append([row["xmin"],row["ymin"],row["xmax"],row["ymax"]])
        out = v.draw_instance_predictions(outputs["instances"][outputs['instances'].pred_classes == 0].to("cpu"))

        for box in boxes:
            metrics = iou_mapping(box,gt_boxes,metrics)

    metrics = evaluate(metrics)
    cache = {"cfg":cfg,"predictor":predictor,"train":dataset_name_train,"test":dataset_name_test,"results":results}
    return (cache,metrics)


def results_logger(cache,params,metrics):
    # base_path = params['output']['results']
    # results_old = max(glob.glob(os.path.join(base_path, '*/')), key=os.path.getmtime)
    # run_num = int((results_old.split("/")[1]).split("_")[1]) + 1
    # new_results = os.path.join(base_path,"results_" + str(run_num))
    # src_dir = params['output']['base_dir']
    # shutil.copytree(src_dir, new_results)
    # print(new_results)
    
    hyper_parameters = {
        "config_file" : params['parameters']['config_file'],
        "NUM_WORKERS" : params['parameters']['NUM_WORKERS'],
        "IMS_PER_BATCH" : params['parameters']['IMS_PER_BATCH'],
        "BASE_LR" : params['parameters']['BASE_LR'],
        "MAX_ITER" :params['parameters']['MAX_ITER'],
        "BATCH_SIZE_PER_IMAGE" : params['parameters']['BATCH_SIZE_PER_IMAGE'],
        "NUM_CLASSES" : params['parameters']['NUM_CLASSES']
    }

    my_metrics = {
        "tp" : metrics["TP"],
        "fp" : metrics["FP"],
        "fn" : metrics["FN"],
        "Precision" : metrics["Precision"],
        "Recall" : metrics["Recall"],
        "F1" : metrics["F1"],
        "Avg_IOU" : metrics["Avg_IOU"]
    }

    with open(os.path.join(outputpred,'det2_hyperparamters_{}.txt'.format(params['ingest']['dcount'])), 'w') as fout:
        fout.write(json.dumps(hyper_parameters, indent = len(hyper_parameters)))

    with open(os.path.join(outputpred,'my_metrics_{}.txt'.format(params['ingest']['dcount'])), 'w') as fout:
        fout.write(json.dumps(my_metrics, indent = len(my_metrics)))

    with open(os.path.join(outputpred,'det2_results_{}.txt'.format(params['ingest']['dcount'])), 'w') as fout:
        fout.write(json.dumps(cache["results"], indent = len(cache["results"])))
    

# @app.route('/')
# def home_endpoint():
#     return 'Hello World!'


# @app.route('/results')
# def get_results():
#     global cache
#     return cache["results"]


# if __name__ == "__main__":
    
params = yaml.safe_load(open('params.yaml'))

cache = dict()
metrics = dict()
cache = {"cfg":None,"predictor":None,"train":None,"test":None,"results":None}
metrics = {"TP":0,"FP":0,"FN":0,"IOU":[]}

print("-------------------------------")
print("Training.....")
print("-------------------------------")

cache = detectron_data_load(cache,params)
cache = detectron_custom_train(cache,params)
# cache = detectron_visualize(cache,params)
cache = detectron_evaluator(cache,params)
(cache, metrics) = custom_evaluator(cache,params,metrics)
results_logger(cache,params,metrics)