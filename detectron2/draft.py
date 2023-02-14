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


def load_model():
    global cfg
    cfg = get_cfg()
    # cfg.MODEL.DEVICE = 'cpu'
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    global predictor
    predictor = DefaultPredictor(cfg)


def predict(base_path):
    global cfg, predictor
    overall_output = []
    
    for i in range(1,11):
        local_path = 'image (' + str(i) + ').jpg'
        im = cv2.imread(os.path.join(base_path, local_path))
        outputs = predictor(im)
        temp = {"img_num":i,"output":outputs}
        overall_output.append(temp)
    return overall_output
        # print(outputs["instances"].pred_classes)
        # print(outputs["instances"].pred_boxes)


def visualize(overall_output, base_path):
    global cfg, predictor
    for i in range(1,11):
        local_path = 'image (' + str(i) + ').jpg'
        im = cv2.imread(os.path.join(base_path, local_path))
        outputs = overall_output[i-1]["output"]
        # pred = [0,2,3,5,6,7,9,10,11]
        pred = [0]
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)

        for j in pred:
            out = v.draw_instance_predictions(outputs["instances"][outputs['instances'].pred_classes == j].to("cpu"))
        cv2.imshow("output",out.get_image())
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def creatingInfoData(Annotpath):
    information={'xmin':[],'ymin':[],'xmax':[],'ymax':[],'ymax':[],'name':[]
                ,'label':[]}

    for file in sorted(glob.glob(str(Annotpath+'/*.xml*'))):
        dat=ET.parse(file)
        for element in dat.iter():    

            if 'object'==element.tag:
                for attribute in list(element):
                    if 'name' in attribute.tag:
                        name = attribute.text                 
                        information['label'] += [name]
                        information['name'] +=[file.split('/')[-1][0:-4]]

                    if 'bndbox'==attribute.tag:
                        for dim in list(attribute):
                            if 'xmin'==dim.tag:
                                xmin=int(round(float(dim.text)))
                                information['xmin']+=[xmin]
                            if 'ymin'==dim.tag:
                                ymin=int(round(float(dim.text)))
                                information['ymin']+=[ymin]
                            if 'xmax'==dim.tag:
                                xmax=int(round(float(dim.text)))
                                information['xmax']+=[xmax]
                            if 'ymax'==dim.tag:
                                ymax=int(round(float(dim.text)))
                                information['ymax']+=[ymax]
                     
    return pd.DataFrame(information)


class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder = None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok = True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)


def custom_dataset_function_train():
    # file_name, height, width, image_id
    #[{'file_name': '/home/samjith/0000180.jpg', 'height': 788, 'width': 1400, 'image_id': 1, 
    #   'annotations': [{'bbox': [250.0, 675.0, 23.0, 17.0], 'bbox_mode': <BoxMode.XYWH_ABS: 1>, 'area': 391.0, 'segmentation': [],
    #        'category_id': 0}, {'bbox': [295.0, 550.0, 21.0, 20.0], 'bbox_mode': <BoxMode.XYWH_ABS: 1>, 'area': 420.0, 'segmentation': [], 'category_id': 0},..
    
    base_path = '/home/yln1kor/nikhil-test/Datasets/archive/Train/Train/JPEGImages'
    annot_path = '/home/yln1kor/nikhil-test/Datasets/archive/Train/Train/Annotations'

    dataset = []
    annotations = []

    for (root,dirs,files) in os.walk(base_path):
        file_name = files

    file_names = [os.path.join(base_path, x) for x in file_name]
    # file_names = file_names[0:len(file_names)//2]
    height = [cv2.imread(x).shape[0] for x in file_names]
    width = [cv2.imread(x).shape[1] for x in file_names]
    image_id = file_name
    '''[0:len(file_name)//2]'''
    dataframe = creatingInfoData(annot_path)

    for i in range(len(file_names)):
        temp = dataframe[dataframe.name == os.path.splitext(file_name[i])[0]]
        persons = []
        for index,row in temp.iterrows():
            person = {'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']], 
            'bbox_mode': 0, 
            'area': (row['xmax'] - row['xmin']) * (row['ymax'] - row['ymin']), 
            'segmentation': [],
            'category_id':0}
            persons.append(person)
        annotations.append(persons)

    for i in range(len(file_names)):
        dataset.append({"file_name":file_names[i], 
                        "height":height[i], 
                        "width":width[i],
                        "image_id":re.findall(r'\d+', image_id[i])[0],
                        "annotations":annotations[i]})
    return dataset


def custom_dataset_function_test():
    # file_name, height, width, image_id
    #[{'file_name': '/home/samjith/0000180.jpg', 'height': 788, 'width': 1400, 'image_id': 1, 
    #   'annotations': [{'bbox': [250.0, 675.0, 23.0, 17.0], 'bbox_mode': <BoxMode.XYWH_ABS: 1>, 'area': 391.0, 'segmentation': [],
    #        'category_id': 0}, {'bbox': [295.0, 550.0, 21.0, 20.0], 'bbox_mode': <BoxMode.XYWH_ABS: 1>, 'area': 420.0, 'segmentation': [], 'category_id': 0},..
    
    base_path = '/home/yln1kor/nikhil-test/Datasets/archive/Test/Test/JPEGImages'
    annot_path = '/home/yln1kor/nikhil-test/Datasets/archive/Test/Test/Annotations'

    dataset = []
    annotations = []

    for (root,dirs,files) in os.walk(base_path):
        file_name = files

    file_names = [os.path.join(base_path, x) for x in file_name]
    # file_names = file_names[0:len(file_names)//2]
    height = [cv2.imread(x).shape[0] for x in file_names]
    width = [cv2.imread(x).shape[1] for x in file_names]
    image_id = file_name
    '''[0:len(file_name)//2]'''
    dataframe = creatingInfoData(annot_path)

    for i in range(len(file_names)):
        temp = dataframe[dataframe.name == os.path.splitext(file_name[i])[0]]
        persons = []
        for index,row in temp.iterrows():
            person = {'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']], 
            'bbox_mode': 0, 
            'area': (row['xmax'] - row['xmin']) * (row['ymax'] - row['ymin']), 
            'segmentation': [],
            'category_id':0}
            persons.append(person)
        annotations.append(persons)

    for i in range(len(file_names)):
        dataset.append({"file_name":file_names[i], 
                        "height":height[i], 
                        "width":width[i],
                        "image_id":re.findall(r'\d+', image_id[i])[0],
                        "annotations":annotations[i]})
    return dataset


def custom_visualization(dataset_dicts,my_dataset_train_metadata):
    for d in random.sample(dataset_dicts, 5):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata = my_dataset_train_metadata, scale = 0.5)
        vis = visualizer.draw_dataset_dict(d)
        # cv2_imshow(vis.get_image()[:, :, ::-1])
        cv2.imshow("output",vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def detectron_data_load():
    # torch.cuda.empty_cache()
    global dataset_name_train,dataset_name_test,cfg,predictor
    dataset_name_train = "karthika95_dataset_train"
    dataset_name_test = "karthika95_dataset_test"

    DatasetCatalog.register(dataset_name_train, custom_dataset_function_train)
    MetadataCatalog.get(dataset_name_train).set(thing_classes = ["person"])

    DatasetCatalog.register(dataset_name_test, custom_dataset_function_test)
    MetadataCatalog.get(dataset_name_test).set(thing_classes = ["person"])

    # dataset_dicts = DatasetCatalog.get(dataset_name_train)
    # print(dataset_dicts[0],len(dataset_dicts))
    # if dataset_name in DatasetCatalog.list():
    #     print(True)
    #     dataset = DatasetCatalog.get(dataset_name)

    # my_dataset_train_metadata = MetadataCatalog.get(dataset_name_train).set(thing_classes = ["person"])
    # dataset_dicts = DatasetCatalog.get(dataset_name_train)
    # custom_visualization(dataset_dicts,my_dataset_train_metadata)


def detectron_train():
    # Custom Training
    global dataset_name_train,dataset_name_test,cfg,predictor
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (dataset_name_train,)
    # cfg.DATASETS.TEST = (dataset_name_test,)

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.001


    # cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.MAX_ITER = 300 #adjust up if val mAP is still rising, adjust down if overfit
    # cfg.SOLVER.STEPS = (100, 150)
    cfg.SOLVER.GAMMA = 0.05

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 4
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 #your number of classes + 1

    # cfg.TEST.EVAL_PERIOD = 200
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # uncomment below to train
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = CocoTrainer(cfg)
    # trainer.resume_or_load(resume=False)
    # trainer.train()


def detectron_visualize():
    # # Detectron2 Visualizer

    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    global dataset_name_train,dataset_name_test,cfg,predictor

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    # DatasetCatalog.register(dataset_name_test, custom_dataset_function_test)
    # MetadataCatalog.get(dataset_name_test).set(thing_classes = ["person"])

    my_dataset_test_metadata = MetadataCatalog.get(dataset_name_test).set(thing_classes = ["person"])
    # from detectron2.utils.visualizer import ColorMode
    dataset_dicts = DatasetCatalog.get(dataset_name_test)
    for d in random.sample(dataset_dicts, 5):    
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    #     v = Visualizer(im[:, :, ::-1],
    #                 metadata=my_dataset_test_metadata, 
    #                 scale=0.5, 
    # #                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    #     )
    #     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #     cv2_imshow(out.get_image()[:, :, ::-1])
        v = Visualizer(img[:, :, ::-1], metadata = my_dataset_test_metadata, scale = 0.5)
        # vis = visualizer.draw_dataset_dict(d)
        out = v.draw_instance_predictions(outputs["instances"][outputs['instances'].pred_classes == 0].to("cpu"))
        # cv2_imshow(vis.get_image()[:, :, ::-1])
        cv2.imshow("output",out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def detectron_evaluator():
    # # Detectron Evaluator

    global dataset_name_train,dataset_name_test,cfg,predictor
    evaluator = COCOEvaluator(dataset_name_test, cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, dataset_name_test)
    eval_results = inference_on_dataset(predictor.model, val_loader, evaluator)


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES']='0'
    base_path_train = '/home/yln1kor/nikhil-test/Datasets/archive/Train/Train/JPEGImages'
    annot_path_train = '/home/yln1kor/nikhil-test/Datasets/archive/Train/Train/Annotations'
    # load_model()
    # overall_output = predict(base_path)
    # visualize(overall_output,base_path)

    # with open('my_dataset.txt', 'w') as fout:
    #     # json.dump(dataset, fout)
    #     fout.write(json.dumps(dataset, indent = 4))

    detectron_data_load()
    # detectron_train()
    detectron_visualize()
    detectron_evaluator()
