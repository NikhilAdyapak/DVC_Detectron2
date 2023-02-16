# def A():
#     global x
#     x = 10

# def B():
#     global x 
#     print(x)

# A()
# B()

# import os
# import cv2
# # for (root,dirs,files) in os.walk(base_path):
# #     print(files)

# def my_dataset_function(base_path):
#     # file_name, height, width, image_id
#     dataset = []
#     for (root,dirs,files) in os.walk(base_path):
#         file_name = files
#     file_names = [os.path.join(base_path, x) for x in file_name]
#     height = [cv2.imread(x).shape[0] for x in file_names]
#     width = [cv2.imread(x).shape[1] for x in file_names]
#     image_id = file_name
#     print(height, '\n', len(height), '\n', len(file_names))

# base_path = '/home/yln1kor/nikhil-test/Datasets/archive/Test/Test/JPEGImages'
# my_dataset_function(base_path)

# import cv2
# base_path = '/home/yln1kor/nikhil-test/Datasets/archive/Test/Test/JPEGImages/image (1).jpg'

# img = cv2.imread(base_path)
# print(img.shape[0],img.shape[1])

# import re
# base_path = '/home/yln1kor/nikhil-test/Datasets/archive/Test/Test/JPEGImages/image (1).jpg'
# print(type(re.findall(r'\d+', base_path)[0]))



# from detectron2.data import transforms as T
# import pandas as pd
# import glob,os, numpy as np
# import xml.etree.ElementTree as ET
# import cv2
# # Define a sequence of augmentations:
# augs = T.AugmentationList([
#     T.RandomBrightness(0.9, 1.1),
#     T.RandomFlip(prob=0.5),
#     T.RandomCrop("absolute", (640, 640))
# ])  # type: T.Augmentation

# def creatingInfoData(Annotpath):
#     information={'xmin':[],'ymin':[],'xmax':[],'ymax':[],'ymax':[],'name':[]
#                 ,'label':[]}

#     for file in sorted(glob.glob(str(Annotpath+'/*.xml*'))):
#         dat=ET.parse(file)
#         for element in dat.iter():    

#             if 'object'==element.tag:
#                 for attribute in list(element):
#                     if 'name' in attribute.tag:
#                         name = attribute.text                 
#                         information['label'] += [name]
#                         information['name'] +=[file.split('/')[-1][0:-4]]

#                     if 'bndbox'==attribute.tag:
#                         for dim in list(attribute):
#                             if 'xmin'==dim.tag:
#                                 xmin=int(round(float(dim.text)))
#                                 information['xmin']+=[xmin]
#                             if 'ymin'==dim.tag:
#                                 ymin=int(round(float(dim.text)))
#                                 information['ymin']+=[ymin]
#                             if 'xmax'==dim.tag:
#                                 xmax=int(round(float(dim.text)))
#                                 information['xmax']+=[xmax]
#                             if 'ymax'==dim.tag:
#                                 ymax=int(round(float(dim.text)))
#                                 information['ymax']+=[ymax]
                     
#     return pd.DataFrame(information)

# base_path = '/home/yln1kor/nikhil-test/Datasets/archive/Test/Test/JPEGImages/image (1).jpg'
# annot_path = '/home/yln1kor/nikhil-test/Datasets/archive/Test/Test/Annotations'
# dataframe = creatingInfoData(annot_path)
# temp = dataframe[dataframe.name == 'image (1)']
# boxes = []
# for index,row in temp.iterrows():
#     boxes.append([row['xmin'],row['ymin'],row['xmax'],row['ymax']])
# print(boxes,'\n')

# image = cv2.imread(base_path)
# # Define the augmentation input ("image" required, others optional):
# input = T.AugInput(image, boxes = boxes)
# # Apply the augmentation:
# transform = augs(input)  # type: T.Transform
# image_transformed = input.image  # new image
# # sem_seg_transformed = input.sem_seg  # new semantic segmentation

# # For any extra data that needs to be augmented together, use transform, e.g.:
# image = cv2.imread(base_path)
# image2_transformed = transform.apply_image(image)
# # polygons = boxes
# # polygons_transformed = transform.apply_polygons(polygons)

# # cv2.imshow('output',transform)
# # cv2.waitKey(0)
# cv2.imshow('output',image_transformed)
# cv2.waitKey(0)
# cv2.imshow('output',image2_transformed)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(image2_transformed,'\n\n')

# d = {'a':1,'b':2,'c':None}

# print(d['a'],d['b'],d['c'])

import json
from collections import OrderedDict

d = OrderedDict([('bbox', {'AP': 57.332991072547934, 'AP50': 80.06881315052435, 'AP75': 65.82521254362214, 'APs': 62.61058631793035, 'APm': None, 'APl': None})])
d1 = next(iter(d.items()))
d2 = next(iter(d.values()))
# d2 = {'AP': 57.332991072547934, 'AP50': 80.06881315052435, 'AP75': 65.82521254362214, 'APs': 62.61058631793035, 'APm': nan, 'APl': nan} 
d = {'AP':d2["AP"],'AP50':d2["AP50"],'AP75':d2["AP75"],'APs':d2['APs']}
s1 = json.dumps(d)
d2 = json.loads(s1)

print(d2, type(d2))

# from draft import creatingInfoData
# import json,pandas as pd

# annot_path = '/home/yln1kor/nikhil-test/Datasets/archive/Train/Train/Annotations'
# dataset = creatingInfoData(annot_path)

# print(dataset)

# print(dataset.info())

# print(dataset.head(),'\n',dataset.tail())