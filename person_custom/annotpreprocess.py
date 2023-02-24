import os
from scipy.io import loadmat
import pandas as pd

import numpy as np
import os, json, cv2, random, glob
import re
import xml.etree.ElementTree as ET

# air_list = os.listdir("images")
# annot_list = os.listdir("annotations")
# air_list.sort()
# annot_list.sort()
# coords = []
# for i in range(len(air_list)):
#     ann = loadmat('annots/' + annot_list[i])
#     temp_lst = []
#     temp_lst.append(air_list[i])
#     temp_lst.append(ann['box_coord'][0][2])
#     temp_lst.append(ann['box_coord'][0][0])
#     temp_lst.append(ann['box_coord'][0][3])
#     temp_lst.append(ann['box_coord'][0][1])
    
#     coords.append(temp_lst)

# df = pd.DataFrame(coords)
# df.to_csv('annotations.csv', index=False, header=None)

def creatingInfoData(Annotpath):
    information={'name':[],'xmin':[],'ymin':[],'xmax':[],'ymax':[],'ymax':[]}

    for file in sorted(glob.glob(str(Annotpath+'/*.xml*'))):
        dat=ET.parse(file)
        for element in dat.iter():    

            if 'object'==element.tag:
                for attribute in list(element):
                    if 'name' in attribute.tag:
                        name = attribute.text                 
                        # information['label'] += [name]
                        information['name'] +=[file.split('/')[-1][0:-4] + '.jpg']

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

Annotpath = '/home/yln1kor/nikhil-test/Datasets/archive/Train/Train/Annotations'
df = creatingInfoData(Annotpath)
df.to_csv('/home/yln1kor/nikhil-test/Models/Custom_Object_Detection/dataset/annotations.csv', index=False, header=None)