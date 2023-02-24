import os
from scipy.io import loadmat
import pandas as pd
ann = loadmat('/home/yln1kor/nikhil-test/Datasets/car_devkit/car_devkit/devkit/cars_train_annos.mat')

base_path = '/home/yln1kor/nikhil-test/Datasets/car_devkit/cars_train'

base = None
for root,subdirs, files in os.walk(base_path):
    abs_path = [os.path.join(root,x) for x in files]
    base = root

from scipy.io import loadmat
import pandas as pd
import cv2 
coords = []
counter = 0
for i in range(len(ann['annotations'][0])):
    img_path = ann['annotations'][0][i][5][0]
    img = os.path.join(base,img_path)
    img = cv2.imread(img)
    xmin = ann['annotations'][0][i][0][0][0]
    ymin = ann['annotations'][0][i][1][0][0]
    xmax = ann['annotations'][0][i][2][0][0]
    ymax = ann['annotations'][0][i][3][0][0]
    print([xmin,xmax,ymin,ymax])
    image = cv2.rectangle(img, (xmin,ymin), (xmax ,ymax), (255,255,0), 2)
    cv2.imshow("out",image)
    cv2.waitKey(0)
    if counter == 10:
        break
    counter += 1


