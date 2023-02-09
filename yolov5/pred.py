from cProfile import label
import torch
import sys
import os
from PIL import Image
import yaml
import glob
import pandas as pd
import cv2

# if len(sys.argv) != 4:
#     sys.stderr.write('Arguments error. Usage:\n')
#     sys.stderr.write(
#         '\tpython3 src/predict.py data/prepared data/predictions data/store\n'
#     )
#     sys.exit(1)

# params = yaml.safe_load(open('params.yaml'))['ingest']


# data_path = os.path.join(sys.argv[1], f"v{params['dcount']}", 'images')
# predict_path = os.path.join(sys.argv[2], f"v{params['dcount']}", 'images')
# origpred = os.path.join(sys.argv[3], f"v{params['dcount']}", 'predictions')

# pred_path = os.path.join(sys.argv[2], "v{}".format(params['dcount']))

# print(predict_path)
# print(data_path)

# os.makedirs(predict_path, exist_ok=True)
# os.makedirs(origpred, exist_ok=True)

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, _verbose =False )
# img = os.path.join(data_path, os.listdir(data_path)[0])

# preds = glob.glob(f'{data_path}/*.jpg', recursive=True)
# labels = os.listdir(data_path)

# information={'xmin':[],'ymin':[],'xmax':[],'ymax':[],'name':[] ,'label':[], 'image':[]}
# def predict(information,data_path):
#     for images in os.listdir(data_path):
#         img = cv2.imread(os.path.join(data_path,images))
#         pred = model(img)
#         pred.render()
#         df = pred.pandas().xyxyn[0]
#         res = df[df["name"]=="person"]
        
#         for index, yolo_bb in res.iterrows():
#             #file_name = images.split('/')[-1][0:-4]
#             information['name']+= [images]
#             information['label']+= [yolo_bb['name']]
#             information['xmin']+= [yolo_bb["xmin"]*img.shape[1]]
#             information['xmax']+= [yolo_bb["xmax"]*img.shape[1]]
#             information['ymin']+= [yolo_bb["ymin"]*img.shape[0]]
#             information['ymax']+= [yolo_bb["ymax"]*img.shape[0]]
#             information['image']+= [f'{data_path}/{images}']
#     return pd.DataFrame(information)


# print("-------------------------------")
# print("Prediction using model.....")
# print("-------------------------------")
# annots_data = predict(information,data_path)
# annots_data.to_pickle(os.path.join(pred_path,'v{}.pkl'.format(params['dcount'])))
# print(annots_data)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, _verbose =False )
base_path = '/home/yln1kor/test/Datasets/archive/Test/Test/JPEGImages'
for i in range(1,15):
    local_path = 'image (' + str(i) + ').jpg'
    img = cv2.imread(os.path.join(base_path, local_path))
    results = model(img)
    # results.xyxy[0]  # img1 predictions (tensor)
    results.render()
    df = results.pandas().xyxy[0]  # img1 predictions (pandas)
    # print(df)
    image = img
    for row, col in df.iterrows():
        box = [0,0,0,0]
        box[0] = df['xmin'][row]
        box[1] = df['ymin'][row]
        box[2] = df['xmax'][row]
        box[3] = df['ymax'][row]
        image = cv2.rectangle(img, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), (0,255,0), 2)
    cv2.imshow('test',image)
    cv2.waitKey(0)