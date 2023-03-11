import sys
import os
import yaml
import pandas as pd
import glob
import xml.etree.ElementTree as ET
import natsort

# from helper.xml_to_df import *
from helper.txt_to_df import *
from helper.voc2coco import *

params = yaml.safe_load(open('params.yaml'))

if len(sys.argv) != 4:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/transform.py data/augmented data/split data/transform\n'
    )
    sys.exit(1)


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
                     os.path.join(Annotpath,file_name),
                     member[0].text,
                     )
            xml_list.append(value)
    column_name = ['xmin', 'ymin', 'xmax', 'ymax', 'name', 'label']
    xml_df = pd.DataFrame(xml_list, columns = column_name)
    return xml_df



if __name__ == "__main__":

    output_transform = os.path.join(sys.argv[3],f"v{params['ingest']['dcount']}")
    os.makedirs(output_transform, exist_ok = True)

    annot_path_train = os.path.join(sys.argv[2],f"v{params['ingest']['dcount']}","train","Annotations")

    # annot_path_aug = os.path.join(sys.argv[1],f"v{params['ingest']['dcount']}","Annotations")

    annot_path_val = os.path.join(sys.argv[2],f"v{params['ingest']['dcount']}","val","Annotations")

    train_output_annot = creatingInfoData(annot_path_train)
    # aug_output_annot = aug_img_df(annot_path_aug)

    # df_train = pd.concat([train_output_annot,aug_output_annot])
    df_train = train_output_annot
    df_val = creatingInfoData(annot_path_val)

    print("-------------------------------")
    print("Tranforming.....")
    print("-------------------------------")

    df_train.to_pickle(os.path.join(output_transform,'v{}_train.pkl'.format(params['ingest']['dcount'])))
    df_val.to_pickle(os.path.join(output_transform,'v{}_val.pkl'.format(params['ingest']['dcount'])))

    xml_list = []
    for root,subdir,files in os.walk(annot_path_train):
        xml_list = files
    xml_list = natsort.natsorted(xml_list)

    with open(os.path.join(output_transform,'train_ids.txt'), 'w') as fp:
        for item in xml_list:
            fp.write("%s\n" % item)
        print('Done')
    
    labels = ["person","person-like"]
    with open(os.path.join(output_transform,'labels.txt'), 'w') as fp:
        for item in labels:
            fp.write("%s\n" % item)
        print('Done')


    xml_list = []
    for root,subdir,files in os.walk(annot_path_val):
        xml_list = files
    xml_list = natsort.natsorted(xml_list)

    with open(os.path.join(output_transform,'val_ids.txt'), 'w') as fp:
        for item in xml_list:
            fp.write("%s\n" % item)
        print('Done')

    cmd_train = "python3 src/helper/voc2coco.py --ann_dir " + annot_path_train + " --ann_ids " + os.path.join(output_transform,'train_ids.txt') + " --labels "+ os.path.join(output_transform,'labels.txt') + " --output " + os.path.join(output_transform,"_annotations_train.coco.json")
    cmd_val = "python3 src/helper/voc2coco.py --ann_dir " + annot_path_val + " --ann_ids " + os.path.join(output_transform,'val_ids.txt') + " --labels "+ os.path.join(output_transform,'labels.txt') + " --output " + os.path.join(output_transform,"_annotations_val.coco.json")

    os.system(cmd_train)
    os.system(cmd_val)
    print("Yaaay")
    # os.system(cmd_test)

    # register_coco_instances("my_dataset_train", {}, "/home/yln1kor/Dataset/karthika95/Train/_annotations.coco.json", "/home/yln1kor/Dataset/karthika95/Train")
    # register_coco_instances("my_dataset_test", {}, "/home/yln1kor/Dataset/karthika95/Val/_annotations.coco.json", "/home/yln1kor/Dataset/karthika95/Val")

    # df = pd.read_pickle(file_name)