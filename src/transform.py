import sys
import os
import yaml
import pandas as pd
import glob
import xml.etree.ElementTree as ET

params = yaml.safe_load(open('params.yaml'))

if len(sys.argv) != 4:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/transform.py data/prepared data/augmented data/transformed\n'
    )
    sys.exit(1)


def creatingInfoData(Annotpath):
    xml_list = []
    for xml_file in sorted(glob.glob(str(Annotpath+'/*.xml*'))):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text),
                     root.find('filename').text,
                     member[0].text,
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
        for data in bbox:
            data = data.split()
            value = (
                int(data[0]),
                int(data[1]),
                int(data[2]),
                int(data[3]),
                files,
                int(data[4]),
            )
            aug_list.append(value)
    column_name = ['xmin', 'ymin', 'xmax', 'ymax', 'name', 'label']
    aug_df = pd.DataFrame(aug_list, columns = column_name)
    return aug_df


if __name__ == "__main__":

    outputannot = os.path.join(sys.argv[3],f"v{params['ingest']['dcount']}")
    os.makedirs(outputannot, exist_ok = True)

    train_annot_path = params['train']['annot_path']
    test_annot_path = params['test']['annot_path']

    train_output_annot = creatingInfoData(train_annot_path)
    test_output_annot = creatingInfoData(test_annot_path)
    print("-------------------------------")
    print("Tranforming.....")
    print("-------------------------------")
    train_output_annot.to_pickle(os.path.join(outputannot,'v{}_train.pkl'.format(params['ingest']['dcount'])))
    test_output_annot.to_pickle(os.path.join(outputannot,'v{}_test.pkl'.format(params['ingest']['dcount'])))