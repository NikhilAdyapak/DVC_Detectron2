# import sys
# import os
# import yaml
# import pandas as pd
# import glob
# import xml.etree.ElementTree as ET

# if len(sys.argv) != 3:
#     sys.stderr.write('Arguments error. Usage:\n')
#     sys.stderr.write(
#         '\tpython3 src/transform.py data/prepared data/transformed\n'
#     )
#     sys.exit(1)

# params = yaml.safe_load(open('params.yaml'))['ingest']

# images = os.path.join(sys.argv[1],f"v{params['dcount']}",'images')
# annots = os.path.join(sys.argv[1],f"v{params['dcount']}",'annotations')

# outputannot = os.path.join(sys.argv[2],f"v{params['dcount']}")
# os.makedirs(outputannot, exist_ok=True)

# def generate_data( Annotpath, Imagepath):
#     information={'xmin':[],'ymin':[],'xmax':[],'ymax':[],'ymax':[],'name':[] ,'label':[], 'image':[]}
#     for file in sorted(glob.glob(str(Annotpath+'/*.xml*'))):
#         dat=ET.parse(file)
#         for element in dat.iter():    
#             if 'object'==element.tag:
#                 for attribute in list(element):
#                     if 'name' in attribute.tag:
#                         name = attribute.text
#                         file_name = file.split('/')[-1][0:-4]
#                         f = os.path.basename(file_name)
#                         information['label'] += [name]
#                         information['name'] +=[f+'.jpg']
#                         information['image'] += [os.path.join(images,f+'.jpg')]
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


# print("-------------------------------")
# print("Converting XML files to dataframe.....")
# print("-------------------------------")
# df = generate_data(annots,images)
# df.to_pickle(os.path.join(outputannot,'v{}.pkl'.format(params['dcount'])))


import sys
import os
import yaml
import pandas as pd
import glob
import xml.etree.ElementTree as ET

params = yaml.safe_load(open('params.yaml'))

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
            data = (f.read()).split()
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


if len(sys.argv) != 4:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/transform.py data/prepared data/augmented data/transformed\n'
    )
    sys.exit(1)

outputannot = os.path.join(sys.argv[3],f"v{params['ingest']['dcount']}")
os.makedirs(outputannot, exist_ok = True)

train_annot_path = params['train']['annot_path']
train_aug_path = params['train']['aug_annot_path']

test_annot_path = params['test']['annot_path']
test_aug_path = params['test']['aug_annot_path']

df1 = creatingInfoData(train_annot_path)
# df2 = aug_img_df(train_aug_path)
df2 = None
# train_output_annot = pd.concat([df1,df2])
train_output_annot = df1

df1 = creatingInfoData(test_annot_path)
# df2 = aug_img_df(test_aug_path)
df2 = None
# test_output_annot = pd.concat([df1,df2])
test_output_annot = df1

print("-------------------------------")
print("Transforming.....")
print("-------------------------------")
train_output_annot.to_pickle(os.path.join(outputannot,'v{}_train.pkl'.format(params['ingest']['dcount'])))
test_output_annot.to_pickle(os.path.join(outputannot,'v{}_test.pkl'.format(params['ingest']['dcount'])))