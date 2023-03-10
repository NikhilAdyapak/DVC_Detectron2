import numpy as np
from pathlib import Path
import xml.etree.cElementTree as ET
from PIL import Image
import os

image_path = "" # provide image path
anno_folder = "" # provide .XML folder path
filename = "" # provide image name
#Read each bounding box as a list of dictionary and append it in the list for each file
All_bboxes = "[{"xmin":1433,"xmax":1503,"ymin":1570,"ymax":1700,"skuLabel":"bus"}]" 


img = np.array(Image.open(image_path).convert('RGB'))

annotation = ET.Element('annotation')
ET.SubElement(annotation, 'folder').text = str(anno_folder)
ET.SubElement(annotation, 'filename').text = str(filename)
ET.SubElement(annotation, 'path').text = str(filename)

source = ET.SubElement(annotation, 'source')
ET.SubElement(source, 'database').text = 'Unknown'

size = ET.SubElement(annotation, 'size')
ET.SubElement(size, 'width').text = str (img.shape[1])
ET.SubElement(size, 'height').text = str(img.shape[0])
ET.SubElement(size, 'depth').text = str(img.shape[2])

ET.SubElement(annotation, 'segmented').text = '0'

for item in All_bboxes:
    label = item['Label']
    xmax = item['xmax']
    xmin = item['xmin']
    ymin = item['ymin']
    ymax = item['ymax']

    object = ET.SubElement(annotation, 'object')
    ET.SubElement(object, 'name').text = label
    ET.SubElement(object, 'pose').text = 'Unspecified'
    ET.SubElement(object, 'truncated').text = '0'
    ET.SubElement(object, 'difficult').text = '0'

    bndbox = ET.SubElement(object, 'bndbox')
    ET.SubElement(bndbox, 'xmin').text = str(xmin)
    ET.SubElement(bndbox, 'ymin').text = str(ymin)
    ET.SubElement(bndbox, 'xmax').text = str(xmax)
    ET.SubElement(bndbox, 'ymax').text = str(ymax)

tree = ET.ElementTree(annotation)
xml_file_name = os.path.join(anno_folder, f'{filename.split('.')[0]}.xml')
tree.write(xml_file_name)