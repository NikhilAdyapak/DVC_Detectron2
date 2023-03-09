import os, glob, pandas as pd

def aug_img_df(Annotpath):
    aug_list = []
    for files in sorted(glob.glob(str(Annotpath+'/*.txt*'))):
        with open(files, "r") as f:
            bbox = (f.read()).split('\n')
        for data in bbox[0:-1]:
            data = data.split()
            value = (
                float(data[0]),
                float(data[1]),
                float(data[2]),
                float(data[3]),
                files.split(".")[0],
                data[4],
            )
            aug_list.append(value)
    column_name = ['xmin', 'ymin', 'xmax', 'ymax', 'name', 'label']
    aug_df = pd.DataFrame(aug_list, columns = column_name)
    return aug_df

# print(aug_img_df("aug"))