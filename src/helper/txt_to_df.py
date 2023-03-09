import os, glob, pandas as pd

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

# print(aug_img_df("aug"))