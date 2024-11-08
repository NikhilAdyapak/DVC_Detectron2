import os,sys
from distutils.dir_util import copy_tree
import yaml
import glob

if len(sys.argv) != 2:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write(
            '\tpython3 src/ingest.py data/prepared\n'
        )
        sys.exit(1)


if __name__ == "__main__":

    params = yaml.safe_load(open('params.yaml'))

    data_path = os.path.join(sys.argv[1], f"v{params['ingest']['dcount']}")
    # print(data_path)
    base_path = "/home/yln1kor/Blob"
    latest = max(glob.glob(os.path.join(base_path, '*/')), key=os.path.getmtime)
    input_dir = params["dataset"]["path"]
    # input_dir = latest
    os.makedirs(data_path, exist_ok = True)

    print("\n\n\n")
    print("-------------------------------")
    print("Ingesting.....")
    print("-------------------------------")
    print("\n\n\n")
    
    copy_tree(input_dir, data_path)

    print("\n\n\n")
    print("-------------------------------")
    print("Finished Ingesting.....")
    print("-------------------------------")
    print("\n\n\n")