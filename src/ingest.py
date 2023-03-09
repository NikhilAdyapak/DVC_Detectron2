import os,sys
from distutils.dir_util import copy_tree
import yaml

if len(sys.argv) != 2:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write(
            '\tpython3 src/ingest.py data/prepared\n'
        )
        sys.exit(1)


if __name__ == "__main__":

    params = yaml.safe_load(open('params.yaml'))

    data_path = os.path.join(sys.argv[1], f"v{params['ingest']['dcount']}")
    print(data_path)

    input_dir = params["dataset"]["path"]
    os.makedirs(data_path, exist_ok = True)

    print("-------------------------------")
    print("Ingesting.....")
    print("-------------------------------")

    copy_tree(input_dir, data_path)