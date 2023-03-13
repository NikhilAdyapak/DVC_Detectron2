import yaml,shutil


if len(sys.argv) != 4:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 src/compare.py data/infer data/predict data/store\n'
    )
    sys.exit(1)


params = yaml.safe_load(open('params.yaml'))

infer_path = os.path.join(sys.argv[1],f"v{params['ingest']['dcount']}")
predict_path = os.path.join(sys.argv[2],f"v{params['ingest']['dcount']}")
store_path = os.path.join(sys.argv[3],f"v{params['ingest']['dcount']}")
os.makedirs(store_path, exist_ok = True)


