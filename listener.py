import os,sys,glob
import yaml,shutil
import json

base_path = "/home/yln1kor/Blob"

latest = max(glob.glob(os.path.join(base_path, '*/')), key=os.path.getmtime)

params = yaml.safe_load(open('params.yaml'))

# if latest != os.path.join(params['latest-run']['folder']):
#     print(latest, params['latest-run']['folder'])
# else:
#     print("ok")

if latest != os.path.join(params['latest-run']['folder']):
    cmd = "dvc repro"
    os.system(cmd)

    dcount = params['ingest']['dcount'] + 1
    params.pop("ingest")
    params.pop("latest-run")
    params.update({"ingest":{"dcount":dcount}})
    params.update({"latest-run":{"folder":latest}})

    with open('params.yaml', 'w') as file:
        outputs = yaml.dump(params, file)

else:
    print("\n\nNo new update\n\n")

