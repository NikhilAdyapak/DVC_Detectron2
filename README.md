# Pedestrian Detection & Segmentation MLOps Pipeline

## DVC pipeline for Pedestrian Detection & Segmentation using Detectron2 with Data Versioning

<p align="center">
    <img src="screenshots/detectron2_dvc_dag.png" alt="Pipeline screenshot" title="DVC Pipeline" height="500">
</p>


# Getting Started
## 1. Create a Python environment
```shell
python3 -m venv <env_name>
source <env_name>/bin/activate
```

## 2. To initialize DVC and GIT
```shell
pip3 install dvc
git init
dvc init
```

## 3. Installing dependencies
To install requirements for running object detection pipeline with Detectron2
Requires PyTorch, CUDA(if GPU Enabled)
```shell
pip3 install -r requirements.txt

Detectron2 Dependencies
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

# On macOS, you may need to prepend the above commands with a few environment variables:
CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" python -m pip install ...

````

## 4. Setting paramenters
Here dcount is the number of versions of datasets uploaded(dcount = 0 when initialised)
```
# file params.yaml
ingest:
    dcount:0
```

## 5. Pipeline DAG
```shell
dvc dag
```

### 6. To run DVC pipeline
```shell
dvc repro
```

## 7. Adding pipeline stage

```shell
dvc run -n <Stage_name> 
    -p ingest.dcount -p <add parameters> 
    -d src/<Stage_file>.py -d <Any dependencies>
    -o data/<Output dir> 
    python3 src/prepare.py
```