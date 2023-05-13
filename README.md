# VisionMakers-ChatGPT-api

## Create environment
```bash
conda create -n visionmakers python=3.9
conda activate visionmakers
```

## Install requirements
```bash
pip install -r requirements.txt
```

## Run server locally
```bash
cd src
uvicorn main:app --reload
```

## Deploy
```bash
cd src
uvicorn main:app --host 0.0.0.0 --port 80
```