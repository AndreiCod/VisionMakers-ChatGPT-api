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
uvicorn src.main:app --reload
```

## Deploy
```bash
uvicorn src.main:app --host 0.0.0.0 --port 80
```

## Create docker image
```bash
docker build -t vision-makers .
```

## Run docker image
```bash
docker run -d --name vision-makers-api -p 80:80 vision-makers
```