FROM python:3.9

WORKDIR /src

COPY ./requirements.txt /src/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /src/requirements.txt

COPY ./src/main.py /src/main.py

RUN mkdir /src/models
RUN mkdir /src/audio

COPY ./models/my_resnet18.pth /src/models/my_resnet18.pth

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
