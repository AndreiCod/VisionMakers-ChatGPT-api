import os
import shutil
from typing import List, Dict
from collections import OrderedDict

import openai
from elevenlabs import generate, save, set_api_key

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from dotenv import dotenv_values

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from PIL import Image
import numpy as np

app = FastAPI()

config = dotenv_values(".env")

openai.organization = config["OPENAI_ORGANIZATION"]
openai.api_key = config["OPENAI_API_SECRET"]

chat_config: List[Dict[str, str]] = [{"role": "system", "content": config["GPT_SYSTEM_PROMPT"]}]
chat_history: List[Dict[str, str]] = chat_config.copy()

def load_model():
    resnet18 = torchvision.models.resnet18(pretrained=False)

    fc = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(512, 128)),
        ('dropout', nn.Dropout(p=.5)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(128, 4)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    resnet18.fc = fc

    device = torch.device("cpu")
    resnet18 = resnet18.to(device)

    resnet18.load_state_dict(torch.load(config["MODEL_PATH"], map_location=torch.device('cpu')))
    resnet18.eval()

    return resnet18

resnet18 = load_model()


set_api_key(config["ELEVENLABS_API_KEY"])


@app.get("/chat/generate_response/")
def chat_response(message: str) -> str:
    chat_history.append({"role": "user", "content": message})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat_history,
    )

    chat_history.append(response.choices[0].message)

    return response.choices[0].message.content

@app.get("/chat/get_last_message")
def get_last_message() -> str:
    return chat_history[-1]["content"]

@app.get("/chat/get_last_message_audio")
def get_last_message_audio() :
    audio = generate(text=chat_history[-1]["content"], voice="Arnold")

    save(audio, "./audio/audio.wav")

    return FileResponse("./audio/audio.wav")


@app.get("/chat/history")
def get_chat_history() -> List[Dict[str, str]]:
    return chat_history

@app.delete("/chat/reset")
def reset_chat() -> str:
    chat_history.clear()
    chat_history.extend(chat_config)

    return "Chat history reset"

@app.get("/image/label")
def get_image_label(file: UploadFile) -> str:
    sample = Image.open(file.file)
    sample = sample.convert('RGB')
    sample = sample.resize((256, 256))

    sample_ = sample.copy()
    sample_ = np.array(sample_)

    sample_ = sample_ / sample_.max()
    sample_ = sample_.transpose(2, 0, 1)
    sample_ = torch.Tensor(sample_)
    sample_ = sample_.unsqueeze(0)

    predict = resnet18(sample_.to("cpu"))
    probability = F.softmax(predict, dim=1)
    top_probability, top_class = probability.topk(1, dim=1)

    print(f'sample {file.filename} got a predict score of {top_probability[0][0]}')
    if str(torch.argmax(predict.cpu(), axis=1)) == 'tensor([0])':
        predict = "bueno"
    elif str(torch.argmax(predict.cpu(), axis=1)) == 'tensor([1])':
        predict = "cola"
    elif str(torch.argmax(predict.cpu(), axis=1)) == 'tensor([2])':
        predict = "doritos"
    elif str(torch.argmax(predict.cpu(), axis=1)) == 'tensor([3])':
        predict = "mnm"
    
    return predict


@app.get("/speech-to-text")
def speech_to_text(file: UploadFile) -> str:
    with open(f"./audio/{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    audio_file = open(f"./audio/{file.filename}", "rb")

    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    return transcript.text