import os
from typing import List, Dict

import openai
from elevenlabs import generate, save, set_api_key

from fastapi import FastAPI
from fastapi.responses import FileResponse
from dotenv import dotenv_values

app = FastAPI()

config = dotenv_values(".env")

openai.organization = config["OPENAI_ORGANIZATION"]
openai.api_key = config["OPENAI_API_SECRET"]

chat_config: List[Dict[str, str]] = [{"role": "system", "content": config["GPT_SYSTEM_PROMPT"]}]
chat_history: List[Dict[str, str]] = chat_config.copy()


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

    save(audio, "../audio/audio.wav")

    return FileResponse("../audio/audio.wav")


@app.get("/chat/history")
def get_chat_history() -> List[Dict[str, str]]:
    return chat_history

@app.delete("/chat/reset")
def reset_chat() -> str:
    chat_history.clear()
    chat_history.extend(chat_config)

    return "Chat history reset"