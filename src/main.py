import os
from typing import List, Dict

import openai
from fastapi import FastAPI
from dotenv import dotenv_values

app = FastAPI()

config = dotenv_values(".env")

openai.organization = config["OPENAI_ORGANIZATION"]
openai.api_key = config["OPENAI_API_SECRET"]

chat_config: List[Dict[str, str]] = [{"role": "system", "content": config["GPT_SYSTEM_PROMPT"]}]
chat_history: List[Dict[str, str]] = chat_config.copy()

@app.get("/chat/")
def chat_completition(message: str) -> None:
    chat_history.append({"role": "user", "content": message})
    print(chat_history)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat_history,
    )

    chat_history.append(response.choices[0].message)

    return response.choices[0].message.content

@app.get("/chat/reset")
def reset_chat() -> None:
    chat_history = chat_config.copy()

    return chat_history