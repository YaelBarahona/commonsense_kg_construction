import os
from pathlib import Path

import requests
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_FILE)

NEBULA_KEY = os.getenv("NEBULA_KEY")

def call_llm(prompt: str) -> str:
    if not NEBULA_KEY:
        raise ValueError(f"NEBULA_KEY not found. Expected in: {ENV_FILE}")

    url = "https://nebula.cs.vu.nl/api/chat/completions"

    headers = {
        "Authorization": f"Bearer {NEBULA_KEY.strip()}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "deepseek-r1:8b",
        "messages": [
            {
                "role": "system",
                "content": "You are a semantic reasoning system using FrameNet."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")

    return response.json()["choices"][0]["message"]["content"]