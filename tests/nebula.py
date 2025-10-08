import os
from dotenv import load_dotenv 

import requests

load_dotenv() 
NEBULA_KEY = os.getenv("NEBULA_KEY")

def chat_with_model():
    url = 'https://nebula.cs.vu.nl/api/chat/completions'
    headers = {
        'Authorization': f'Bearer {NEBULA_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": 'gpt-oss:20b',
        "messages": [
            {
                "role": "system",
                "content": "You are a knowledge engineer working on Peter Gardenfors's theory of conceptual spaces."
            },
            {
                "role": "user",
                "content": "What are the main domains and their corresponding quality dimensions that exist?"
            }
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    print(response)
    return response.json()['choices'][0]['message']['content']

print(chat_with_model())