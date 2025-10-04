from groq import Groq
# import instructor
from .base_client import LLMClient

class GroqClient(LLMClient):
    def __init__(self, api_key: str, model_name: str, use_instructor: bool = False, response_model=None):
        self.api_key = api_key
        self.model_name = model_name
        self.use_instructor = use_instructor
        self.response_model = response_model

        raw_client = Groq(api_key=api_key)
        self.client = raw_client #instructor.from_groq(raw_client, mode=instructor.Mode.JSON) if use_instructor else raw_client
        # self.client = instructor.from_groq(raw_client, mode=instructor.Mode.JSON) if use_instructor else raw_client

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if self.use_instructor:
            if not self.response_model:
                raise ValueError("response_model must be provided when using instructor mode.")
            return self.client.chat.completions.create(
                model=self.model_name,
                response_model=self.response_model,
                messages=messages,
                temperature=0.65
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.65
            )
            return response.choices[0].message.content
